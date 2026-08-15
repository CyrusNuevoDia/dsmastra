import type { LanguageModel } from "ai"

import { at, last } from "../collections"
import type { Fields } from "../fields"
import type { Example, Program } from "../program"
import { createRNG, sample, shuffle } from "../random"
import type { MetricOutput } from "../scorers"
import type { RunContext, TraceStep } from "../step"
import { isEqualJSON } from "../utils"

/**
 * Shared bootstrap machinery, ported faithfully from
 * dspy.teleprompt.bootstrap.BootstrapFewShot and its LabeledFewShot dependency
 * (dspy/dspy/teleprompt/{bootstrap,vanilla}.py). Program-level and internal:
 * the public entry points in bootstrap-few-shot.ts and labeled-few-shot.ts
 * wrap these for workflows, and GEPA's few-shot pre-pass calls them directly.
 */

/**
 * Deviation from DSPy: upstream's metric returns `bool | float`, ours always
 * returns `{ score }`. A pass/fail metric writes `{ score: 1 }` / `{ score: 0 }`,
 * and with no `metricThreshold` set any score above zero counts as success —
 * value-for-value the same decision upstream's `bool(metric_val)` makes.
 */
export type BootstrapMetric<TInput = Fields, TOutput = Fields> = (
  gold: Example<TInput, TOutput>,
  prediction: TOutput | null,
  trace: TraceStep[],
  /** The attempt's RunContext after the rollout — carries the engine-derived
   * `trajectory` when the teacher runs through Mastra's engine. */
  ctx?: RunContext
) => MetricOutput

export type BootstrapOptions<TInput = Fields, TOutput = Fields> = {
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number
  maxFewShotExamples?: number
  maxLabeledExamples?: number
  maxRounds?: number
  metric?: BootstrapMetric<TInput, TOutput>
  metricThreshold?: number
  teacher?: Program<TInput, TOutput>
  teacherSettings?: { model?: LanguageModel; temperature?: number }
}

/** dspy.settings.max_errors default. */
const DEFAULT_MAX_ERRORS = 10

const cloneExample = (example: Example): Example => ({
  inputData: structuredClone(example.inputData),
  outputData: structuredClone(example.outputData),
})

const examplesEqual = (a: Example, b: Example): boolean =>
  isEqualJSON(a.inputData, b.inputData) &&
  isEqualJSON(a.outputData, b.outputData)

/** FNV-1a over the JSON rendering — stands in for dspy's Hasher.hash. */
const contentHash = (value: Example[]): number => {
  const text = JSON.stringify(value)
  /* oxlint-disable no-bitwise -- FNV-1a is bit-twiddling by design */
  let hash = 0x81_1c_9d_c5
  for (let i = 0; i < text.length; i += 1) {
    // oxlint-disable-next-line unicorn/prefer-code-point -- FNV-1a here is defined over UTF-16 code units; code points would change every hash and so every seeded sample
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01_00_01_93)
  }
  return hash >>> 0
  /* oxlint-enable no-bitwise */
}

/** Reset copy: fresh clone with step examples cleared (dspy reset_copy). */
const resetCopy = <TInput, TOutput>(
  program: Program<TInput, TOutput>
): Program<TInput, TOutput> => {
  const copy = program.clone()
  for (const step of copy.steps) {
    step.examples = []
  }
  return copy
}

/**
 * dspy.teleprompt.vanilla.LabeledFewShot: install k labeled examples as
 * few-shot examples on a reset copy of the student. Each step draws its own
 * sample from the same seed-0 RNG stream; nothing else is shuffled.
 */
export const labeledFewShotProgram = <
  TInput extends Fields,
  TOutput extends Fields,
>(
  student: Program<TInput, TOutput>,
  trainingSet: Example<TInput, TOutput>[],
  k = 16
): Program<TInput, TOutput> => {
  const compiled = resetCopy(student)
  if (trainingSet.length === 0) {
    return compiled
  }
  const rng = createRNG(0)
  for (const step of compiled.steps) {
    step.examples = sample(
      rng,
      trainingSet,
      Math.min(k, trainingSet.length)
    ).map(cloneExample)
  }
  return compiled
}

/**
 * _prepare_student_and_teacher: reset copy for the student, deep copy for
 * the teacher — then LabeledFewShot over a reset teacher copy when labeled
 * examples are requested (our programs carry no _compiled flag; a provided
 * teacher is treated as uncompiled, see the doc's deviation list). Validates
 * _prepare_predictor_mappings: same structure, matched by position + id.
 */
export const prepareStudentAndTeacher = <
  TInput extends Fields,
  TOutput extends Fields,
>(
  studentProgram: Program<TInput, TOutput>,
  trainingSet: Example<TInput, TOutput>[],
  options: Pick<
    BootstrapOptions<TInput, TOutput>,
    "maxLabeledExamples" | "teacher"
  >
) => {
  const { maxLabeledExamples = 16 } = options
  const student = resetCopy(studentProgram)
  let teacher = (options.teacher ?? studentProgram).clone()
  if (maxLabeledExamples > 0) {
    teacher = labeledFewShotProgram(teacher, trainingSet, maxLabeledExamples)
  }
  if (student.steps.length !== teacher.steps.length) {
    throw new Error("Student and teacher must have the same number of steps.")
  }
  for (const [idx, studentStep] of student.steps.entries()) {
    const teacherStep = teacher.steps[idx]
    if (studentStep.id !== teacherStep?.id) {
      throw new Error(
        "Student and teacher must have the same program structure."
      )
    }
    if (studentStep === teacherStep) {
      throw new Error("Student and teacher must be different objects.")
    }
  }
  return { student, teacher }
}

/**
 * One teacher attempt over one example: hide any installed example equal to
 * the one being bootstrapped, roll the teacher out with trace capture, and
 * decide acceptance through the metric. Rounds past the first take a fresh
 * rollout at temperature=1.0 to bypass caches — the round index maps onto the
 * seed parameter, exactly like SIMBA's prepareModelsForResampling. Throws
 * when the rollout or the metric throws; the caller owns error counting.
 */
export const runBootstrapAttempt = async <
  TInput extends Fields,
  TOutput extends Fields,
>(
  teacher: Program<TInput, TOutput>,
  example: Example<TInput, TOutput>,
  roundIdx: number,
  options: Pick<
    BootstrapOptions<TInput, TOutput>,
    "metric" | "metricThreshold" | "teacherSettings"
  >
): Promise<{ success: boolean; trace: TraceStep[] }> => {
  const { metric, metricThreshold, teacherSettings } = options
  const trace: TraceStep[] = []
  const ctx: RunContext = {
    model: teacherSettings?.model,
    temperature: teacherSettings?.temperature,
    trace,
  }
  if (roundIdx > 0) {
    ctx.seed = roundIdx
    ctx.temperature = 1
  }
  // Hide any example equal to the one being bootstrapped, restore after.
  const exampleCache = teacher.steps.map((step) => step.examples)
  for (const step of teacher.steps) {
    step.examples = step.examples.filter(
      (installed) => !examplesEqual(installed, example)
    )
  }
  let prediction: TOutput | null
  try {
    prediction = await teacher.run(example.inputData, ctx)
  } finally {
    for (const [idx, step] of teacher.steps.entries()) {
      step.examples = at(exampleCache, idx, "teacher example cache")
    }
  }
  if (!metric) {
    return { success: true, trace }
  }
  const { score } = await metric(example, prediction, trace, ctx)
  return {
    success:
      metricThreshold === undefined ? score > 0 : score >= metricThreshold,
    trace,
  }
}

/**
 * Fold one accepted attempt's trace into the per-step example pools. Multiple
 * traces for one step in one example: keep ONE, sampled 50/50 from the first
 * N-1 or the last, seeded by example content.
 */
export const harvestTraceExamples = (
  id2traces: Map<string, Example[]>,
  trace: TraceStep[]
): void => {
  const examplesById = new Map<string, Example[]>()
  for (const traceStep of trace) {
    if (!id2traces.has(traceStep.stepId)) {
      continue
    }
    const harvested: Example = {
      inputData: traceStep.inputData,
      outputData: traceStep.outputData,
    }
    const list = examplesById.get(traceStep.stepId) ?? []
    list.push(harvested)
    examplesById.set(traceStep.stepId, list)
  }
  for (const [stepId, harvested] of examplesById) {
    let kept = harvested
    if (harvested.length > 1) {
      const rng = createRNG(contentHash(harvested))
      kept = [
        rng() < 0.5
          ? at(
              harvested,
              Math.floor(rng() * (harvested.length - 1)),
              "trace examples"
            )
          : last(harvested, "trace examples"),
      ]
    }
    id2traces.get(stepId)?.push(...kept)
  }
}

/**
 * _train: bootstrapped examples first, labeled backfill after. The
 * un-bootstrapped pool is seed-0 shuffled, and the Python quirk that
 * rawExamples is REASSIGNED to each step's sample is preserved, so later
 * steps draw from the shrinking pool. Mutates and returns `student`.
 */
export const installTrainExamples = <
  TInput extends Fields,
  TOutput extends Fields,
>(
  student: Program<TInput, TOutput>,
  id2traces: Map<string, Example[]>,
  unBootstrapped: Example<TInput, TOutput>[],
  options: Pick<
    Required<BootstrapOptions<TInput, TOutput>>,
    "maxFewShotExamples" | "maxLabeledExamples"
  >
): Program<TInput, TOutput> => {
  const validation = [...unBootstrapped]
  shuffle(createRNG(0), validation)
  const rng = createRNG(0)
  let rawExamples: Example[] = validation
  for (const step of student.steps) {
    const harvested = (id2traces.get(step.id) ?? []).slice(
      0,
      options.maxFewShotExamples
    )
    const sampleSize = Math.max(
      0,
      Math.min(
        options.maxLabeledExamples - harvested.length,
        rawExamples.length
      )
    )
    rawExamples = sample(rng, rawExamples, sampleSize)
    step.examples = [...harvested, ...rawExamples.map(cloneExample)]
  }
  return student
}

/** dspy.settings.max_errors default, shared with the workflow driver. */
export const BOOTSTRAP_DEFAULT_MAX_ERRORS = DEFAULT_MAX_ERRORS

/**
 * BootstrapFewShot.compile: run a teacher over the trainingSet, capture the trace
 * of every metric-passing run as bootstrapped examples per step, and fill the
 * remaining slots with raw labeled examples. The durable workflow driver in
 * bootstrap-few-shot.ts runs the same helpers one attempt per loop iteration.
 */
export const bootstrapFewShotProgram = async <
  TInput extends Fields,
  TOutput extends Fields,
>(
  studentProgram: Program<TInput, TOutput>,
  trainingSet: Example<TInput, TOutput>[],
  options: BootstrapOptions<TInput, TOutput> = {}
): Promise<Program<TInput, TOutput>> => {
  const {
    maxErrors = DEFAULT_MAX_ERRORS,
    maxFewShotExamples = 4,
    maxLabeledExamples = 16,
    maxRounds = 1,
  } = options

  const { student, teacher } = prepareStudentAndTeacher(
    studentProgram,
    trainingSet,
    options
  )

  // _bootstrap ---------------------------------------------------------------
  const id2traces = new Map<string, Example[]>(
    student.steps.map((step) => [step.id, []])
  )
  let errorCount = 0

  const bootstrapOneExample = async (
    example: Example<TInput, TOutput>,
    roundIdx: number
  ): Promise<boolean> => {
    let success = false
    try {
      const attempt = await runBootstrapAttempt(
        teacher,
        example,
        roundIdx,
        options
      )
      ;({ success } = attempt)
      if (success) {
        harvestTraceExamples(id2traces, attempt.trace)
      }
    } catch (error) {
      errorCount += 1
      if (errorCount >= maxErrors) {
        throw error
      }
      console.error(`Failed to run or evaluate example due to ${error}.`)
    }
    return success
  }

  const bootstrapped = new Set<number>()
  for (const [exampleIdx, example] of trainingSet.entries()) {
    if (bootstrapped.size >= maxFewShotExamples) {
      break
    }
    for (let roundIdx = 0; roundIdx < maxRounds; roundIdx += 1) {
      // oxlint-disable-next-line no-await-in-loop -- rounds are inherently sequential
      if (await bootstrapOneExample(example, roundIdx)) {
        bootstrapped.add(exampleIdx)
        break
      }
    }
  }

  return installTrainExamples(
    student,
    id2traces,
    trainingSet.filter((_x, idx) => !bootstrapped.has(idx)),
    { maxFewShotExamples, maxLabeledExamples }
  )
}
