import type { LanguageModel } from "ai"

import { at, last } from "@/collections"
import type { Fields } from "@/fields"
import type { MetricOutput } from "@/metric"
import type { Demo, RunContext, TraceStep } from "@/predictor"
import type { Example, Program } from "@/program"
import { createRNG, sample, shuffle } from "@/random"

/**
 * Faithful port of dspy.teleprompt.bootstrap.BootstrapFewShot and its
 * LabeledFewShot dependency (dspy/dspy/teleprompt/{bootstrap,vanilla}.py).
 * Demos land on the returned program's predictors; compose with GEPA/SIMBA by
 * running this first and optimizing the demo-carrying program.
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
  trace: TraceStep[]
) => MetricOutput

export type BootstrapConfig<TInput = Fields, TOutput = Fields> = {
  maxBootstrappedDemos?: number
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number
  maxLabeledDemos?: number
  maxRounds?: number
  metric?: BootstrapMetric<TInput, TOutput>
  metricThreshold?: number
  teacher?: Program<TInput, TOutput>
  teacherSettings?: { model?: LanguageModel; temperature?: number }
}

/** dspy.settings.max_errors default. */
const DEFAULT_MAX_ERRORS = 10

/** A labeled example rendered as a demo: inputs in, expected fields out. */
const exampleToDemo = (example: Example): Demo => ({
  inputs: structuredClone(example.inputs),
  outputs: structuredClone(example.outputs),
})

const demoMatchesExample = (demo: Demo, example: Example): boolean =>
  JSON.stringify(demo.inputs) === JSON.stringify(example.inputs) &&
  JSON.stringify(demo.outputs) === JSON.stringify(example.outputs)

/** FNV-1a over the JSON rendering — stands in for dspy's Hasher.hash. */
const contentHash = (value: Demo[]): number => {
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

/** Reset copy: fresh clone with predictor demos cleared (dspy reset_copy). */
const resetCopy = <TInput, TOutput>(
  program: Program<TInput, TOutput>
): Program<TInput, TOutput> => {
  const copy = program.clone()
  for (const predictor of copy.predictors) {
    predictor.demos = []
  }
  return copy
}

/**
 * dspy.teleprompt.vanilla.LabeledFewShot: install k labeled examples as demos
 * on a reset copy of the student. Each predictor draws its own sample from the
 * same seed-0 RNG stream; nothing else is shuffled.
 */
export const labeledFewShot = <TInput extends Fields, TOutput extends Fields>(
  student: Program<TInput, TOutput>,
  trainset: Example<TInput, TOutput>[],
  k = 16
): Program<TInput, TOutput> => {
  const compiled = resetCopy(student)
  if (trainset.length === 0) {
    return compiled
  }
  const rng = createRNG(0)
  for (const predictor of compiled.predictors) {
    predictor.demos = sample(rng, trainset, Math.min(k, trainset.length)).map(
      exampleToDemo
    )
  }
  return compiled
}

/**
 * BootstrapFewShot.compile: run a teacher over the trainset, capture the trace
 * of every metric-passing run as `augmented` demos per predictor, and fill the
 * remaining demo slots with raw labeled examples.
 */
export const bootstrapFewShot = async <
  TInput extends Fields,
  TOutput extends Fields,
>(
  studentProgram: Program<TInput, TOutput>,
  trainset: Example<TInput, TOutput>[],
  config: BootstrapConfig<TInput, TOutput> = {}
): Promise<Program<TInput, TOutput>> => {
  const {
    maxBootstrappedDemos = 4,
    maxErrors = DEFAULT_MAX_ERRORS,
    maxLabeledDemos = 16,
    maxRounds = 1,
    metric,
    metricThreshold,
    teacherSettings,
  } = config

  // _prepare_student_and_teacher: reset copy for the student, deep copy for
  // the teacher — then LabeledFewShot over a reset teacher copy when labeled
  // demos are requested (our programs carry no _compiled flag; a provided
  // teacher is treated as uncompiled, see the doc's deviation list).
  const student = resetCopy(studentProgram)
  let teacher = (config.teacher ?? studentProgram).clone()
  if (maxLabeledDemos > 0) {
    teacher = labeledFewShot(teacher, trainset, maxLabeledDemos)
  }

  // _prepare_predictor_mappings: same structure, matched by position + name.
  if (student.predictors.length !== teacher.predictors.length) {
    throw new Error(
      "Student and teacher must have the same number of predictors."
    )
  }
  for (const [idx, studentPredictor] of student.predictors.entries()) {
    const teacherPredictor = teacher.predictors[idx]
    if (studentPredictor.name !== teacherPredictor?.name) {
      throw new Error(
        "Student and teacher must have the same program structure."
      )
    }
    if (studentPredictor === teacherPredictor) {
      throw new Error("Student and teacher must be different objects.")
    }
  }
  const predictorNames = student.predictors.map((p) => p.name)

  // _bootstrap ---------------------------------------------------------------
  const name2traces = new Map<string, Demo[]>(
    predictorNames.map((name) => [name, []])
  )
  let errorCount = 0

  const runTeacherAttempt = async (
    example: Example<TInput, TOutput>,
    roundIdx: number,
    trace: TraceStep[]
  ): Promise<TOutput | null> => {
    // Rounds past the first take a fresh rollout at temperature=1.0 to
    // bypass caches — the rollout id maps onto the seed parameter, exactly
    // like SIMBA's prepareModelsForResampling.
    const ctx: RunContext = {
      model: teacherSettings?.model,
      temperature: teacherSettings?.temperature,
      trace,
    }
    if (roundIdx > 0) {
      ctx.seed = roundIdx
      ctx.temperature = 1
    }
    // Hide any demo equal to the example being bootstrapped, restore after.
    const demoCache = teacher.predictors.map((p) => p.demos)
    for (const predictor of teacher.predictors) {
      predictor.demos = predictor.demos.filter(
        (demo) => !demoMatchesExample(demo, example)
      )
    }
    try {
      return await teacher.run(example.inputs, ctx)
    } finally {
      for (const [idx, predictor] of teacher.predictors.entries()) {
        predictor.demos = at(demoCache, idx, "teacher demo cache")
      }
    }
  }

  const harvestTraceDemos = (trace: TraceStep[]): void => {
    const demosByName = new Map<string, Demo[]>()
    for (const step of trace) {
      if (!name2traces.has(step.predictorName)) {
        continue
      }
      const demo: Demo = {
        augmented: true,
        inputs: step.inputs,
        outputs: step.outputs,
      }
      const list = demosByName.get(step.predictorName) ?? []
      list.push(demo)
      demosByName.set(step.predictorName, list)
    }
    for (const [name, demos] of demosByName) {
      let kept = demos
      // Multiple traces for one predictor in one example: keep ONE, sampled
      // 50/50 from the first N-1 or the last, seeded by demo content.
      if (demos.length > 1) {
        const rng = createRNG(contentHash(demos))
        kept = [
          rng() < 0.5
            ? at(demos, Math.floor(rng() * (demos.length - 1)), "trace demos")
            : last(demos, "trace demos"),
        ]
      }
      name2traces.get(name)?.push(...kept)
    }
  }

  const bootstrapOneExample = async (
    example: Example<TInput, TOutput>,
    roundIdx: number
  ): Promise<boolean> => {
    const trace: TraceStep[] = []
    let success = false
    try {
      const prediction = await runTeacherAttempt(example, roundIdx, trace)
      if (metric) {
        const { score } = await metric(example, prediction, trace)
        success =
          metricThreshold === undefined ? score > 0 : score >= metricThreshold
      } else {
        success = true
      }
    } catch (error) {
      success = false
      errorCount += 1
      if (errorCount >= maxErrors) {
        throw error
      }
      console.error(`Failed to run or evaluate example due to ${error}.`)
    }

    if (success) {
      harvestTraceDemos(trace)
    }
    return success
  }

  const bootstrapped = new Set<number>()
  for (const [exampleIdx, example] of trainset.entries()) {
    if (bootstrapped.size >= maxBootstrappedDemos) {
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

  // Un-bootstrapped examples become the labeled-demo pool, seed-0 shuffled.
  const validation = trainset.filter((_x, idx) => !bootstrapped.has(idx))
  shuffle(createRNG(0), validation)

  // _train: augmented demos first, labeled backfill after — preserving the
  // Python quirk that rawDemos is REASSIGNED to each predictor's sample, so
  // later predictors draw from the shrinking pool.
  const rng = createRNG(0)
  let rawDemos = validation
  for (const predictor of student.predictors) {
    const augmented = (name2traces.get(predictor.name) ?? []).slice(
      0,
      maxBootstrappedDemos
    )
    const sampleSize = Math.max(
      0,
      Math.min(maxLabeledDemos - augmented.length, rawDemos.length)
    )
    rawDemos = sample(rng, rawDemos, sampleSize)
    predictor.demos = [...augmented, ...rawDemos.map(exampleToDemo)]
  }

  return student
}
