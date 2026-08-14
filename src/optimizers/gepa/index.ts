import { at, first } from "@/collections"
import type { Fields } from "@/fields"
import { exactMatch } from "@/metrics"
import type { Metric } from "@/metrics"
import { bootstrapFewShotProgram } from "@/optimizers/bootstrap"
import { createProgramAdapter } from "@/optimizers/gepa/adapter"
import type { GEPAMetric, ReflectionModel } from "@/optimizers/gepa/adapter"
import { aggregateScore, argmax, runGEPA } from "@/optimizers/gepa/engine"
import type { Candidate, GEPAState } from "@/optimizers/gepa/engine"
import {
  programToWorkflow,
  promptsOf,
  workflowToProgram,
} from "@/optimizers/utils"
import type { SavePrompts } from "@/optimizers/utils"
import type { Example, Program } from "@/program"
import { createRNG } from "@/random"
import type { CommittedWorkflow, StepMap } from "@/workflow"

// --- Budget -----------------------------------------------------------------

const AUTO_CANDIDATES = { heavy: 18, light: 6, medium: 12 } as const

export type GEPAAuto = keyof typeof AUTO_CANDIDATES

/**
 * DSPy's auto-budget estimate. `fullEvalSteps` (m) exists only here — the
 * engine has no periodic full-eval scheduling.
 */
export const autoBudget = (
  stepsCount: number,
  candidates: number,
  validationSetSize: number,
  minibatchSize = 35,
  fullEvalSteps = 5
): number => {
  const trialsCount = Math.floor(
    Math.max(2 * (stepsCount * 2) * Math.log2(candidates), 1.5 * candidates)
  )
  if (trialsCount < 0 || validationSetSize < 0 || minibatchSize < 0) {
    throw new Error("autoBudget arguments must be non-negative")
  }
  if (fullEvalSteps < 1) {
    throw new Error("fullEvalSteps must be >= 1")
  }
  let total = validationSetSize + candidates * 5 + trialsCount * minibatchSize
  if (trialsCount === 0) {
    return total
  }
  total +=
    (Math.floor((trialsCount + 1) / fullEvalSteps) +
      1 +
      (trialsCount < fullEvalSteps ? 1 : 0)) *
    validationSetSize
  return total
}

// --- Configuration ----------------------------------------------------------

type EngineTuning<TInput, TOutput> = {
  addFormatFailureAsFeedback?: boolean
  candidateSelectionStrategy?: "currentBest" | "pareto"
  componentSelector?: "all" | "roundRobin"
  failureScore?: number
  maxMergeInvocations?: number
  perfectScore?: number
  reflectionMinibatchSize?: number
  seed?: number
  skipPerfectScore?: boolean
  useMerge?: boolean
  validationSet?: Example<TInput, TOutput>[]
  warnOnScoreMismatch?: boolean
}

export type GEPAProgramConfig<TInput = Fields, TOutput = Fields> = EngineTuning<
  TInput,
  TOutput
> & {
  /** Exactly one of `auto`, `maxFullEvals`, `maxMetricCalls` must be set. */
  auto?: GEPAAuto
  maxFullEvals?: number
  maxMetricCalls?: number
  metric: GEPAMetric<TInput, TOutput>
  /** Called with each new aggregate-score-best program — checkpointing. */
  onImprovement?: (program: Program<TInput, TOutput>) => Promise<void>
  reflectionModel: ReflectionModel
}

export type GEPAConfig = EngineTuning<Fields, Fields> & {
  /** Exactly one of `auto`, `maxFullEvals`, `maxMetricCalls` must be set. */
  auto?: GEPAAuto
  /** When > 0, a bootstrapFewShot pre-pass installs few-shot examples first. */
  maxFewShotExamples?: number
  /** Labeled backfill cap for the pre-pass; defaults to maxFewShotExamples. */
  maxLabeledExamples?: number
  maxFullEvals?: number
  maxMetricCalls?: number
  /** Defaults to exact match on every expected field. */
  metric?: Metric
  /** LM used to propose new descriptions; defaults to the first step's model. */
  reflectionModel?: ReflectionModel
  savePrompts: SavePrompts
  trainingSet: readonly Example[]
}

const VALIDATION_SET_SIZE_NOTE = 35

const resolveBudget = (
  config: {
    auto?: GEPAAuto
    maxFullEvals?: number
    maxMetricCalls?: number
    validationSet?: readonly unknown[]
  },
  stepsCount: number,
  trainingSet: readonly unknown[],
  validationSetSize: number
): number => {
  const provided = [
    config.auto,
    config.maxFullEvals,
    config.maxMetricCalls,
  ].filter((value) => value !== undefined)
  if (provided.length !== 1) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
    )
  }
  if (config.maxMetricCalls !== undefined) {
    return config.maxMetricCalls
  }
  if (config.maxFullEvals !== undefined) {
    // DSPy quirk: when no validationSet was given the multiplier is
    // len(trainingSet) only, even though validationSet then defaults to it.
    return (
      config.maxFullEvals *
      (trainingSet.length + (config.validationSet?.length ?? 0))
    )
  }
  if (config.auto === undefined) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
    )
  }
  return autoBudget(
    Math.max(stepsCount, 1),
    AUTO_CANDIDATES[config.auto],
    validationSetSize
  )
}

// --- Result -----------------------------------------------------------------

export type GEPAProgramResult<TInput, TOutput> = {
  /** argmax of validationAggregateScores, lowest index winning ties. */
  bestIdx: number
  candidates: Candidate[]
  discoveryEvalCounts: number[]
  validationSetEvalsCount: number
  parents: (number | null)[][]
  perValidationInstanceBestCandidates: Map<number, Set<number>>
  program: Program<TInput, TOutput>
  seed: number
  totalMetricCalls: number
  validationAggregateScores: number[]
  validationSubscores: number[][]
}

export const buildResult = <TInput, TOutput>(
  state: GEPAState,
  validationSetSize: number,
  seed: number,
  buildProgram: (candidate: Candidate) => Program<TInput, TOutput>
): GEPAProgramResult<TInput, TOutput> => {
  const validationAggregateScores =
    state.candidateValidationSubscores.map(aggregateScore)
  const bestIdx = argmax(validationAggregateScores)
  return {
    bestIdx,
    candidates: state.programCandidates,
    discoveryEvalCounts: state.metricCallCountsByDiscovery,
    parents: state.parentProgramForCandidate,
    perValidationInstanceBestCandidates:
      state.programAtParetoFrontValidationSet,
    program: buildProgram(at(state.programCandidates, bestIdx, "candidates")),
    seed,
    totalMetricCalls: state.totalEvalsCount,
    validationAggregateScores,
    validationSetEvalsCount: state.validationSetEvalsCount,
    validationSubscores: Array.from(
      { length: state.programCandidates.length },
      (_, idx) =>
        Array.from(
          { length: validationSetSize },
          (_2, validationId) =>
            at(
              state.candidateValidationSubscores,
              idx,
              "candidate validation subscores"
            ).get(validationId) ?? Number.NaN
        )
    ),
  }
}

// --- Program-level entry point ----------------------------------------------

export const gepaProgram = async <TInput, TOutput>(
  student: Program<TInput, TOutput>,
  trainingSet: Example<TInput, TOutput>[],
  config: GEPAProgramConfig<TInput, TOutput>
): Promise<GEPAProgramResult<TInput, TOutput>> => {
  if (trainingSet.length === 0) {
    throw new Error("GEPA requires a non-empty trainingSet")
  }
  const validationSet = config.validationSet ?? trainingSet
  if (!config.validationSet) {
    console.warn(
      "GEPA: no validationSet provided; using the trainingSet for validation."
    )
  }
  if (validationSet.length > VALIDATION_SET_SIZE_NOTE) {
    console.warn(
      `GEPA: validationSet has ${validationSet.length} examples; every accepted candidate costs a full validationSet eval.`
    )
  }

  const program = student
  // Description text only, like upstream; the student's few-shot examples stay
  // on its steps and reach every candidate through buildProgram's clone.
  const seedCandidate: Candidate = Object.fromEntries(
    program.steps.map((step) => [step.id, step.description])
  )
  const maxMetricCalls = resolveBudget(
    config,
    program.steps.length,
    trainingSet,
    validationSet.length
  )

  const seed = config.seed ?? 0
  const engineRNG = createRNG(seed)
  const adapterRNG = createRNG(seed)

  const adapter = createProgramAdapter({
    adapterRNG,
    addFormatFailureAsFeedback: config.addFormatFailureAsFeedback ?? false,
    failureScore: config.failureScore ?? 0,
    metric: config.metric,
    program,
    reflectionModel: config.reflectionModel,
    warnOnScoreMismatch: config.warnOnScoreMismatch ?? true,
  })

  const { onImprovement } = config
  const state = await runGEPA(adapter, {
    candidateSelectionStrategy: config.candidateSelectionStrategy ?? "pareto",
    componentSelector: config.componentSelector ?? "roundRobin",
    maxMergeInvocations: config.maxMergeInvocations ?? 5,
    maxMetricCalls,
    onImprovement:
      onImprovement &&
      ((candidate) => onImprovement(adapter.buildProgram(candidate))),
    perfectScore: config.perfectScore ?? 1,
    reflectionMinibatchSize: config.reflectionMinibatchSize ?? 3,
    rng: engineRNG,
    seedCandidate,
    skipPerfectScore: config.skipPerfectScore ?? true,
    trainingSet,
    useMerge: config.useMerge ?? true,
    validationSet,
  })

  return buildResult(state, validationSet.length, seed, adapter.buildProgram)
}

// --- Workflow-level entry point ----------------------------------------------

/**
 * Genetic-Pareto reflective prompt evolution. When `maxFewShotExamples` > 0 a
 * BootstrapFewShot pre-pass installs few-shot examples on the steps first
 * (DSPy-style teleprompter composition); its metric calls are not billed to
 * GEPA's budget, matching DSPy.
 */
export const gepa = async <TSteps extends StepMap>(
  workflow: CommittedWorkflow<TSteps>,
  config: GEPAConfig
): Promise<{ score: number; workflow: CommittedWorkflow<TSteps> }> => {
  const {
    maxFewShotExamples = 0,
    maxLabeledExamples,
    metric: configMetric,
    reflectionModel,
    savePrompts,
    trainingSet,
    ...tuning
  } = config
  const metric = configMetric ?? exactMatch
  const examples = [...trainingSet]

  let student = workflowToProgram(workflow)
  if (maxFewShotExamples > 0) {
    student = await bootstrapFewShotProgram(student, examples, {
      maxFewShotExamples,
      // A TOTAL cap per step, so the labeled backfill shares it instead of
      // DSPy's default 16 — unless the caller raises it explicitly.
      maxLabeledExamples: maxLabeledExamples ?? maxFewShotExamples,
      // Same contract on both sides, so the user's metric passes straight through.
      metric: (gold, prediction) => metric(gold, prediction ?? undefined),
    })
  }

  const result = await gepaProgram(student, examples, {
    ...tuning,
    // Any `feedback` the metric reported rides along — GEPA's reflection reads it.
    metric: (gold, prediction) => metric(gold, prediction ?? undefined),
    onImprovement: async (program) => {
      await savePrompts(promptsOf(program))
    },
    reflectionModel:
      reflectionModel ?? first(student.steps, "workflow steps").model,
  })
  await savePrompts(promptsOf(result.program))
  return {
    score: at(
      result.validationAggregateScores,
      result.bestIdx,
      "aggregate scores"
    ),
    workflow: programToWorkflow(result.program, workflow),
  }
}
