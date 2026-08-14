import { at } from "@/collections"
import type { Fields } from "@/fields"
import { createProgramAdapter } from "@/gepa/adapter"
import type { GEPAMetric, ReflectionLM } from "@/gepa/adapter"
import { aggregateScore, argmax, runGEPA } from "@/gepa/engine"
import type { Candidate, GEPAState } from "@/gepa/engine"
import type { Example, Program } from "@/program"
import { createRNG } from "@/random"

// --- Budget -----------------------------------------------------------------

const AUTO_CANDIDATES = { heavy: 18, light: 6, medium: 12 } as const

export type GEPAAuto = keyof typeof AUTO_CANDIDATES

/**
 * DSPy's auto-budget estimate. `fullEvalSteps` (m) exists only here — the
 * engine has no periodic full-eval scheduling.
 */
export const autoBudget = (
  numPreds: number,
  numCandidates: number,
  valsetSize: number,
  minibatchSize = 35,
  fullEvalSteps = 5
): number => {
  const numTrials = Math.floor(
    Math.max(2 * (numPreds * 2) * Math.log2(numCandidates), 1.5 * numCandidates)
  )
  if (numTrials < 0 || valsetSize < 0 || minibatchSize < 0) {
    throw new Error("autoBudget arguments must be non-negative")
  }
  if (fullEvalSteps < 1) {
    throw new Error("fullEvalSteps must be >= 1")
  }
  let total = valsetSize + numCandidates * 5 + numTrials * minibatchSize
  if (numTrials === 0) {
    return total
  }
  total +=
    (Math.floor((numTrials + 1) / fullEvalSteps) +
      1 +
      (numTrials < fullEvalSteps ? 1 : 0)) *
    valsetSize
  return total
}

// --- Configuration ----------------------------------------------------------

export type GEPAConfig<TInput = Fields, TOutput = Fields> = {
  addFormatFailureAsFeedback?: boolean
  /** Exactly one of `auto`, `maxFullEvals`, `maxMetricCalls` must be set. */
  auto?: GEPAAuto
  candidateSelectionStrategy?: "currentBest" | "pareto"
  componentSelector?: "all" | "roundRobin"
  failureScore?: number
  maxFullEvals?: number
  maxMergeInvocations?: number
  maxMetricCalls?: number
  metric: GEPAMetric<TInput, TOutput>
  perfectScore?: number
  reflectionLM: ReflectionLM
  reflectionMinibatchSize?: number
  seed?: number
  skipPerfectScore?: boolean
  useMerge?: boolean
  valset?: Example<TInput, TOutput>[]
  warnOnScoreMismatch?: boolean
}

const VALSET_SIZE_NOTE = 35

const resolveBudget = (
  config: Pick<GEPAConfig, "auto" | "maxFullEvals" | "maxMetricCalls"> & {
    valset?: readonly unknown[]
  },
  numPreds: number,
  trainset: readonly unknown[],
  valsetSize: number
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
    // DSPy quirk: when no valset was given the multiplier is len(trainset)
    // only, even though valset then defaults to trainset.
    return (
      config.maxFullEvals * (trainset.length + (config.valset?.length ?? 0))
    )
  }
  if (config.auto === undefined) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
    )
  }
  return autoBudget(
    Math.max(numPreds, 1),
    AUTO_CANDIDATES[config.auto],
    valsetSize
  )
}

// --- Result -----------------------------------------------------------------

export type GEPAResult<TInput, TOutput> = {
  /** argmax of valAggregateScores, lowest index winning ties. */
  bestIdx: number
  candidates: Candidate[]
  discoveryEvalCounts: number[]
  numFullValEvals: number
  parents: (number | null)[][]
  perValInstanceBestCandidates: Map<number, Set<number>>
  program: Program<TInput, TOutput>
  seed: number
  totalMetricCalls: number
  valAggregateScores: number[]
  valSubscores: number[][]
}

export const buildResult = <TInput, TOutput>(
  state: GEPAState,
  valsetSize: number,
  seed: number,
  buildProgram: (candidate: Candidate) => Program<TInput, TOutput>
): GEPAResult<TInput, TOutput> => {
  const valAggregateScores = state.progCandidateValSubscores.map(aggregateScore)
  const bestIdx = argmax(valAggregateScores)
  return {
    bestIdx,
    candidates: state.programCandidates,
    discoveryEvalCounts: state.numMetricCallsByDiscovery,
    numFullValEvals: state.numFullDSEvals,
    parents: state.parentProgramForCandidate,
    perValInstanceBestCandidates: state.programAtParetoFrontValset,
    program: buildProgram(at(state.programCandidates, bestIdx, "candidates")),
    seed,
    totalMetricCalls: state.totalNumEvals,
    valAggregateScores,
    valSubscores: Array.from(
      { length: state.programCandidates.length },
      (_, idx) =>
        Array.from(
          { length: valsetSize },
          (_2, valId) =>
            at(
              state.progCandidateValSubscores,
              idx,
              "candidate val subscores"
            ).get(valId) ?? Number.NaN
        )
    ),
  }
}

// --- Entry point ------------------------------------------------------------

export const gepa = async <TInput, TOutput>(
  student: Program<TInput, TOutput>,
  trainset: Example<TInput, TOutput>[],
  config: GEPAConfig<TInput, TOutput>
): Promise<GEPAResult<TInput, TOutput>> => {
  if (trainset.length === 0) {
    throw new Error("GEPA requires a non-empty trainset")
  }
  const valset = config.valset ?? trainset
  if (!config.valset) {
    console.warn("GEPA: no valset provided; using the trainset for validation.")
  }
  if (valset.length > VALSET_SIZE_NOTE) {
    console.warn(
      `GEPA: valset has ${valset.length} examples; every accepted candidate costs a full valset eval.`
    )
  }

  const program = student
  // Instruction text only, like upstream; the student's demos stay on its
  // predictors and reach every candidate through buildProgram's clone.
  const seedCandidate: Candidate = Object.fromEntries(
    program.predictors.map((p) => [p.name, p.instructions])
  )
  const numPreds = Object.keys(seedCandidate).length
  const maxMetricCalls = resolveBudget(
    config,
    numPreds,
    trainset,
    valset.length
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
    reflectionLM: config.reflectionLM,
    warnOnScoreMismatch: config.warnOnScoreMismatch ?? true,
  })

  const state = await runGEPA(adapter, {
    candidateSelectionStrategy: config.candidateSelectionStrategy ?? "pareto",
    componentSelector: config.componentSelector ?? "roundRobin",
    maxMergeInvocations: config.maxMergeInvocations ?? 5,
    maxMetricCalls,
    perfectScore: config.perfectScore ?? 1,
    reflectionMinibatchSize: config.reflectionMinibatchSize ?? 3,
    rng: engineRNG,
    seedCandidate,
    skipPerfectScore: config.skipPerfectScore ?? true,
    trainset,
    useMerge: config.useMerge ?? true,
    valset,
  })

  return buildResult(state, valset.length, seed, adapter.buildProgram)
}
