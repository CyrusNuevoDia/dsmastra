import { at, get, pop, prop } from "../../collections"
import type { Fields } from "../../fields"
import type { Example } from "../../program"
import { shuffle, weightedChoice, weightedChoiceStrict } from "../../random"
import type { RNG } from "../../random"
import type { TraceStep } from "../../step"

export type { RNG } from "../../random"

/**
 * A candidate is a map of component (step) name → instruction text,
 * exactly upstream's dict[str, str]. Few-shot examples live on the program's steps
 * (bootstrapFewShot pre-pass) and flow through build_program untouched.
 */
export type Candidate = Record<string, string>

export type Trajectory<TInput = Fields, TOutput = Fields> = {
  example: Example<TInput, TOutput>
  prediction: TOutput | null
  score: number
  trace: GEPATraceStep[]
}

/**
 * Trace steps as GEPA sees them. `parseFailure` carries the model's raw
 * response when the output couldn't be parsed into the expected format.
 */
export type GEPATraceStep = TraceStep & {
  parseFailure?: string
}

export type EvaluationBatch<TInput = Fields, TOutput = Fields> = {
  outputData: (TOutput | null)[]
  scores: number[]
  trajectories?: Trajectory<TInput, TOutput>[]
}

export type ReflectiveExample = {
  Inputs: Record<string, string>
  /** A record of output fields, or the verbatim parse-failure string. */
  "Generated Outputs": Record<string, string> | string
  Feedback: string
}

export type ReflectiveDataset = Record<string, ReflectiveExample[]>

/**
 * The engine/adapter seam: the engine owns state, the Pareto frontier, the
 * loop, and merge; the adapter owns everything program- and LM-specific.
 */
export type GEPAAdapter<TInput = Fields, TOutput = Fields> = {
  evaluate: (
    batch: Example<TInput, TOutput>[],
    candidate: Candidate,
    captureTraces: boolean
  ) => Promise<EvaluationBatch<TInput, TOutput>>
  makeReflectiveDataset: (
    candidate: Candidate,
    evalBatch: EvaluationBatch<TInput, TOutput>,
    componentsToUpdate: string[]
  ) => Promise<ReflectiveDataset>
  proposeNewTexts: (
    candidate: Candidate,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ) => Promise<Record<string, string>>
}

export type GEPAState = {
  i: number
  stepIdToUpdateNextForCandidate: number[]
  validationSetEvalsCount: number
  metricCallCountsByDiscovery: number[]
  paretoFrontValidationSet: Map<number, number>
  parentProgramForCandidate: (number | null)[][]
  candidateValidationSubscores: Map<number, number>[]
  programAtParetoFrontValidationSet: Map<number, Set<number>>
  programCandidates: Candidate[]
  totalEvalsCount: number
}

// --- Small helpers ----------------------------------------------------------

export const MERGE_VALIDATION_OVERLAP_FLOOR = 5

const toSubscores = (scores: number[]): Map<number, number> =>
  new Map(scores.map((score, validationId) => [validationId, score]))

/**
 * Per-instance frontier update after a candidate's full validationSet eval. Strict
 * improvement replaces the set; an exact float tie (no epsilon) adds to it.
 */
export const updateParetoFront = (
  front: Map<number, number>,
  frontPrograms: Map<number, Set<number>>,
  candidateIdx: number,
  subscores: Map<number, number>
): void => {
  for (const [validationId, score] of subscores) {
    const prev = front.get(validationId)
    if (prev === undefined || score > prev) {
      front.set(validationId, score)
      frontPrograms.set(validationId, new Set([candidateIdx]))
    } else if (score === prev) {
      frontPrograms.get(validationId)?.add(candidateIdx)
    }
  }
}

/**
 * Register an accepted candidate: snapshot the budget as its discovery count
 * BEFORE billing its full eval, bill it, update the frontier, and inherit
 * `max(parent cursors)` as its round-robin cursor.
 */
const addCandidate = (
  state: GEPAState,
  candidate: Candidate,
  parents: (number | null)[],
  subscores: Map<number, number>,
  validationSetSize: number
): number => {
  const idx = state.programCandidates.length
  state.programCandidates.push(candidate)
  state.parentProgramForCandidate.push(parents)
  state.candidateValidationSubscores.push(subscores)
  state.metricCallCountsByDiscovery.push(state.totalEvalsCount)
  state.stepIdToUpdateNextForCandidate.push(
    Math.max(
      0,
      ...parents
        .filter((p): p is number => p !== null)
        .map((p) => at(state.stepIdToUpdateNextForCandidate, p, "step cursors"))
    )
  )
  state.totalEvalsCount += validationSetSize
  state.validationSetEvalsCount += 1
  updateParetoFront(
    state.paretoFrontValidationSet,
    state.programAtParetoFrontValidationSet,
    idx,
    subscores
  )
  return idx
}

const sum = (values: number[]): number => values.reduce((acc, v) => acc + v, 0)

/** Mean of a candidate's per-instance scores; -Infinity if unevaluated. */
export const aggregateScore = (subscores: Map<number, number>): number => {
  if (subscores.size === 0) {
    return Number.NEGATIVE_INFINITY
  }
  return sum([...subscores.values()]) / subscores.size
}

/** argmax with lowest-index-wins ties. */
export const argmax = (values: number[]): number => {
  let best = 0
  for (let i = 1; i < values.length; i += 1) {
    if (at(values, i, "argmax values") > at(values, best, "argmax values")) {
      best = i
    }
  }
  return best
}

// --- Pareto frontier --------------------------------------------------------

/**
 * Dominance filter: a candidate survives iff it is the sole occupant of at
 * least one instance's front. Scan candidates ascending by aggregate score,
 * mark one dominated when every front containing it also holds another
 * non-dominated candidate, and restart after each removal — so ties resolve
 * toward the higher aggregate scorer.
 */
export const removeDominatedPrograms = (
  frontPrograms: Map<number, Set<number>>,
  aggScores: number[]
): number[] => {
  const members = new Set<number>()
  for (const set of frontPrograms.values()) {
    for (const idx of set) {
      members.add(idx)
    }
  }
  // Stable sort by score alone: exact-score ties keep front-iteration
  // encounter order, matching Python's sorted(programs, key=score).
  const ascending = [...members].toSorted(
    (a, b) =>
      (aggScores[a] ?? Number.NEGATIVE_INFINITY) -
      (aggScores[b] ?? Number.NEGATIVE_INFINITY)
  )
  const dominated = new Set<number>()
  let changed = true
  while (changed) {
    changed = false
    for (const candidate of ascending) {
      if (dominated.has(candidate)) {
        continue
      }
      const fronts = [...frontPrograms.values()].filter((set) =>
        set.has(candidate)
      )
      const coveredEverywhere = fronts.every((set) =>
        [...set].some((other) => other !== candidate && !dominated.has(other))
      )
      if (coveredEverywhere) {
        dominated.add(candidate)
        changed = true
        break
      }
    }
  }
  return [...members]
    .filter((idx) => !dominated.has(idx))
    .toSorted((a, b) => a - b)
}

/**
 * Pareto parent sampling: dominance-filter the frontier, then pick a survivor
 * with probability proportional to how many instance fronts it occupies.
 */
export const selectParetoParent = (
  frontPrograms: Map<number, Set<number>>,
  aggScores: number[],
  rng: RNG
): number => {
  const survivors = removeDominatedPrograms(frontPrograms, aggScores)
  const frequencies = survivors.map(
    (idx) => [...frontPrograms.values()].filter((set) => set.has(idx)).length
  )
  return weightedChoice(rng, survivors, frequencies)
}

// --- Minibatch sampler ------------------------------------------------------

/** The epoch sampler's whole world between draws — JSON, for durable runs. */
export type EpochSamplerState = {
  epoch: number
  shuffled: number[]
}

export const initEpochSamplerState = (): EpochSamplerState => ({
  epoch: -1,
  shuffled: [],
})

/**
 * Epoch-shuffled minibatch draw: shuffle the train ids once per epoch, pad
 * the tail to a multiple of bsize with the least-frequently-used id, and serve
 * sequential windows keyed by iteration number. Mutates `samplerState` in
 * place; the state plus the RNG's checkpoint fully determine the stream.
 */
export const sampleEpochShuffled = (
  rng: RNG,
  trainSize: number,
  bsize: number,
  samplerState: EpochSamplerState,
  iteration: number
): number[] => {
  const reshuffle = () => {
    const shuffled = Array.from({ length: trainSize }, (_, i) => i)
    shuffle(rng, shuffled)
    // Counter insertion order = first-occurrence order in the shuffled list;
    // Python pads with most_common()[::-1][0][0] — among the least-frequent
    // ids, the one whose first occurrence is LATEST.
    const counts = new Map<number, number>()
    for (const id of shuffled) {
      counts.set(id, (counts.get(id) ?? 0) + 1)
    }
    while (shuffled.length % bsize !== 0) {
      let pick = -1
      let pickCount = Number.POSITIVE_INFINITY
      for (const [id, count] of counts) {
        if (count <= pickCount) {
          pick = id
          pickCount = count
        }
      }
      shuffled.push(pick)
      counts.set(pick, pickCount + 1)
    }
    samplerState.shuffled = shuffled
  }

  const base = iteration * bsize
  const currEpoch =
    samplerState.epoch === -1
      ? 0
      : Math.floor(base / Math.max(samplerState.shuffled.length, 1))
  if (samplerState.shuffled.length === 0 || currEpoch > samplerState.epoch) {
    samplerState.epoch = currEpoch
    reshuffle()
  }
  const start = base % samplerState.shuffled.length
  return samplerState.shuffled.slice(start, start + bsize)
}

/** Closure form of sampleEpochShuffled, for in-memory (non-durable) drivers. */
export const createEpochShuffledSampler = (
  rng: RNG,
  trainSize: number,
  bsize: number
): ((iteration: number) => number[]) => {
  const samplerState = initEpochSamplerState()
  return (iteration: number): number[] =>
    sampleEpochShuffled(rng, trainSize, bsize, samplerState, iteration)
}

// --- Merge (crossover) ------------------------------------------------------

/** Transitive closure over parent lineage (excludes the candidate itself). */
export const findAncestors = (
  parents: (number | null)[][],
  idx: number
): Set<number> => {
  const ancestors = new Set<number>()
  const stack = (parents[idx] ?? []).filter((p): p is number => p !== null)
  while (stack.length > 0) {
    const current = pop(stack, "ancestor stack")
    if (ancestors.has(current)) {
      continue
    }
    ancestors.add(current)
    stack.push(
      ...(parents[current] ?? []).filter((p): p is number => p !== null)
    )
  }
  return ancestors
}

/**
 * Per-component merge rule: agreeing descendants win outright; a lone
 * divergence from the ancestor wins; a double divergence goes to the
 * higher-aggregate descendant, coin-flipped on an exact tie.
 */
export const buildMergedCandidate = (
  ancestor: Candidate,
  descendantI: Candidate,
  descendantJ: Candidate,
  aggI: number,
  aggJ: number,
  rng: RNG
) => {
  const merged = { ...ancestor }
  for (const component of Object.keys(ancestor)) {
    const anc = prop(ancestor, component, "ancestor components")
    const di = descendantI[component] ?? anc
    const dj = descendantJ[component] ?? anc
    if (di === dj) {
      merged[component] = di
    } else if (anc === di) {
      merged[component] = dj
    } else if (anc === dj) {
      merged[component] = di
    } else if (aggI > aggJ || (aggI === aggJ && rng() < 0.5)) {
      merged[component] = di
    } else {
      merged[component] = dj
    }
  }
  return merged
}

export type MergeProposal = {
  ancestor: number
  candidate: Candidate
  parentI: number
  parentJ: number
}

export type MergeMemory = {
  /** Merged candidates already produced, keyed per pair. */
  producedByPair: Set<string>
  /** Ancestor triplets already tried — recorded for the returned proposal only, before acceptance. */
  triedTriplets: Set<string>
}

export const tripletKey = (ancestor: number, i: number, j: number) =>
  `${ancestor}|${i}|${j}`

const MERGE_MAX_ATTEMPTS = 10

const candidateKey = (candidate: Candidate): string =>
  JSON.stringify(
    Object.entries(candidate).toSorted(([a], [b]) => a.localeCompare(b))
  )

/**
 * One pair-sampling attempt loop (Python's find_common_ancestor_pair): sample
 * two distinct non-ancestral survivors, then pick ONE aggregate-score-weighted
 * common ancestor among those that (a) don't outscore either descendant,
 * (b) have a component exactly one descendant changed, (c) aren't a tried
 * triplet. No eligible ancestor → resample the pair.
 */
const findCommonAncestorPair = (
  state: GEPAState,
  aggScores: number[],
  survivors: number[],
  rng: RNG,
  memory: MergeMemory
): [number, number, number] | null => {
  for (let attempt = 0; attempt < MERGE_MAX_ATTEMPTS; attempt += 1) {
    if (survivors.length < 2) {
      return null
    }
    const first = at(
      survivors,
      Math.floor(rng() * survivors.length),
      "merge survivors"
    )
    const rest = survivors.filter((idx) => idx !== first)
    const second = at(rest, Math.floor(rng() * rest.length), "merge survivors")
    const i = Math.min(first, second)
    const j = Math.max(first, second)
    const ancestorsI = findAncestors(state.parentProgramForCandidate, i)
    const ancestorsJ = findAncestors(state.parentProgramForCandidate, j)
    if (ancestorsI.has(j) || ancestorsJ.has(i)) {
      continue
    }
    const eligible = [...ancestorsI]
      .filter((idx) => ancestorsJ.has(idx))
      .filter((anc) => {
        if (memory.triedTriplets.has(tripletKey(anc, i, j))) {
          return false
        }
        const ancScore = aggScores[anc] ?? Number.NEGATIVE_INFINITY
        if (
          ancScore > (aggScores[i] ?? Number.NEGATIVE_INFINITY) ||
          ancScore > (aggScores[j] ?? Number.NEGATIVE_INFINITY)
        ) {
          return false
        }
        const ancestor = at(state.programCandidates, anc, "candidates")
        const ci = at(state.programCandidates, i, "candidates")
        const cj = at(state.programCandidates, j, "candidates")
        return Object.keys(ancestor).some((component) => {
          const anc_ = ancestor[component]
          const di = ci[component]
          const dj = cj[component]
          return di !== dj && (anc_ === di || anc_ === dj)
        })
      })
    if (eligible.length === 0) {
      continue
    }
    // rng.choices parity: an all-zero weight total throws and aborts the run.
    const ancestor = weightedChoiceStrict(
      rng,
      eligible,
      eligible.map((anc) => aggScores[anc] ?? 0)
    )
    return [i, j, ancestor]
  }
  return null
}

/**
 * Ancestor-triplet merge search (Python's
 * sample_and_attempt_merge_programs_by_common_predictors): every failure —
 * no triplet, duplicate merged candidate, insufficient val overlap —
 * resamples a fresh pair rather than giving up. The tried-triplet memo is
 * recorded by the CALLER for the returned proposal only.
 */
export const proposeMerge = (
  state: GEPAState,
  aggScores: number[],
  rng: RNG,
  memory: MergeMemory,
  valOverlapFloor = MERGE_VALIDATION_OVERLAP_FLOOR
): MergeProposal | null => {
  const survivors = removeDominatedPrograms(
    state.programAtParetoFrontValidationSet,
    aggScores
  )
  if (survivors.length < 2 || state.parentProgramForCandidate.length < 3) {
    return null
  }
  for (let attempt = 0; attempt < MERGE_MAX_ATTEMPTS; attempt += 1) {
    const triplet = findCommonAncestorPair(
      state,
      aggScores,
      survivors,
      rng,
      memory
    )
    if (!triplet) {
      continue
    }
    const [i, j, ancestor] = triplet
    if (memory.triedTriplets.has(tripletKey(ancestor, i, j))) {
      continue
    }
    const merged = buildMergedCandidate(
      at(state.programCandidates, ancestor, "candidates"),
      at(state.programCandidates, i, "candidates"),
      at(state.programCandidates, j, "candidates"),
      aggScores[i] ?? Number.NEGATIVE_INFINITY,
      aggScores[j] ?? Number.NEGATIVE_INFINITY,
      rng
    )
    const pairKey = `${i}|${j}|${candidateKey(merged)}`
    if (memory.producedByPair.has(pairKey)) {
      continue
    }
    const subscoresI = at(
      state.candidateValidationSubscores,
      i,
      "validation subscores"
    )
    const subscoresJ = at(
      state.candidateValidationSubscores,
      j,
      "validation subscores"
    )
    const sharedIds = [...subscoresI.keys()].filter((validationId) =>
      subscoresJ.has(validationId)
    )
    if (sharedIds.length < valOverlapFloor) {
      continue
    }
    memory.producedByPair.add(pairKey)
    return { ancestor, candidate: merged, parentI: i, parentJ: j }
  }
  return null
}

const MERGE_SUBSAMPLE_SIZE = 5
const MERGE_PER_BUCKET = Math.max(1, Math.ceil(MERGE_SUBSAMPLE_SIZE / 3))

const sampleUpTo = <T>(rng: RNG, items: T[], count: number): T[] => {
  const pool = [...items]
  shuffle(rng, pool)
  return pool.slice(0, count)
}

/**
 * Balanced 5-id subsample over the parents' shared val ids: up to 2 each from
 * the i-better / j-better / tied buckets, topped up from unused ids (with
 * replacement only as a last resort), truncated to 5.
 */
export const buildMergeSubsample = (
  sharedIds: number[],
  subscoresI: Map<number, number>,
  subscoresJ: Map<number, number>,
  rng: RNG
): number[] => {
  const iBetter: number[] = []
  const jBetter: number[] = []
  const tied: number[] = []
  for (const validationId of sharedIds) {
    const si = get(subscoresI, validationId, "subscores I")
    const sj = get(subscoresJ, validationId, "subscores J")
    if (si > sj) {
      iBetter.push(validationId)
    } else if (sj > si) {
      jBetter.push(validationId)
    } else {
      tied.push(validationId)
    }
  }
  const chosen: number[] = []
  for (const bucket of [iBetter, jBetter, tied]) {
    if (chosen.length >= MERGE_SUBSAMPLE_SIZE) {
      break
    }
    const available = bucket.filter(
      (validationId) => !chosen.includes(validationId)
    )
    const take = Math.min(
      available.length,
      MERGE_PER_BUCKET,
      MERGE_SUBSAMPLE_SIZE - chosen.length
    )
    if (take > 0) {
      chosen.push(...sampleUpTo(rng, available, take))
    }
  }
  const remaining = MERGE_SUBSAMPLE_SIZE - chosen.length
  if (remaining > 0) {
    const unused = sharedIds.filter(
      (validationId) => !chosen.includes(validationId)
    )
    if (unused.length >= remaining) {
      chosen.push(...sampleUpTo(rng, unused, remaining))
    } else if (sharedIds.length > 0) {
      // Last resort: sample WITH replacement from all shared ids.
      for (let k = 0; k < remaining; k += 1) {
        chosen.push(
          at(sharedIds, Math.floor(rng() * sharedIds.length), "shared ids")
        )
      }
    }
  }
  return chosen.slice(0, MERGE_SUBSAMPLE_SIZE)
}

export type MergeOutcome = "accepted" | "none" | "rejected"

/**
 * One merge iteration: propose an ancestor-triplet merge, eval it on the
 * balanced 5-id subsample (billed even when rejected), and accept iff the
 * merged sum is `>=` the better parent's sum on the same ids — non-strict,
 * unlike reflection's strict `>`.
 */
export const runMergeIteration = async <TInput, TOutput>(
  adapter: GEPAAdapter<TInput, TOutput>,
  state: GEPAState,
  options: {
    onAccepted?: (
      candidate: Candidate,
      subscores: Map<number, number>
    ) => Promise<void>
    rng: RNG
    validationSet: Example<TInput, TOutput>[]
  },
  memory: MergeMemory
): Promise<MergeOutcome> => {
  const { rng, validationSet } = options
  const aggScores = state.candidateValidationSubscores.map(aggregateScore)
  const proposal = proposeMerge(state, aggScores, rng, memory)
  if (!proposal) {
    return "none"
  }
  // The tried-triplet memo is recorded for the returned proposal only,
  // before its subsample eval and acceptance decision.
  memory.triedTriplets.add(
    tripletKey(proposal.ancestor, proposal.parentI, proposal.parentJ)
  )
  const subscoresI = at(
    state.candidateValidationSubscores,
    proposal.parentI,
    "validation subscores"
  )
  const subscoresJ = at(
    state.candidateValidationSubscores,
    proposal.parentJ,
    "validation subscores"
  )
  const sharedIds = [...subscoresI.keys()].filter((validationId) =>
    subscoresJ.has(validationId)
  )
  const subsample = buildMergeSubsample(sharedIds, subscoresI, subscoresJ, rng)
  const batch = subsample.map((validationId) =>
    at(validationSet, validationId, "validationSet")
  )
  const mergedEval = await adapter.evaluate(batch, proposal.candidate, false)
  state.totalEvalsCount += batch.length
  const sumMerged = sum(mergedEval.scores)
  const sumI = sum(
    subsample.map((validationId) =>
      get(subscoresI, validationId, "subscores I")
    )
  )
  const sumJ = sum(
    subsample.map((validationId) =>
      get(subscoresJ, validationId, "subscores J")
    )
  )
  if (sumMerged < Math.max(sumI, sumJ)) {
    return "rejected"
  }
  const fullEval = await adapter.evaluate(
    validationSet,
    proposal.candidate,
    false
  )
  const subscores = toSubscores(fullEval.scores)
  addCandidate(
    state,
    proposal.candidate,
    [proposal.parentI, proposal.parentJ],
    subscores,
    validationSet.length
  )
  await options.onAccepted?.(proposal.candidate, subscores)
  console.log(
    `GEPA iteration ${state.i}: accepted merge of ${proposal.parentI} and ${proposal.parentJ}`
  )
  return "accepted"
}

// --- Engine -----------------------------------------------------------------

export type EngineOptions<TInput = Fields, TOutput = Fields> = {
  candidateSelectionStrategy: "currentBest" | "pareto"
  componentSelector: "all" | "roundRobin"
  maxMergeInvocations: number
  maxMetricCalls: number
  /** Called whenever an accepted candidate becomes the new aggregate-score best. */
  onImprovement?: (candidate: Candidate) => Promise<void>
  perfectScore: number
  reflectionMinibatchSize: number
  rng: RNG
  seedCandidate: Candidate
  skipPerfectScore: boolean
  trainingSet: Example<TInput, TOutput>[]
  useMerge: boolean
  validationSet: Example<TInput, TOutput>[]
}

const selectParent = (
  state: GEPAState,
  options: Pick<EngineOptions, "candidateSelectionStrategy" | "rng">
): number => {
  const aggScores = state.candidateValidationSubscores.map(aggregateScore)
  if (options.candidateSelectionStrategy === "currentBest") {
    return argmax(aggScores)
  }
  return selectParetoParent(
    state.programAtParetoFrontValidationSet,
    aggScores,
    options.rng
  )
}

/** GEPAState after the seed candidate's full validationSet eval. */
export const initGEPAState = (
  seedCandidate: Candidate,
  seedScores: number[]
): GEPAState => {
  const state: GEPAState = {
    candidateValidationSubscores: [toSubscores(seedScores)],
    i: -1,
    metricCallCountsByDiscovery: [0],
    parentProgramForCandidate: [[null]],
    paretoFrontValidationSet: new Map(),
    programAtParetoFrontValidationSet: new Map(),
    programCandidates: [seedCandidate],
    stepIdToUpdateNextForCandidate: [0],
    totalEvalsCount: seedScores.length,
    validationSetEvalsCount: 1,
  }
  updateParetoFront(
    state.paretoFrontValidationSet,
    state.programAtParetoFrontValidationSet,
    0,
    at(state.candidateValidationSubscores, 0, "validation subscores")
  )
  return state
}

/**
 * The loop-level bookkeeping that rides alongside GEPAState between
 * iterations: merge scheduling, the improvement watermark for onImprovement,
 * and the minibatch sampler's position.
 */
export type GEPALoopState = {
  bestAgg: number
  lastIterFoundNewProgram: boolean
  mergeMemory: MergeMemory
  mergesDue: number
  samplerState: EpochSamplerState
  totalMergesTested: number
}

export const initGEPALoopState = (state: GEPAState): GEPALoopState => ({
  bestAgg: aggregateScore(
    at(state.candidateValidationSubscores, 0, "validation subscores")
  ),
  lastIterFoundNewProgram: false,
  mergeMemory: { producedByPair: new Set(), triedTriplets: new Set() },
  mergesDue: 0,
  samplerState: initEpochSamplerState(),
  totalMergesTested: 0,
})

const noteImprovement = async (
  loop: GEPALoopState,
  onImprovement: EngineOptions["onImprovement"],
  candidate: Candidate,
  subscores: Map<number, number>
): Promise<void> => {
  const agg = aggregateScore(subscores)
  if (agg > loop.bestAgg) {
    loop.bestAgg = agg
    await onImprovement?.(candidate)
  }
}

/**
 * Everything the reflection LM call needs, and everything the acceptance
 * decision needs afterwards — JSON, so the durable workflow driver can put
 * the proposal call in its own step.
 */
export type ReflectionPlan = {
  /** Components that actually have reflective records — the proposal targets. */
  components: string[]
  minibatchIds: number[]
  parentIdx: number
  parentScores: number[]
  reflectiveDataset: ReflectiveDataset
}

/**
 * The reflection prologue: pick a parent, evaluate it on the epoch-shuffled
 * minibatch with traces, advance the component cursor, and build the
 * reflective dataset. Returns null when the iteration produces nothing to
 * reflect on (no trajectories, all-perfect scores, or no component records) —
 * budget is still billed, exactly like upstream.
 */
export const prepareReflection = async <TInput, TOutput>(
  adapter: GEPAAdapter<TInput, TOutput>,
  state: GEPAState,
  loop: GEPALoopState,
  options: EngineOptions<TInput, TOutput>
): Promise<ReflectionPlan | null> => {
  const componentNames = Object.keys(options.seedCandidate)
  const parentIdx = selectParent(state, options)
  const parent = at(state.programCandidates, parentIdx, "candidates")

  const minibatchIds = sampleEpochShuffled(
    options.rng,
    options.trainingSet.length,
    options.reflectionMinibatchSize,
    loop.samplerState,
    state.i
  )
  const batch = minibatchIds.map((id) =>
    at(options.trainingSet, id, "trainingSet")
  )
  const parentEval = await adapter.evaluate(batch, parent, true)
  state.totalEvalsCount += batch.length

  if (!parentEval.trajectories || parentEval.trajectories.length === 0) {
    return null
  }
  if (
    options.skipPerfectScore &&
    parentEval.scores.every((score) => score >= options.perfectScore)
  ) {
    return null
  }

  let components: string[]
  if (options.componentSelector === "all") {
    components = componentNames
  } else {
    // The cursor advances on the parent even if the proposal is rejected.
    const cursor = at(
      state.stepIdToUpdateNextForCandidate,
      parentIdx,
      "step cursors"
    )
    components = [
      at(componentNames, cursor % componentNames.length, "component names"),
    ]
    state.stepIdToUpdateNextForCandidate[parentIdx] =
      (cursor + 1) % componentNames.length
  }

  const reflectiveDataset = await adapter.makeReflectiveDataset(
    parent,
    parentEval,
    components
  )
  const componentsWithData = components.filter(
    (name) => (reflectiveDataset[name] ?? []).length > 0
  )
  if (componentsWithData.length === 0) {
    return null
  }

  return {
    components: componentsWithData,
    minibatchIds,
    parentIdx,
    parentScores: parentEval.scores,
    reflectiveDataset,
  }
}

/**
 * The reflection epilogue: apply the proposed texts to the parent, evaluate
 * the child on the same minibatch, and accept iff its score SUM strictly
 * beats the parent's (upstream's rule) — acceptance bills a full
 * validationSet eval, registers the candidate, and schedules a merge.
 */
export const acceptReflection = async <TInput, TOutput>(
  adapter: GEPAAdapter<TInput, TOutput>,
  state: GEPAState,
  loop: GEPALoopState,
  options: EngineOptions<TInput, TOutput>,
  plan: ReflectionPlan,
  newTexts: Record<string, string>
): Promise<boolean> => {
  const parent = at(state.programCandidates, plan.parentIdx, "candidates")
  // Python keeps empty extracted text as a real proposal; only an empty
  // proposal DICT skips the round.
  const applicable = Object.entries(newTexts).filter(([name]) => name in parent)
  if (applicable.length === 0) {
    return false
  }
  const child = { ...parent }
  for (const [name, text] of applicable) {
    child[name] = text
  }

  const batch = plan.minibatchIds.map((id) =>
    at(options.trainingSet, id, "trainingSet")
  )
  // Python evaluates children with capture_traces=True (used only for
  // logging upstream, but it keeps the adapter call shape identical).
  const childEval = await adapter.evaluate(batch, child, true)
  state.totalEvalsCount += batch.length

  // Strict > on SUMS (not means) for reflection acceptance.
  if (sum(childEval.scores) <= sum(plan.parentScores)) {
    return false
  }
  const fullEval = await adapter.evaluate(options.validationSet, child, false)
  const subscores = toSubscores(fullEval.scores)
  addCandidate(
    state,
    child,
    [plan.parentIdx],
    subscores,
    options.validationSet.length
  )
  await noteImprovement(loop, options.onImprovement, child, subscores)
  loop.lastIterFoundNewProgram = true
  if (loop.totalMergesTested < options.maxMergeInvocations) {
    loop.mergesDue += 1
  }
  console.log(
    `GEPA iteration ${state.i}: accepted child of ${plan.parentIdx} (component ${plan.components.join(", ")})`
  )
  return true
}

/**
 * The merge branch of one iteration, when it is due: returns the outcome and
 * does the loop bookkeeping. "none" means the search was fruitless and the
 * iteration should fall through to reflection.
 */
export const runMergeBranch = async <TInput, TOutput>(
  adapter: GEPAAdapter<TInput, TOutput>,
  state: GEPAState,
  loop: GEPALoopState,
  options: EngineOptions<TInput, TOutput>
): Promise<MergeOutcome> => {
  loop.lastIterFoundNewProgram = false
  const outcome = await runMergeIteration(
    adapter,
    state,
    {
      onAccepted: (candidate, subscores) =>
        noteImprovement(loop, options.onImprovement, candidate, subscores),
      rng: options.rng,
      validationSet: options.validationSet,
    },
    loop.mergeMemory
  )
  if (outcome === "accepted") {
    loop.mergesDue -= 1
    loop.totalMergesTested += 1
  }
  return outcome
}

/** Whether the next iteration should try a merge before reflecting. */
export const mergeDue = (
  loop: GEPALoopState,
  options: Pick<EngineOptions, "useMerge">
): boolean =>
  options.useMerge && loop.mergesDue > 0 && loop.lastIterFoundNewProgram

export const runGEPA = async <TInput, TOutput>(
  adapter: GEPAAdapter<TInput, TOutput>,
  options: EngineOptions<TInput, TOutput>
): Promise<GEPAState> => {
  const seedEval = await adapter.evaluate(
    options.validationSet,
    options.seedCandidate,
    false
  )
  const state = initGEPAState(options.seedCandidate, seedEval.scores)
  const loop = initGEPALoopState(state)

  while (state.totalEvalsCount < options.maxMetricCalls) {
    state.i += 1
    if (mergeDue(loop, options)) {
      // oxlint-disable-next-line no-await-in-loop -- iterations are inherently sequential
      const outcome = await runMergeBranch(adapter, state, loop, options)
      // A PRODUCED merge proposal ends the iteration whether accepted or
      // rejected; a fruitless search falls through to reflection.
      if (outcome !== "none") {
        continue
      }
    }
    // oxlint-disable-next-line no-await-in-loop -- GEPA iterations are inherently sequential: each reflection reads state the previous one wrote
    const plan = await prepareReflection(adapter, state, loop, options)
    if (!plan) {
      continue
    }
    const parent = at(state.programCandidates, plan.parentIdx, "candidates")
    // oxlint-disable-next-line no-await-in-loop -- see above
    const newTexts = await adapter.proposeNewTexts(
      parent,
      plan.reflectiveDataset,
      plan.components
    )
    // oxlint-disable-next-line no-await-in-loop -- see above
    await acceptReflection(adapter, state, loop, options, plan, newTexts)
  }

  return state
}

// --- Durable-state codec ------------------------------------------------------
//
// GEPAState and GEPALoopState keep Maps and Sets for the hot paths; the
// durable workflow driver moves them between steps as JSON via this codec.

export type SerializedGEPAState = {
  candidateValidationSubscores: [number, number][][]
  i: number
  metricCallCountsByDiscovery: number[]
  paretoFrontValidationSet: [number, number][]
  parentProgramForCandidate: (number | null)[][]
  programAtParetoFrontValidationSet: [number, number[]][]
  programCandidates: Candidate[]
  stepIdToUpdateNextForCandidate: number[]
  totalEvalsCount: number
  validationSetEvalsCount: number
}

export type SerializedGEPALoopState = {
  bestAgg: number
  lastIterFoundNewProgram: boolean
  mergeMemory: { producedByPair: string[]; triedTriplets: string[] }
  mergesDue: number
  samplerState: EpochSamplerState
  totalMergesTested: number
}

export const serializeGEPAState = (state: GEPAState): SerializedGEPAState => ({
  candidateValidationSubscores: state.candidateValidationSubscores.map(
    (subscores) => [...subscores.entries()]
  ),
  i: state.i,
  metricCallCountsByDiscovery: [...state.metricCallCountsByDiscovery],
  parentProgramForCandidate: state.parentProgramForCandidate.map((parents) => [
    ...parents,
  ]),
  paretoFrontValidationSet: [...state.paretoFrontValidationSet.entries()],
  programAtParetoFrontValidationSet: [
    ...state.programAtParetoFrontValidationSet.entries(),
  ].map(([validationId, programs]) => [validationId, [...programs]]),
  programCandidates: structuredClone(state.programCandidates),
  stepIdToUpdateNextForCandidate: [...state.stepIdToUpdateNextForCandidate],
  totalEvalsCount: state.totalEvalsCount,
  validationSetEvalsCount: state.validationSetEvalsCount,
})

export const deserializeGEPAState = (
  serialized: SerializedGEPAState
): GEPAState => ({
  candidateValidationSubscores: serialized.candidateValidationSubscores.map(
    (subscores) => new Map(subscores)
  ),
  i: serialized.i,
  metricCallCountsByDiscovery: [...serialized.metricCallCountsByDiscovery],
  parentProgramForCandidate: serialized.parentProgramForCandidate.map(
    (parents) => [...parents]
  ),
  paretoFrontValidationSet: new Map(serialized.paretoFrontValidationSet),
  programAtParetoFrontValidationSet: new Map(
    serialized.programAtParetoFrontValidationSet.map(
      ([validationId, programs]) => [validationId, new Set(programs)]
    )
  ),
  programCandidates: structuredClone(serialized.programCandidates),
  stepIdToUpdateNextForCandidate: [
    ...serialized.stepIdToUpdateNextForCandidate,
  ],
  totalEvalsCount: serialized.totalEvalsCount,
  validationSetEvalsCount: serialized.validationSetEvalsCount,
})

export const serializeGEPALoopState = (
  loop: GEPALoopState
): SerializedGEPALoopState => ({
  bestAgg: loop.bestAgg,
  lastIterFoundNewProgram: loop.lastIterFoundNewProgram,
  mergeMemory: {
    producedByPair: [...loop.mergeMemory.producedByPair],
    triedTriplets: [...loop.mergeMemory.triedTriplets],
  },
  mergesDue: loop.mergesDue,
  samplerState: structuredClone(loop.samplerState),
  totalMergesTested: loop.totalMergesTested,
})

export const deserializeGEPALoopState = (
  serialized: SerializedGEPALoopState
): GEPALoopState => ({
  bestAgg: serialized.bestAgg,
  lastIterFoundNewProgram: serialized.lastIterFoundNewProgram,
  mergeMemory: {
    producedByPair: new Set(serialized.mergeMemory.producedByPair),
    triedTriplets: new Set(serialized.mergeMemory.triedTriplets),
  },
  mergesDue: serialized.mergesDue,
  samplerState: structuredClone(serialized.samplerState),
  totalMergesTested: serialized.totalMergesTested,
})
