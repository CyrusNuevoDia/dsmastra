import type { TraceStep } from "@/predictor"
import type { Example } from "@/program"
import {
  type RNG,
  shuffle,
  weightedChoice,
  weightedChoiceStrict,
} from "@/random"

export type { RNG } from "@/random"

/**
 * A candidate is a map of component (predictor) name → instruction text,
 * exactly upstream's dict[str, str]. Demos live on the program's predictors
 * (bootstrapFewShot pre-pass) and flow through build_program untouched.
 */
export type Candidate = Record<string, string>

export type Trajectory = {
  example: Example
  prediction: Record<string, unknown> | null
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

export type EvaluationBatch = {
  outputs: (Record<string, unknown> | null)[]
  scores: number[]
  trajectories?: Trajectory[]
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
export type GEPAAdapter = {
  evaluate: (
    batch: Example[],
    candidate: Candidate,
    captureTraces: boolean
  ) => Promise<EvaluationBatch>
  makeReflectiveDataset: (
    candidate: Candidate,
    evalBatch: EvaluationBatch,
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
  namedPredictorIdToUpdateNextForProgramCandidate: number[]
  numFullDSEvals: number
  numMetricCallsByDiscovery: number[]
  paretoFrontValset: Map<number, number>
  parentProgramForCandidate: (number | null)[][]
  progCandidateValSubscores: Map<number, number>[]
  programAtParetoFrontValset: Map<number, Set<number>>
  programCandidates: Candidate[]
  totalNumEvals: number
}

// --- Small helpers ----------------------------------------------------------

function sum(values: number[]): number {
  return values.reduce((acc, v) => acc + v, 0)
}

/** Mean of a candidate's per-instance scores; -Infinity if unevaluated. */
export function aggregateScore(subscores: Map<number, number>): number {
  if (subscores.size === 0) {
    return Number.NEGATIVE_INFINITY
  }
  return sum([...subscores.values()]) / subscores.size
}

/** argmax with lowest-index-wins ties. */
export function argmax(values: number[]): number {
  let best = 0
  for (let i = 1; i < values.length; i += 1) {
    if ((values[i] as number) > (values[best] as number)) {
      best = i
    }
  }
  return best
}

// --- Pareto frontier --------------------------------------------------------

/**
 * Per-instance frontier update after a candidate's full valset eval. Strict
 * improvement replaces the set; an exact float tie (no epsilon) adds to it.
 */
export function updateParetoFront(
  front: Map<number, number>,
  frontPrograms: Map<number, Set<number>>,
  candidateIdx: number,
  subscores: Map<number, number>
): void {
  for (const [valId, score] of subscores) {
    const prev = front.get(valId)
    if (prev === undefined || score > prev) {
      front.set(valId, score)
      frontPrograms.set(valId, new Set([candidateIdx]))
    } else if (score === prev) {
      frontPrograms.get(valId)?.add(candidateIdx)
    }
  }
}

/**
 * Dominance filter: a candidate survives iff it is the sole occupant of at
 * least one instance's front. Scan candidates ascending by aggregate score,
 * mark one dominated when every front containing it also holds another
 * non-dominated candidate, and restart after each removal — so ties resolve
 * toward the higher aggregate scorer.
 */
export function removeDominatedPrograms(
  frontPrograms: Map<number, Set<number>>,
  aggScores: number[]
): number[] {
  const members = new Set<number>()
  for (const set of frontPrograms.values()) {
    for (const idx of set) {
      members.add(idx)
    }
  }
  // Stable sort by score alone: exact-score ties keep front-iteration
  // encounter order, matching Python's sorted(programs, key=score).
  const ascending = [...members].sort(
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
  return [...members].filter((idx) => !dominated.has(idx)).sort((a, b) => a - b)
}

/**
 * Pareto parent sampling: dominance-filter the frontier, then pick a survivor
 * with probability proportional to how many instance fronts it occupies.
 */
export function selectParetoParent(
  frontPrograms: Map<number, Set<number>>,
  aggScores: number[],
  rng: RNG
): number {
  const survivors = removeDominatedPrograms(frontPrograms, aggScores)
  const frequencies = survivors.map(
    (idx) => [...frontPrograms.values()].filter((set) => set.has(idx)).length
  )
  return weightedChoice(rng, survivors, frequencies)
}

// --- Minibatch sampler ------------------------------------------------------

/**
 * Epoch-shuffled minibatch sampler: shuffle the train ids once per epoch, pad
 * the tail to a multiple of bsize with the least-frequently-used id, and serve
 * sequential windows keyed by iteration number.
 */
export function createEpochShuffledSampler(
  rng: RNG,
  trainSize: number,
  bsize: number
): (iteration: number) => number[] {
  let shuffled: number[] = []
  let epoch = -1

  const reshuffle = () => {
    shuffled = Array.from({ length: trainSize }, (_, i) => i)
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
  }

  return (iteration: number): number[] => {
    const base = iteration * bsize
    const currEpoch =
      epoch === -1 ? 0 : Math.floor(base / Math.max(shuffled.length, 1))
    if (shuffled.length === 0 || currEpoch > epoch) {
      epoch = currEpoch
      reshuffle()
    }
    const start = base % shuffled.length
    return shuffled.slice(start, start + bsize)
  }
}

// --- Merge (crossover) ------------------------------------------------------

/** Transitive closure over parent lineage (excludes the candidate itself). */
export function findAncestors(
  parents: (number | null)[][],
  idx: number
): Set<number> {
  const ancestors = new Set<number>()
  const stack = [
    ...((parents[idx] ?? []).filter((p) => p !== null) as number[]),
  ]
  while (stack.length > 0) {
    const current = stack.pop() as number
    if (ancestors.has(current)) {
      continue
    }
    ancestors.add(current)
    stack.push(
      ...((parents[current] ?? []).filter((p) => p !== null) as number[])
    )
  }
  return ancestors
}

/**
 * Per-component merge rule: agreeing descendants win outright; a lone
 * divergence from the ancestor wins; a double divergence goes to the
 * higher-aggregate descendant, coin-flipped on an exact tie.
 */
export function buildMergedCandidate(
  ancestor: Candidate,
  descendantI: Candidate,
  descendantJ: Candidate,
  aggI: number,
  aggJ: number,
  rng: RNG
): Candidate {
  const merged: Candidate = { ...ancestor }
  for (const component of Object.keys(ancestor)) {
    const anc = ancestor[component] as string
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

function candidateKey(candidate: Candidate): string {
  return JSON.stringify(
    Object.entries(candidate).sort(([a], [b]) => a.localeCompare(b))
  )
}

/**
 * One pair-sampling attempt loop (Python's find_common_ancestor_pair): sample
 * two distinct non-ancestral survivors, then pick ONE aggregate-score-weighted
 * common ancestor among those that (a) don't outscore either descendant,
 * (b) have a component exactly one descendant changed, (c) aren't a tried
 * triplet. No eligible ancestor → resample the pair.
 */
function findCommonAncestorPair(
  state: GEPAState,
  aggScores: number[],
  survivors: number[],
  rng: RNG,
  memory: MergeMemory
): [number, number, number] | null {
  for (let attempt = 0; attempt < MERGE_MAX_ATTEMPTS; attempt += 1) {
    if (survivors.length < 2) {
      return null
    }
    const first = survivors[Math.floor(rng() * survivors.length)] as number
    const rest = survivors.filter((idx) => idx !== first)
    const second = rest[Math.floor(rng() * rest.length)] as number
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
        const ancestor = state.programCandidates[anc] as Candidate
        const ci = state.programCandidates[i] as Candidate
        const cj = state.programCandidates[j] as Candidate
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
export function proposeMerge(
  state: GEPAState,
  aggScores: number[],
  rng: RNG,
  memory: MergeMemory,
  valOverlapFloor = MERGE_VAL_OVERLAP_FLOOR
): MergeProposal | null {
  const survivors = removeDominatedPrograms(
    state.programAtParetoFrontValset,
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
      state.programCandidates[ancestor] as Candidate,
      state.programCandidates[i] as Candidate,
      state.programCandidates[j] as Candidate,
      aggScores[i] ?? Number.NEGATIVE_INFINITY,
      aggScores[j] ?? Number.NEGATIVE_INFINITY,
      rng
    )
    const pairKey = `${i}|${j}|${candidateKey(merged)}`
    if (memory.producedByPair.has(pairKey)) {
      continue
    }
    const subscoresI = state.progCandidateValSubscores[i] as Map<number, number>
    const subscoresJ = state.progCandidateValSubscores[j] as Map<number, number>
    const sharedIds = [...subscoresI.keys()].filter((valId) =>
      subscoresJ.has(valId)
    )
    if (sharedIds.length < valOverlapFloor) {
      continue
    }
    memory.producedByPair.add(pairKey)
    return { ancestor, candidate: merged, parentI: i, parentJ: j }
  }
  return null
}

export const MERGE_VAL_OVERLAP_FLOOR = 5
const MERGE_SUBSAMPLE_SIZE = 5
const MERGE_PER_BUCKET = Math.max(1, Math.ceil(MERGE_SUBSAMPLE_SIZE / 3))

function sampleUpTo<T>(rng: RNG, items: T[], count: number): T[] {
  const pool = [...items]
  shuffle(rng, pool)
  return pool.slice(0, count)
}

/**
 * Balanced 5-id subsample over the parents' shared val ids: up to 2 each from
 * the i-better / j-better / tied buckets, topped up from unused ids (with
 * replacement only as a last resort), truncated to 5.
 */
export function buildMergeSubsample(
  sharedIds: number[],
  subscoresI: Map<number, number>,
  subscoresJ: Map<number, number>,
  rng: RNG
): number[] {
  const iBetter: number[] = []
  const jBetter: number[] = []
  const tied: number[] = []
  for (const valId of sharedIds) {
    const si = subscoresI.get(valId) as number
    const sj = subscoresJ.get(valId) as number
    if (si > sj) {
      iBetter.push(valId)
    } else if (sj > si) {
      jBetter.push(valId)
    } else {
      tied.push(valId)
    }
  }
  const chosen: number[] = []
  for (const bucket of [iBetter, jBetter, tied]) {
    if (chosen.length >= MERGE_SUBSAMPLE_SIZE) {
      break
    }
    const available = bucket.filter((valId) => !chosen.includes(valId))
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
    const unused = sharedIds.filter((valId) => !chosen.includes(valId))
    if (unused.length >= remaining) {
      chosen.push(...sampleUpTo(rng, unused, remaining))
    } else if (sharedIds.length > 0) {
      // Last resort: sample WITH replacement from all shared ids.
      for (let k = 0; k < remaining; k += 1) {
        chosen.push(sharedIds[Math.floor(rng() * sharedIds.length)] as number)
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
export async function runMergeIteration(
  adapter: GEPAAdapter,
  state: GEPAState,
  options: { rng: RNG; valset: Example[] },
  memory: MergeMemory
): Promise<MergeOutcome> {
  const { rng, valset } = options
  const aggScores = state.progCandidateValSubscores.map(aggregateScore)
  const proposal = proposeMerge(state, aggScores, rng, memory)
  if (!proposal) {
    return "none"
  }
  // The tried-triplet memo is recorded for the returned proposal only,
  // before its subsample eval and acceptance decision.
  memory.triedTriplets.add(
    tripletKey(proposal.ancestor, proposal.parentI, proposal.parentJ)
  )
  const subscoresI = state.progCandidateValSubscores[proposal.parentI] as Map<
    number,
    number
  >
  const subscoresJ = state.progCandidateValSubscores[proposal.parentJ] as Map<
    number,
    number
  >
  const sharedIds = [...subscoresI.keys()].filter((valId) =>
    subscoresJ.has(valId)
  )
  const subsample = buildMergeSubsample(sharedIds, subscoresI, subscoresJ, rng)
  const batch = subsample.map((valId) => valset[valId] as Example)
  const mergedEval = await adapter.evaluate(batch, proposal.candidate, false)
  state.totalNumEvals += batch.length
  const sumMerged = sum(mergedEval.scores)
  const sumI = sum(subsample.map((valId) => subscoresI.get(valId) as number))
  const sumJ = sum(subsample.map((valId) => subscoresJ.get(valId) as number))
  if (sumMerged < Math.max(sumI, sumJ)) {
    return "rejected"
  }
  const fullEval = await adapter.evaluate(valset, proposal.candidate, false)
  addCandidate(
    state,
    proposal.candidate,
    [proposal.parentI, proposal.parentJ],
    toSubscores(fullEval.scores),
    valset.length
  )
  console.log(
    `GEPA iteration ${state.i}: accepted merge of ${proposal.parentI} and ${proposal.parentJ}`
  )
  return "accepted"
}

// --- Engine -----------------------------------------------------------------

export type EngineOptions = {
  candidateSelectionStrategy: "currentBest" | "pareto"
  componentSelector: "all" | "roundRobin"
  maxMergeInvocations: number
  maxMetricCalls: number
  perfectScore: number
  reflectionMinibatchSize: number
  rng: RNG
  seedCandidate: Candidate
  skipPerfectScore: boolean
  trainset: Example[]
  useMerge: boolean
  valset: Example[]
}

function toSubscores(scores: number[]): Map<number, number> {
  return new Map(scores.map((score, valId) => [valId, score]))
}

/**
 * Register an accepted candidate: snapshot the budget as its discovery count
 * BEFORE billing its full eval, bill it, update the frontier, and inherit
 * `max(parent cursors)` as its round-robin cursor.
 */
function addCandidate(
  state: GEPAState,
  candidate: Candidate,
  parents: (number | null)[],
  subscores: Map<number, number>,
  valsetSize: number
): number {
  const idx = state.programCandidates.length
  state.programCandidates.push(candidate)
  state.parentProgramForCandidate.push(parents)
  state.progCandidateValSubscores.push(subscores)
  state.numMetricCallsByDiscovery.push(state.totalNumEvals)
  state.namedPredictorIdToUpdateNextForProgramCandidate.push(
    Math.max(
      0,
      ...parents
        .filter((p): p is number => p !== null)
        .map(
          (p) =>
            state.namedPredictorIdToUpdateNextForProgramCandidate[p] as number
        )
    )
  )
  state.totalNumEvals += valsetSize
  state.numFullDSEvals += 1
  updateParetoFront(
    state.paretoFrontValset,
    state.programAtParetoFrontValset,
    idx,
    subscores
  )
  return idx
}

function selectParent(state: GEPAState, options: EngineOptions): number {
  const aggScores = state.progCandidateValSubscores.map(aggregateScore)
  if (options.candidateSelectionStrategy === "currentBest") {
    return argmax(aggScores)
  }
  return selectParetoParent(
    state.programAtParetoFrontValset,
    aggScores,
    options.rng
  )
}

export async function runGEPA(
  adapter: GEPAAdapter,
  options: EngineOptions
): Promise<GEPAState> {
  const { rng, trainset, valset } = options
  const componentNames = Object.keys(options.seedCandidate)

  const seedEval = await adapter.evaluate(valset, options.seedCandidate, false)
  const state: GEPAState = {
    i: -1,
    namedPredictorIdToUpdateNextForProgramCandidate: [0],
    numFullDSEvals: 1,
    numMetricCallsByDiscovery: [0],
    parentProgramForCandidate: [[null]],
    paretoFrontValset: new Map(),
    progCandidateValSubscores: [toSubscores(seedEval.scores)],
    programAtParetoFrontValset: new Map(),
    programCandidates: [options.seedCandidate],
    totalNumEvals: valset.length,
  }
  updateParetoFront(
    state.paretoFrontValset,
    state.programAtParetoFrontValset,
    0,
    state.progCandidateValSubscores[0] as Map<number, number>
  )

  const sampler = createEpochShuffledSampler(
    rng,
    trainset.length,
    options.reflectionMinibatchSize
  )
  const mergeMemory: MergeMemory = {
    producedByPair: new Set(),
    triedTriplets: new Set(),
  }
  let mergesDue = 0
  let totalMergesTested = 0
  let lastIterFoundNewProgram = false

  const runReflection = async (): Promise<void> => {
    const parentIdx = selectParent(state, options)
    const parent = state.programCandidates[parentIdx] as Candidate

    const minibatchIds = sampler(state.i)
    const batch = minibatchIds.map((id) => trainset[id] as Example)
    const parentEval = await adapter.evaluate(batch, parent, true)
    state.totalNumEvals += batch.length

    if (!parentEval.trajectories || parentEval.trajectories.length === 0) {
      return
    }
    if (
      options.skipPerfectScore &&
      parentEval.scores.every((score) => score >= options.perfectScore)
    ) {
      return
    }

    let components: string[]
    if (options.componentSelector === "all") {
      components = componentNames
    } else {
      // The cursor advances on the parent even if the proposal is rejected.
      const cursor = state.namedPredictorIdToUpdateNextForProgramCandidate[
        parentIdx
      ] as number
      components = [componentNames[cursor % componentNames.length] as string]
      state.namedPredictorIdToUpdateNextForProgramCandidate[parentIdx] =
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
      return
    }

    const newTexts = await adapter.proposeNewTexts(
      parent,
      reflectiveDataset,
      componentsWithData
    )
    // Python keeps empty extracted text as a real proposal; only an empty
    // proposal DICT skips the round.
    const applicable = Object.entries(newTexts).filter(
      ([name]) => name in parent
    )
    if (applicable.length === 0) {
      return
    }
    const child: Candidate = { ...parent }
    for (const [name, text] of applicable) {
      child[name] = text
    }

    // Python evaluates children with capture_traces=True (used only for
    // logging upstream, but it keeps the adapter call shape identical).
    const childEval = await adapter.evaluate(batch, child, true)
    state.totalNumEvals += batch.length

    // Strict > on SUMS (not means) for reflection acceptance.
    if (sum(childEval.scores) > sum(parentEval.scores)) {
      const fullEval = await adapter.evaluate(valset, child, false)
      addCandidate(
        state,
        child,
        [parentIdx],
        toSubscores(fullEval.scores),
        valset.length
      )
      lastIterFoundNewProgram = true
      if (totalMergesTested < options.maxMergeInvocations) {
        mergesDue += 1
      }
      console.log(
        `GEPA iteration ${state.i}: accepted child of ${parentIdx} (component ${components.join(", ")})`
      )
    }
  }

  while (state.totalNumEvals < options.maxMetricCalls) {
    state.i += 1
    if (options.useMerge && mergesDue > 0 && lastIterFoundNewProgram) {
      lastIterFoundNewProgram = false
      // biome-ignore lint/performance/noAwaitInLoops: iterations are inherently sequential
      const outcome = await runMergeIteration(
        adapter,
        state,
        { rng, valset },
        mergeMemory
      )
      if (outcome === "accepted") {
        mergesDue -= 1
        totalMergesTested += 1
      }
      // A PRODUCED merge proposal ends the iteration whether accepted or
      // rejected; a fruitless search falls through to reflection.
      if (outcome !== "none") {
        continue
      }
    }
    await runReflection()
  }

  return state
}
