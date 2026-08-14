import { expect, test } from "bun:test"
import { z } from "zod"
import { autoBudget, gepa } from "@/gepa"
import { createProgramAdapter, extractInstructionText } from "@/gepa/adapter"
import {
  buildMergeSubsample,
  type Candidate,
  createEpochShuffledSampler,
  type EngineOptions,
  type GEPAAdapter,
  type GEPAState,
  proposeMerge,
  removeDominatedPrograms,
  runGEPA,
  runMergeIteration,
  selectParetoParent,
  updateParetoFront,
} from "@/gepa/engine"
import type { Demo, Predictor } from "@/predictor"
import { createProgram, type Program } from "@/program"
import type { Example } from "@/simba"

const zero = () => 0

function examples(n: number): Example[] {
  return Array.from({ length: n }, (_, id) => ({
    inputs: { id },
    outputs: {},
  }))
}

// --- Auto-budget -------------------------------------------------------------

test("autoBudget matches hand-computed values", () => {
  // numPreds=1, n=6, V=10: N = floor(max(4*log2(6), 9)) = 10
  // total = 10 + 30 + 350 + (floor(11/5)+1)*10 = 420
  expect(autoBudget(1, 6, 10)).toBe(420)
  // N<m branch: numPreds=1, n=2: N = 4 < 5
  // total = 4 + 10 + 140 + (floor(5/5)+1+1)*4 = 166
  expect(autoBudget(1, 2, 4)).toBe(166)
})

test("autoBudget N=0 branch returns before periodic evals", () => {
  // n=0 → max(-Inf, 0) → N=0 → total = valsetSize only
  expect(autoBudget(1, 0, 7)).toBe(7)
})

test("autoBudget guards throw", () => {
  expect(() => autoBudget(1, 2, -1)).toThrow()
  expect(() => autoBudget(1, 2, 4, -1)).toThrow()
  expect(() => autoBudget(1, 2, 4, 35, 0)).toThrow()
})

// --- Pareto frontier ---------------------------------------------------------

test("frontier update: exact tie adds to the set, improvement replaces it", () => {
  const front = new Map<number, number>()
  const frontPrograms = new Map<number, Set<number>>()
  updateParetoFront(
    front,
    frontPrograms,
    0,
    new Map([
      [0, 0.5],
      [1, 0.5],
    ])
  )
  updateParetoFront(
    front,
    frontPrograms,
    1,
    new Map([
      [0, 0.5],
      [1, 0.7],
    ])
  )
  expect(front.get(0)).toBe(0.5)
  expect([...(frontPrograms.get(0) as Set<number>)].sort()).toEqual([0, 1])
  expect(front.get(1)).toBe(0.7)
  expect([...(frontPrograms.get(1) as Set<number>)]).toEqual([1])
})

test("dominance filter: sole occupant survives, covered candidate is dominated", () => {
  const fronts = new Map<number, Set<number>>([
    [0, new Set([0])],
    [1, new Set([0, 1])],
  ])
  expect(removeDominatedPrograms(fronts, [0.5, 0.9])).toEqual([0])
})

test("dominance filter: full ties resolve toward the higher aggregate", () => {
  const fronts = new Map<number, Set<number>>([
    [0, new Set([0, 1])],
    [1, new Set([0, 1])],
  ])
  expect(removeDominatedPrograms(fronts, [0.3, 0.7])).toEqual([1])
})

test("parent sampling is frequency-weighted over frontier instances", () => {
  const fronts = new Map<number, Set<number>>([
    [0, new Set([1])],
    [1, new Set([1])],
    [2, new Set([2])],
  ])
  const aggScores = [0, 0.9, 0.8]
  // candidate 1 sits on 2 fronts, candidate 2 on 1 → weights 2:1 over total 3
  expect(selectParetoParent(fronts, aggScores, () => 0.4)).toBe(1)
  expect(selectParetoParent(fronts, aggScores, () => 0.7)).toBe(2)
})

// --- Minibatch sampler -------------------------------------------------------

test("epoch-shuffled sampler: windows, least-frequent padding, reshuffle boundary", () => {
  // With rng()=0 the shuffle of [0..4] is deterministic: [1,2,3,4,0], padded
  // with the least-frequent (lowest) id 0 to [1,2,3,4,0,0].
  const sampler = createEpochShuffledSampler(zero, 5, 2)
  expect(sampler(0)).toEqual([1, 2])
  expect(sampler(1)).toEqual([3, 4])
  expect(sampler(2)).toEqual([0, 0])
  // iteration 3 crosses the epoch boundary → reshuffle, window restarts
  expect(sampler(3)).toEqual([1, 2])
})

test("sampler padding tie-break: least-frequent id with the LATEST first occurrence", () => {
  // rng()=0.9 leaves [0,1,2] unshuffled; among the all-tied counts Python's
  // most_common()[::-1] picks the latest-inserted id (2), not the lowest.
  const sampler = createEpochShuffledSampler(() => 0.9, 3, 2)
  expect(sampler(0)).toEqual([0, 1])
  expect(sampler(1)).toEqual([2, 2])
})

// --- Instruction extraction --------------------------------------------------

test("instruction extraction edge cases", () => {
  expect(extractInstructionText("no fences at all")).toBe("no fences at all")
  expect(
    extractInstructionText("Here:\n```text\nNew instructions\n```\nthanks")
  ).toBe("New instructions")
  expect(extractInstructionText("```\nNew instructions")).toBe(
    "New instructions"
  )
  expect(extractInstructionText("New instructions\n```")).toBe(
    "New instructions"
  )
  // Python's leading-fence regex has an optional newline: "```x" is consumed
  // entirely as fence + language tag, leaving an empty instruction.
  expect(extractInstructionText("```x")).toBe("")
})

// --- Engine: acceptance, cursor, budget --------------------------------------

function scriptedAdapter(
  scoreOf: (candidate: Candidate, example: Example) => number,
  propose: (component: string) => string,
  proposalLog?: string[][]
): GEPAAdapter {
  return {
    evaluate: (batch, candidate, captureTraces) => {
      const scores = batch.map((example) => scoreOf(candidate, example))
      return Promise.resolve({
        outputs: batch.map(() => ({})),
        scores,
        ...(captureTraces
          ? {
              trajectories: batch.map((example, k) => ({
                example,
                prediction: {},
                score: scores[k] as number,
                trace: [{ inputs: {}, outputs: {}, predictorName: "c1" }],
              })),
            }
          : {}),
      })
    },
    makeReflectiveDataset: (_candidate, _evalBatch, components) =>
      Promise.resolve(
        Object.fromEntries(
          components.map((name) => [
            name,
            [{ Feedback: "f", "Generated Outputs": {}, Inputs: {} }],
          ])
        )
      ),
    proposeNewTexts: (_candidate, _dataset, components) => {
      proposalLog?.push([...components])
      return Promise.resolve(
        Object.fromEntries(components.map((name) => [name, propose(name)]))
      )
    },
  }
}

function engineOptions(overrides: Partial<EngineOptions>): EngineOptions {
  return {
    candidateSelectionStrategy: "pareto",
    componentSelector: "roundRobin",
    maxMergeInvocations: 5,
    maxMetricCalls: 0,
    perfectScore: 1,
    reflectionMinibatchSize: 3,
    rng: zero,
    seedCandidate: { c1: "s1", c2: "s2" },
    skipPerfectScore: true,
    trainset: examples(3),
    useMerge: true,
    valset: examples(4),
    ...overrides,
  }
}

test("reflection acceptance is strict > on sums; cursor advances on rejection", async () => {
  const proposalLog: string[][] = []
  // Every candidate scores identically → child sum == parent sum → reject.
  const adapter = scriptedAdapter(
    () => 0.5,
    () => "changed",
    proposalLog
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 16 }))
  expect(state.programCandidates.length).toBe(1)
  // Round-robin advanced on the parent despite both rejections.
  expect(proposalLog).toEqual([["c1"], ["c2"]])
  expect(state.namedPredictorIdToUpdateNextForProgramCandidate[0]).toBe(0)
  // Rejected proposals still bill both minibatch evals: 4 + 2*(3+3) = 16.
  expect(state.totalNumEvals).toBe(16)
})

test("accepted child: full eval billed, discovery snapshot, cursor inheritance", async () => {
  const adapter = scriptedAdapter(
    (candidate) => (candidate.c1 === "better" ? 1 : 0.5),
    () => "better"
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 14 }))
  expect(state.programCandidates.length).toBe(2)
  expect(state.programCandidates[1]).toEqual({
    c1: "better",
    c2: "s2",
  })
  expect(state.parentProgramForCandidate[1]).toEqual([0])
  // Snapshot taken BEFORE billing the full eval: 4 seed + 3 + 3 = 10.
  expect(state.numMetricCallsByDiscovery[1]).toBe(10)
  expect(state.totalNumEvals).toBe(14)
  expect(state.numFullDSEvals).toBe(2)
  // Child inherits max(parent cursors) = 1 (parent advanced c1 → c2).
  expect(state.namedPredictorIdToUpdateNextForProgramCandidate[1]).toBe(1)
  // Frontier replaced by the strictly better child on every instance.
  for (const valId of [0, 1, 2, 3]) {
    expect([
      ...(state.programAtParetoFrontValset.get(valId) as Set<number>),
    ]).toEqual([1])
  }
})

test("empty proposed text still produces a real child that gets evaluated", async () => {
  // Python keeps an empty extraction as a proposal; only an empty proposal
  // dict skips the round — so the child minibatch eval is still billed.
  const adapter = scriptedAdapter(
    () => 0.5,
    () => ""
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 16 }))
  expect(state.programCandidates.length).toBe(1) // equal sums → rejected
  // 4 seed + 2 iterations × (3 parent + 3 child) = 16, child evals included.
  expect(state.totalNumEvals).toBe(16)
})

test("a fruitless merge search falls through to reflection in the same iteration", async () => {
  const adapter = scriptedAdapter(
    (candidate) => (candidate.c1 === "better" ? 1 : 0.5),
    () => "better"
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 26 }))
  // Iteration 0 accepts (14 evals) and schedules a merge. Iteration 1's
  // merge search finds only one frontier survivor → no proposal → the SAME
  // iteration still runs reflection (parent minibatch billed, then the
  // all-perfect skip). Every iteration after the first therefore bills 3:
  // 14, 17, 20, 23, 26 → five iterations, ending at i=4 (burning the merge
  // iteration would end at i=5).
  expect(state.totalNumEvals).toBe(26)
  expect(state.i).toBe(4)
})

test("budget is checked at the top of the loop only (bounded overshoot)", async () => {
  const adapter = scriptedAdapter(
    (candidate) => (candidate.c1 === "better" ? 1 : 0.5),
    () => "better"
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 5 }))
  // The single iteration runs to completion past the budget: 4+3+3+4 = 14.
  expect(state.totalNumEvals).toBe(14)
})

// --- Merge -------------------------------------------------------------------

function mergeLineageState(): GEPAState {
  const candidates: Candidate[] = [
    { c1: "s", c2: "s" },
    { c1: "A", c2: "s" },
    { c1: "s", c2: "B" },
  ]
  // The seed scores 0.25 everywhere: a zero-score ancestor would abort the
  // run upstream (random.choices raises on an all-zero weight total).
  const subscore = (idx: number, valId: number): number => {
    if (idx === 0) {
      return 0.25
    }
    const strongHalf = idx === 1 ? valId <= 2 : valId >= 3
    return strongHalf ? 1 : 0
  }
  const subscores = candidates.map(
    (_c, idx) =>
      new Map(
        Array.from(
          { length: 6 },
          (_2, valId) => [valId, subscore(idx, valId)] as [number, number]
        )
      )
  )
  const front = new Map<number, number>()
  const frontPrograms = new Map<number, Set<number>>()
  for (const [idx, scores] of subscores.entries()) {
    updateParetoFront(front, frontPrograms, idx, scores)
  }
  return {
    i: 0,
    namedPredictorIdToUpdateNextForProgramCandidate: [0, 0, 0],
    numFullDSEvals: 3,
    numMetricCallsByDiscovery: [0, 6, 12],
    parentProgramForCandidate: [[null], [0], [0]],
    paretoFrontValset: front,
    progCandidateValSubscores: subscores,
    programAtParetoFrontValset: frontPrograms,
    programCandidates: candidates,
    totalNumEvals: 18,
  }
}

test("merge candidate construction on a hand-built lineage", () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  const aggScores = [0.25, 0.5, 0.5]
  const proposal = proposeMerge(state, aggScores, zero, memory)
  expect(proposal).not.toBeNull()
  expect(proposal?.parentI).toBe(1)
  expect(proposal?.parentJ).toBe(2)
  expect(proposal?.ancestor).toBe(0)
  // Per-component rule: each lone divergence from the ancestor wins.
  expect(proposal?.candidate).toEqual({ c1: "A", c2: "B" })
  // The tried-triplet memo is the CALLER's job (recorded for the returned
  // proposal only) — the search itself must not memoize.
  expect(memory.triedTriplets.size).toBe(0)
})

test("all-zero ancestor weights abort the run (random.choices parity)", () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  // Ancestor aggregate 0 → weight total 0 → throws, like Python under
  // raise_on_exception=True.
  expect(() => proposeMerge(state, [0, 0.5, 0.5], zero, memory)).toThrow(
    "Total of weights must be greater than zero"
  )
})

test("merge subsample is balanced across better/worse/tied buckets", () => {
  const state = mergeLineageState()
  const sub = buildMergeSubsample(
    [0, 1, 2, 3, 4, 5],
    state.progCandidateValSubscores[1] as Map<number, number>,
    state.progCandidateValSubscores[2] as Map<number, number>,
    zero
  )
  expect(sub.length).toBe(5)
  expect(sub.filter((valId) => valId <= 2).length).toBeGreaterThanOrEqual(2)
  expect(sub.filter((valId) => valId >= 3).length).toBeGreaterThanOrEqual(2)
})

function mergeScoreAdapter(
  mergedScore: (valId: number) => number
): GEPAAdapter {
  return scriptedAdapter(
    (candidate, example) => {
      const valId = example.inputs.id as number
      const { c1, c2 } = candidate
      if (c1 === "A" && c2 === "B") {
        return mergedScore(valId)
      }
      if (c1 === "A") {
        return valId <= 2 ? 1 : 0
      }
      if (c2 === "B") {
        return valId >= 3 ? 1 : 0
      }
      return 0
    },
    () => ""
  )
}

test("merge acceptance is non-strict >= on subsample sums", async () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  // Merged scores exactly equal the better parent's on the subsample → accept.
  const adapter = mergeScoreAdapter((valId) => (valId >= 3 ? 1 : 0))
  const outcome = await runMergeIteration(
    adapter,
    state,
    { rng: zero, valset: examples(6) },
    memory
  )
  expect(outcome).toBe("accepted")
  expect(state.programCandidates.length).toBe(4)
  expect(state.parentProgramForCandidate[3]).toEqual([1, 2])
  // The triplet is memoized before the acceptance decision.
  expect(memory.triedTriplets.has("0|1|2")).toBe(true)
  // 18 + 5 (subsample) + 6 (full valset eval) = 29.
  expect(state.totalNumEvals).toBe(29)
})

test("rejected merge still bills its subsample eval", async () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  // Slightly below the better parent's sum → strictly less → reject.
  const adapter = mergeScoreAdapter((valId) => (valId >= 4 ? 1 : 0))
  const outcome = await runMergeIteration(
    adapter,
    state,
    { rng: zero, valset: examples(6) },
    memory
  )
  expect(outcome).toBe("rejected")
  expect(state.programCandidates.length).toBe(3)
  expect(state.totalNumEvals).toBe(23)
})

// --- Adapter: evaluate never throws ------------------------------------------

type MathPredictor = Predictor<
  z.ZodType<{ x: number }>,
  z.ZodType<{ y: number }>
>

function makeMathPredictor(name: string): MathPredictor {
  const predictor: MathPredictor = {
    call: (inputs, ctx) => {
      const { x } = inputs
      if (x === 3 && !predictor.instructions.includes("double")) {
        throw new Error("boom")
      }
      const y = predictor.instructions.includes("double") ? x * 2 : x
      ctx?.trace?.push({ inputs, outputs: { y }, predictorName: name })
      return Promise.resolve({ y })
    },
    clone: () => {
      const cloned = makeMathPredictor(name)
      cloned.instructions = predictor.instructions
      cloned.demos = structuredClone(predictor.demos)
      return cloned
    },
    demos: [],
    inputSchema: z.object({ x: z.number() }),
    instructions: "identity",
    model: "stub" as never,
    name,
    outputSchema: z.object({ y: z.number() }),
  }
  return predictor
}

function makeMathProgram(): Program<never, unknown> {
  const predictor = makeMathPredictor("math")
  return createProgram({
    forward: (call, input: Record<string, unknown>) => call("math", input),
    predictors: [predictor],
  }) as Program<never, unknown>
}

test("evaluate never throws per example: failures score failureScore with null output", async () => {
  const adapter = createProgramAdapter({
    adapterRNG: zero,
    addFormatFailureAsFeedback: false,
    failureScore: -1,
    metric: (_gold, prediction) => {
      if (!prediction) {
        throw new Error("no prediction")
      }
      return 1
    },
    program: makeMathProgram(),
    reflectionLM: () => Promise.resolve(""),
    warnOnScoreMismatch: true,
  })
  const batch: Example[] = [
    { inputs: { x: 1 }, outputs: { y: 1 } },
    { inputs: { x: 3 }, outputs: { y: 6 } },
  ]
  const result = await adapter.evaluate(batch, { math: "identity" }, false)
  expect(result.outputs).toEqual([{ y: 1 }, null])
  expect(result.scores).toEqual([1, -1])
})

// --- Demos flow through instruction-only candidates --------------------------

test("student demos survive gepa untouched; candidates stay instruction-only", async () => {
  const demo: Demo = { augmented: true, inputs: { x: 5 }, outputs: { y: 10 } }
  const program = makeMathProgram()
  ;(program.predictors[0] as Predictor).demos = [demo]
  const result = await gepa(program, examples(3), {
    maxMetricCalls: 1, // seed eval alone exceeds this → loop never runs
    metric: () => 0,
    reflectionLM: () => Promise.resolve(""),
  })
  // The candidate is a bare instruction map, exactly upstream's dict[str, str].
  expect(result.candidates[0]).toEqual({ math: "identity" })
  // The rebuilt best program still carries the pre-installed demos.
  expect(result.program.predictors[0]?.demos).toEqual([demo])
})

// --- End-to-end toy run ------------------------------------------------------

test("gepa e2e: budget enforced, best candidate beats or matches the seed", async () => {
  const trainset: Example[] = Array.from({ length: 6 }, (_, i) => ({
    inputs: { x: i + 1 },
    outputs: { y: (i + 1) * 2 },
  }))
  const result = await gepa(makeMathProgram(), trainset, {
    maxMetricCalls: 60,
    metric: (gold, prediction) => (prediction?.y === gold.outputs.y ? 1 : 0),
    reflectionLM: () =>
      Promise.resolve("```\nAlways double the input: return y = x * 2.\n```"),
    seed: 1,
  })
  expect(result.totalMetricCalls).toBeGreaterThanOrEqual(60)
  // Overshoot is bounded by one valset eval plus two minibatch evals.
  expect(result.totalMetricCalls).toBeLessThanOrEqual(60 + 6 + 2 * 3)
  expect(result.candidates.length).toBeGreaterThan(1)
  expect(result.bestIdx).toBeGreaterThan(0)
  expect(result.valAggregateScores[result.bestIdx]).toBeGreaterThanOrEqual(
    result.valAggregateScores[0] as number
  )
  expect(result.valAggregateScores[result.bestIdx]).toBe(1)
  // The returned program actually carries the improved instructions.
  expect(result.program.predictors[0]?.instructions.includes("double")).toBe(
    true
  )
})
