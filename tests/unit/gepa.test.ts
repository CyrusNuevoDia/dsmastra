import { expect, test } from "bun:test"

import { z } from "zod"

import type { Fields } from "@/fields"
import { autoBudget, gepa, gepaProgram } from "@/optimizers/gepa"
import {
  createProgramAdapter,
  extractInstructionText,
} from "@/optimizers/gepa/adapter"
import {
  buildMergeSubsample,
  createEpochShuffledSampler,
  proposeMerge,
  removeDominatedPrograms,
  runGEPA,
  runMergeIteration,
  selectParetoParent,
  updateParetoFront,
} from "@/optimizers/gepa/engine"
import type {
  Candidate,
  EngineOptions,
  GEPAAdapter,
  GEPAState,
} from "@/optimizers/gepa/engine"
import type { Prompts } from "@/optimizers/utils"
import { createProgram } from "@/program"
import type { Example, Program } from "@/program"
import type { AnyTunableStep } from "@/step"
import { createWorkflow } from "@/workflow"

const zero = () => 0

const examples = (n: number): Example[] =>
  Array.from({ length: n }, (_, id) => ({
    inputData: { id },
    outputData: {},
  }))

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
  expect([...(frontPrograms.get(0) as Set<number>)].toSorted()).toEqual([0, 1])
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

const scriptedAdapter = (
  scoreOf: (candidate: Candidate, example: Example) => number,
  propose: (component: string) => string,
  proposalLog?: string[][]
): GEPAAdapter => ({
  evaluate: (batch, candidate, captureTraces) => {
    const scores = batch.map((example) => scoreOf(candidate, example))
    return Promise.resolve({
      outputData: batch.map(() => ({})),
      scores,
      ...(captureTraces
        ? {
            trajectories: batch.map((example, k) => ({
              example,
              prediction: {},
              score: scores[k] as number,
              trace: [{ inputData: {}, outputData: {}, stepId: "c1" }],
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
})

const engineOptions = (overrides: Partial<EngineOptions>): EngineOptions => ({
  candidateSelectionStrategy: "pareto",
  componentSelector: "roundRobin",
  maxMergeInvocations: 5,
  maxMetricCalls: 0,
  perfectScore: 1,
  reflectionMinibatchSize: 3,
  rng: zero,
  seedCandidate: { c1: "s1", c2: "s2" },
  skipPerfectScore: true,
  trainingSet: examples(3),
  useMerge: true,
  validationSet: examples(4),
  ...overrides,
})

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
  expect(state.stepIdToUpdateNextForCandidate[0]).toBe(0)
  // Rejected proposals still bill both minibatch evals: 4 + 2*(3+3) = 16.
  expect(state.totalEvalsCount).toBe(16)
})

test("accepted child: full eval billed, discovery snapshot, cursor inheritance, onImprovement fires", async () => {
  const adapter = scriptedAdapter(
    (candidate) => (candidate.c1 === "better" ? 1 : 0.5),
    () => "better"
  )
  const improvements: Candidate[] = []
  const state = await runGEPA(
    adapter,
    engineOptions({
      maxMetricCalls: 14,
      onImprovement: (candidate) => {
        improvements.push(candidate)
        return Promise.resolve()
      },
    })
  )
  expect(state.programCandidates.length).toBe(2)
  expect(state.programCandidates[1]).toEqual({
    c1: "better",
    c2: "s2",
  })
  expect(state.parentProgramForCandidate[1]).toEqual([0])
  // Snapshot taken BEFORE billing the full eval: 4 seed + 3 + 3 = 10.
  expect(state.metricCallCountsByDiscovery[1]).toBe(10)
  expect(state.totalEvalsCount).toBe(14)
  expect(state.validationSetEvalsCount).toBe(2)
  // Child inherits max(parent cursors) = 1 (parent advanced c1 → c2).
  expect(state.stepIdToUpdateNextForCandidate[1]).toBe(1)
  // Frontier replaced by the strictly better child on every instance.
  for (const validationId of [0, 1, 2, 3]) {
    expect([
      ...(state.programAtParetoFrontValidationSet.get(
        validationId
      ) as Set<number>),
    ]).toEqual([1])
  }
  // The accepted child beat the seed's aggregate, so it checkpointed mid-run.
  expect(improvements).toEqual([{ c1: "better", c2: "s2" }])
})

test("empty proposed text still produces a real child that gets evaluated", async () => {
  // Python keeps an empty extraction as a proposal; only an empty proposal
  // dict skips the round — so the child minibatch eval is still billed.
  const adapter = scriptedAdapter(
    () => 0.5,
    () => ""
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 16 }))
  // Equal sums mean the merge is rejected, so no candidate is added.
  expect(state.programCandidates.length).toBe(1)
  // 4 seed + 2 iterations × (3 parent + 3 child) = 16, child evals included.
  expect(state.totalEvalsCount).toBe(16)
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
  expect(state.totalEvalsCount).toBe(26)
  expect(state.i).toBe(4)
})

test("budget is checked at the top of the loop only (bounded overshoot)", async () => {
  const adapter = scriptedAdapter(
    (candidate) => (candidate.c1 === "better" ? 1 : 0.5),
    () => "better"
  )
  const state = await runGEPA(adapter, engineOptions({ maxMetricCalls: 5 }))
  // The single iteration runs to completion past the budget: 4+3+3+4 = 14.
  expect(state.totalEvalsCount).toBe(14)
})

// --- Merge -------------------------------------------------------------------

const mergeLineageState = (): GEPAState => {
  const candidates: Candidate[] = [
    { c1: "s", c2: "s" },
    { c1: "A", c2: "s" },
    { c1: "s", c2: "B" },
  ]
  // The seed scores 0.25 everywhere: a zero-score ancestor would abort the
  // run upstream (random.choices raises on an all-zero weight total).
  const subscore = (idx: number, validationId: number): number => {
    if (idx === 0) {
      return 0.25
    }
    const strongHalf = idx === 1 ? validationId <= 2 : validationId >= 3
    return strongHalf ? 1 : 0
  }
  const subscores = candidates.map(
    (_c, idx) =>
      new Map(
        Array.from(
          { length: 6 },
          (_2, validationId) =>
            [validationId, subscore(idx, validationId)] as [number, number]
        )
      )
  )
  const front = new Map<number, number>()
  const frontPrograms = new Map<number, Set<number>>()
  for (const [idx, scores] of subscores.entries()) {
    updateParetoFront(front, frontPrograms, idx, scores)
  }
  return {
    candidateValidationSubscores: subscores,
    i: 0,
    metricCallCountsByDiscovery: [0, 6, 12],
    parentProgramForCandidate: [[null], [0], [0]],
    paretoFrontValidationSet: front,
    programAtParetoFrontValidationSet: frontPrograms,
    programCandidates: candidates,
    stepIdToUpdateNextForCandidate: [0, 0, 0],
    totalEvalsCount: 18,
    validationSetEvalsCount: 3,
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
    state.candidateValidationSubscores[1] as Map<number, number>,
    state.candidateValidationSubscores[2] as Map<number, number>,
    zero
  )
  expect(sub.length).toBe(5)
  expect(
    sub.filter((validationId) => validationId <= 2).length
  ).toBeGreaterThanOrEqual(2)
  expect(
    sub.filter((validationId) => validationId >= 3).length
  ).toBeGreaterThanOrEqual(2)
})

const mergeScoreAdapter = (
  mergedScore: (validationId: number) => number
): GEPAAdapter =>
  scriptedAdapter(
    (candidate, example) => {
      const validationId = example.inputData.id as number
      const { c1, c2 } = candidate
      if (c1 === "A" && c2 === "B") {
        return mergedScore(validationId)
      }
      if (c1 === "A") {
        return validationId <= 2 ? 1 : 0
      }
      if (c2 === "B") {
        return validationId >= 3 ? 1 : 0
      }
      return 0
    },
    () => ""
  )

test("merge acceptance is non-strict >= on subsample sums", async () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  // Merged scores exactly equal the better parent's on the subsample → accept.
  const adapter = mergeScoreAdapter((validationId) =>
    validationId >= 3 ? 1 : 0
  )
  const outcome = await runMergeIteration(
    adapter,
    state,
    { rng: zero, validationSet: examples(6) },
    memory
  )
  expect(outcome).toBe("accepted")
  expect(state.programCandidates.length).toBe(4)
  expect(state.parentProgramForCandidate[3]).toEqual([1, 2])
  // The triplet is memoized before the acceptance decision.
  expect(memory.triedTriplets.has("0|1|2")).toBe(true)
  // 18 + 5 (subsample) + 6 (full validationSet eval) = 29.
  expect(state.totalEvalsCount).toBe(29)
})

test("rejected merge still bills its subsample eval", async () => {
  const state = mergeLineageState()
  const memory = {
    producedByPair: new Set<string>(),
    triedTriplets: new Set<string>(),
  }
  // Slightly below the better parent's sum → strictly less → reject.
  const adapter = mergeScoreAdapter((validationId) =>
    validationId >= 4 ? 1 : 0
  )
  const outcome = await runMergeIteration(
    adapter,
    state,
    { rng: zero, validationSet: examples(6) },
    memory
  )
  expect(outcome).toBe("rejected")
  expect(state.programCandidates.length).toBe(3)
  expect(state.totalEvalsCount).toBe(23)
})

// --- Adapter: evaluate never throws ------------------------------------------

const makeMathStep = (id: string): AnyTunableStep => {
  const step: AnyTunableStep = {
    clone: () => {
      const cloned = makeMathStep(id)
      cloned.description = step.description
      cloned.examples = structuredClone(step.examples)
      return cloned
    },
    description: "identity",
    examples: [],
    execute: ({ inputData }, ctx) => {
      const x = inputData.x as number
      if (x === 3 && !step.description.includes("double")) {
        throw new Error("boom")
      }
      const y = step.description.includes("double") ? x * 2 : x
      ctx?.trace?.push({ inputData, outputData: { y }, stepId: id })
      return Promise.resolve({ y })
    },
    id,
    inputSchema: z.object({ x: z.number() }),
    model: "stub" as never,
    outputSchema: z.object({ y: z.number() }),
    settings: {},
  }
  return step
}

const makeMathProgram = (): Program =>
  createProgram({
    forward: (call, inputData: Fields) => call("math", inputData),
    steps: [makeMathStep("math")],
  })

test("evaluate never throws per example: failures score failureScore with null output", async () => {
  const adapter = createProgramAdapter({
    adapterRNG: zero,
    addFormatFailureAsFeedback: false,
    failureScore: -1,
    metric: (_gold, prediction) => {
      if (!prediction) {
        throw new Error("no prediction")
      }
      return { score: 1 }
    },
    program: makeMathProgram(),
    reflectionModel: () => Promise.resolve(""),
    warnOnScoreMismatch: true,
  })
  const batch: Example[] = [
    { inputData: { x: 1 }, outputData: { y: 1 } },
    { inputData: { x: 3 }, outputData: { y: 6 } },
  ]
  const result = await adapter.evaluate(batch, { math: "identity" }, false)
  expect(result.outputData).toEqual([{ y: 1 }, null])
  expect(result.scores).toEqual([1, -1])
})

// --- Examples flow through description-only candidates -----------------------

test("student examples survive gepa untouched; candidates stay description-only", async () => {
  const example: Example = { inputData: { x: 5 }, outputData: { y: 10 } }
  const program = makeMathProgram()
  ;(program.steps[0] as AnyTunableStep).examples = [example]
  const result = await gepaProgram(program, examples(3), {
    // The seed eval alone exceeds this, so the loop never runs.
    maxMetricCalls: 1,
    metric: () => ({ score: 0 }),
    reflectionModel: () => Promise.resolve(""),
  })
  // The candidate is a bare description map, exactly upstream's dict[str, str].
  expect(result.candidates[0]).toEqual({ math: "identity" })
  // The rebuilt best program still carries the pre-installed examples.
  expect(result.program.steps[0]?.examples).toEqual([example])
})

// --- End-to-end toy run ------------------------------------------------------

test("gepa e2e: budget enforced, best candidate beats or matches the seed", async () => {
  const trainingSet: Example[] = Array.from({ length: 6 }, (_, i) => ({
    inputData: { x: i + 1 },
    outputData: { y: (i + 1) * 2 },
  }))
  const result = await gepaProgram(makeMathProgram(), trainingSet, {
    maxMetricCalls: 60,
    metric: (gold, prediction) => ({
      score: prediction?.y === gold.outputData.y ? 1 : 0,
    }),
    reflectionModel: () =>
      Promise.resolve("```\nAlways double the input: return y = x * 2.\n```"),
    seed: 1,
  })
  expect(result.totalMetricCalls).toBeGreaterThanOrEqual(60)
  // Overshoot is bounded by one validationSet eval plus two minibatch evals.
  expect(result.totalMetricCalls).toBeLessThanOrEqual(60 + 6 + 2 * 3)
  expect(result.candidates.length).toBeGreaterThan(1)
  expect(result.bestIdx).toBeGreaterThan(0)
  expect(
    result.validationAggregateScores[result.bestIdx]
  ).toBeGreaterThanOrEqual(result.validationAggregateScores[0] as number)
  expect(result.validationAggregateScores[result.bestIdx]).toBe(1)
  // The returned program actually carries the improved description.
  expect(result.program.steps[0]?.description.includes("double")).toBe(true)
})

// --- Workflow-level gepa ------------------------------------------------------

const mathWorkflow = () =>
  createWorkflow({
    id: "math",
    inputSchema: z.object({ x: z.number() }),
    outputSchema: z.object({ y: z.number() }),
  })
    .then(makeMathStep("math"))
    .commit()

test("autoBudget matches hand-computed values", () => {
  // stepsCount=1, n=6, V=10: N = floor(max(4*log2(6), 9)) = 10
  // total = 10 + 30 + 350 + (floor(11/5)+1)*10 = 420
  expect(autoBudget(1, 6, 10)).toBe(420)
  // N<m branch: stepsCount=1, n=2: N = 4 < 5
  // total = 4 + 10 + 140 + (floor(5/5)+1+1)*4 = 166
  expect(autoBudget(1, 2, 4)).toBe(166)
})

test("autoBudget N=0 branch returns before periodic evals", () => {
  // n=0 → max(-Inf, 0) → N=0 → total = validationSetSize only
  expect(autoBudget(1, 0, 7)).toBe(7)
})

test("autoBudget guards throw", () => {
  expect(() => autoBudget(1, 2, -1)).toThrow()
  expect(() => autoBudget(1, 2, 4, -1)).toThrow()
  expect(() => autoBudget(1, 2, 4, 35, 0)).toThrow()
})

test("gepa budget knobs: exactly one of auto, maxFullEvals, maxMetricCalls", async () => {
  const savePrompts = () => Promise.resolve()
  await expect(
    gepa(mathWorkflow(), { savePrompts, trainingSet: examples(2) })
  ).rejects.toThrow(
    "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
  )
  await expect(
    gepa(mathWorkflow(), {
      maxFullEvals: 1,
      maxMetricCalls: 1,
      savePrompts,
      trainingSet: examples(2),
    })
  ).rejects.toThrow(
    "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
  )
})

test("gepa workflow: few-shot pre-pass runs un-billed, savePrompts checkpoints and finishes", async () => {
  const trainingSet: Example[] = Array.from({ length: 4 }, (_, i) => ({
    inputData: { x: i + 4 },
    outputData: { y: (i + 4) * 2 },
  }))
  let metricCalls = 0
  const saved: Prompts[] = []
  const { score, workflow: tuned } = await gepa(mathWorkflow(), {
    // Budget of 1: GEPA's own loop never runs (the seed eval exhausts it) —
    // yet the pre-pass still bootstrapped examples, proving its metric calls
    // are not billed against GEPA's budget.
    maxFewShotExamples: 2,
    maxMetricCalls: 1,
    metric: (gold, prediction) => {
      metricCalls += 1
      return { score: prediction?.y === gold.outputData.y ? 1 : 0 }
    },
    reflectionModel: () => Promise.resolve(""),
    savePrompts: (prompts) => {
      saved.push(structuredClone(prompts))
      return Promise.resolve()
    },
    trainingSet,
  })
  // Identity math step: x∈[4..7] all fail (y=x ≠ 2x)? No — identity returns
  // y=x, expected 2x, so bootstrap accepts none and backfills labeled ones.
  const tunedStep = tuned.steps.math as AnyTunableStep
  expect(tunedStep.examples.length).toBeGreaterThan(0)
  expect(score).toBe(0)
  // Final savePrompts always fires, carrying the tuned state.
  expect(saved.length).toBeGreaterThanOrEqual(1)
  const last = saved.at(-1) as Prompts
  expect(last.version).toBe(1)
  expect(last.steps.math?.examples).toEqual(tunedStep.examples)
  // The metric ran for the bootstrap attempts plus the seed eval — more calls
  // than GEPA's entire budget, none of which aborted the pre-pass.
  expect(metricCalls).toBeGreaterThan(1)
})
