import { expect, test } from "bun:test"

import { z } from "zod"

import { bootstrapFewShot, labeledFewShot } from "@/bootstrap"
import type { Fields } from "@/fields"
import { gepa } from "@/gepa"
import type { Demo, Predictor, RunContext } from "@/predictor"
import { createProgram } from "@/program"
import type { Example, Program } from "@/program"

type Call = { ctx?: RunContext; inputs: Record<string, unknown> }

/**
 * Deterministic fake predictor: `fn` maps inputs (+ctx) to outputs, calls are
 * logged, clones share the log and behavior but copy demos/instructions.
 */
const fakePredictor = (
  name: string,
  fn: (
    inputs: Record<string, unknown>,
    ctx?: RunContext
  ) => Record<string, unknown>,
  log: Call[] = []
): Predictor => {
  const predictor: Predictor = {
    call: (inputs: Record<string, unknown>, ctx?: RunContext) => {
      log.push({ ctx, inputs })
      const outputs = fn(inputs, ctx)
      ctx?.trace?.push({ inputs, outputs, predictorName: name })
      return Promise.resolve(outputs)
    },
    clone: () => {
      const cloned = fakePredictor(name, fn, log)
      cloned.demos = structuredClone(predictor.demos)
      cloned.instructions = predictor.instructions
      return cloned
    },
    demos: [],
    inputSchema: z.record(z.string(), z.unknown()),
    instructions: "solve",
    model: "stub" as never,
    name,
    outputSchema: z.record(z.string(), z.unknown()),
  }
  return predictor
}

const singleProgram = (
  fn: (
    inputs: Record<string, unknown>,
    ctx?: RunContext
  ) => Record<string, unknown>,
  log: Call[] = []
): Program<Record<string, unknown>, Record<string, unknown>> =>
  createProgram({
    forward: (call, input: Record<string, unknown>) => call("solve", input),
    predictors: [fakePredictor("solve", fn, log)],
  })

const dataset = (xs: number[]): Example[] =>
  xs.map((x) => ({ inputs: { x }, outputs: { y: x * 2 } }))

const exactMetric = (gold: Example, prediction: Fields | null) => ({
  score: prediction?.y === gold.outputs.y ? 1 : 0,
})

// --- LabeledFewShot ----------------------------------------------------------

test("labeledFewShot installs k labeled demos on a reset copy", () => {
  const program = singleProgram((inputs) => ({ y: inputs.x }))
  ;(program.predictors[0] as Predictor).demos = [
    { inputs: { x: 0 }, outputs: { y: 0 } },
  ]
  const trainset = dataset([1, 2, 3, 4, 5])
  const compiled = labeledFewShot(program, trainset, 3)
  const demos = compiled.predictors[0]?.demos as Demo[]
  expect(demos.length).toBe(3)
  // Reset copy: the pre-existing demo was cleared, not kept alongside.
  expect(demos.every((d) => (d.inputs.x as number) >= 1)).toBe(true)
  // Labeled demos are raw examples, never marked augmented.
  expect(demos.every((d) => d.augmented === undefined)).toBe(true)
  // The source program is untouched.
  expect(program.predictors[0]?.demos.length).toBe(1)
})

test("labeledFewShot caps k at the trainset size and handles empty trainsets", () => {
  const program = singleProgram((inputs) => ({ y: inputs.x }))
  expect(
    labeledFewShot(program, dataset([1, 2]), 16).predictors[0]?.demos.length
  ).toBe(2)
  expect(labeledFewShot(program, [], 16).predictors[0]?.demos.length).toBe(0)
})

// --- Bootstrap: demo capture and acceptance ----------------------------------

test("bootstrapped demos only come from metric-passing traces", async () => {
  // Only even x produces the right answer.
  const program = singleProgram((inputs) => ({
    y: (inputs.x as number) % 2 === 0 ? (inputs.x as number) * 2 : 0,
  }))
  const compiled = await bootstrapFewShot(program, dataset([1, 2, 3, 4]), {
    maxLabeledDemos: 0,
    metric: exactMetric,
  })
  const demos = compiled.predictors[0]?.demos as Demo[]
  expect(
    demos.map((d) => d.inputs.x as number).toSorted((a, b) => a - b)
  ).toEqual([2, 4])
  expect(demos.every((d) => d.augmented === true)).toBe(true)
  expect(
    demos.map((d) => d.outputs.y as number).toSorted((a, b) => a - b)
  ).toEqual([4, 8])
})

test("without a threshold, only a score above zero is accepted", async () => {
  const program = singleProgram((inputs) => ({ y: inputs.x }))
  // Upstream DSPy tests `bool(metric_val)`, which counts a negative score as a
  // pass. We test `score > 0` instead, so a penalty score is a failure.
  const signedScore = (gold: Example) => ({
    score: (gold.inputs.x as number) - 2,
  })
  const compiled = await bootstrapFewShot(program, dataset([1, 2, 3]), {
    maxLabeledDemos: 0,
    metric: signedScore,
  })
  // x=1 scores -1 and x=2 scores 0; only x=3, scoring 1, is bootstrapped.
  expect(compiled.predictors[0]?.demos.map((d) => d.inputs.x)).toEqual([3])
})

test("no metric accepts every trace", async () => {
  const program = singleProgram(() => ({ y: 0 }))
  const compiled = await bootstrapFewShot(program, dataset([1, 2, 3]), {
    maxBootstrappedDemos: 2,
    maxLabeledDemos: 0,
  })
  expect(compiled.predictors[0]?.demos.length).toBe(2)
})

test("metricThreshold switches acceptance from truthiness to >= threshold", async () => {
  const program = singleProgram((inputs) => ({ y: inputs.x }))
  const partialCredit = (gold: Example) => ({
    score: (gold.inputs.x as number) === 1 ? 0.4 : 0.6,
  })
  // Without a threshold every score above zero is accepted.
  const loose = await bootstrapFewShot(program, dataset([1, 2]), {
    maxLabeledDemos: 0,
    metric: partialCredit,
  })
  expect(loose.predictors[0]?.demos.length).toBe(2)
  // With threshold 0.5 only the 0.6-scoring example passes.
  const strict = await bootstrapFewShot(program, dataset([1, 2]), {
    maxLabeledDemos: 0,
    metric: partialCredit,
    metricThreshold: 0.5,
  })
  expect(strict.predictors[0]?.demos.map((d) => d.inputs.x)).toEqual([2])
})

// --- Caps and labeled backfill -----------------------------------------------

test("maxBootstrappedDemos caps bootstrapped EXAMPLES; labeled backfill fills to maxLabeledDemos", async () => {
  const program = singleProgram((inputs) => ({ y: (inputs.x as number) * 2 }))
  const trainset = dataset([1, 2, 3, 4, 5, 6])
  const compiled = await bootstrapFewShot(program, trainset, {
    maxBootstrappedDemos: 2,
    maxLabeledDemos: 5,
    metric: exactMetric,
  })
  const demos = compiled.predictors[0]?.demos as Demo[]
  const augmented = demos.filter((d) => d.augmented)
  const labeled = demos.filter((d) => !d.augmented)
  // First two trainset examples bootstrap, then the loop stops.
  expect(augmented.map((d) => d.inputs.x)).toEqual([1, 2])
  // Backfill: min(maxLabeled - aug, |unbootstrapped|) = min(3, 4) = 3, drawn
  // only from the un-bootstrapped examples, ordered aug-first.
  expect(labeled.length).toBe(3)
  expect(demos.slice(0, 2)).toEqual(augmented)
  for (const demo of labeled) {
    expect([3, 4, 5, 6]).toContain(demo.inputs.x as number)
    expect(demo.outputs.y).toBe((demo.inputs.x as number) * 2)
  }
})

test("later predictors backfill from the shrinking labeled pool (Python quirk)", async () => {
  const logA: Call[] = []
  const predictors = [
    fakePredictor("a", (inputs) => ({ mid: inputs.x }), logA),
    fakePredictor("b", (inputs) => ({ y: inputs.mid })),
  ]
  const program = createProgram({
    forward: async (call, input: Record<string, unknown>) => {
      const mid = await call("a", input)
      return call("b", mid)
    },
    predictors,
  })
  const trainset = dataset([2, 4, 6, 8, 10]).map((e) => ({
    ...e,
    outputs: { y: e.inputs.x },
  }))
  const compiled = await bootstrapFewShot(program, trainset, {
    maxBootstrappedDemos: 1,
    maxLabeledDemos: 3,
    metric: exactMetric,
  })
  const [demosA, demosB] = compiled.predictors.map((p) => p.demos)
  const labeledOf = (demos: Demo[] | undefined) =>
    (demos ?? []).filter((d) => !d.augmented).map((d) => JSON.stringify(d))
  // rawDemos is reassigned to each sample: b's labeled demos are a subset of a's.
  const labeledA = new Set(labeledOf(demosA))
  expect(labeledOf(demosA).length).toBe(2)
  expect(labeledOf(demosB).length).toBe(2)
  for (const demo of labeledOf(demosB)) {
    expect(labeledA.has(demo)).toBe(true)
  }
})

// --- maxRounds and the cache-bypass rollout ----------------------------------

test("maxRounds retries with a fresh rollout: seed=roundIdx, temperature=1", async () => {
  const log: Call[] = []
  // Succeeds only on the round-1 rollout (ctx.seed === 1).
  const program = singleProgram(
    (inputs, ctx) =>
      ctx?.seed === 1 ? { y: (inputs.x as number) * 2 } : { y: 0 },
    log
  )
  const compiled = await bootstrapFewShot(program, dataset([7]), {
    maxLabeledDemos: 0,
    maxRounds: 3,
    metric: exactMetric,
  })
  expect(compiled.predictors[0]?.demos.length).toBe(1)
  expect(log.length).toBe(2)
  // Round 0: no override. Round 1: rollout id as seed, temperature 1.0.
  expect(log[0]?.ctx?.seed).toBeUndefined()
  expect(log[0]?.ctx?.temperature).toBeUndefined()
  expect(log[1]?.ctx?.seed).toBe(1)
  expect(log[1]?.ctx?.temperature).toBe(1)
})

test("teacherSettings override the teacher run; round overrides win on temperature", async () => {
  const log: Call[] = []
  const program = singleProgram(
    (inputs, ctx) =>
      ctx?.seed === 1 ? { y: (inputs.x as number) * 2 } : { y: 0 },
    log
  )
  await bootstrapFewShot(program, dataset([1]), {
    maxLabeledDemos: 0,
    maxRounds: 2,
    metric: exactMetric,
    teacherSettings: { model: "teacher-model" as never, temperature: 0.3 },
  })
  expect(log[0]?.ctx?.model).toBe("teacher-model" as never)
  expect(log[0]?.ctx?.temperature).toBe(0.3)
  expect(log[1]?.ctx?.temperature).toBe(1)
})

// --- Teacher/student separation and error accounting -------------------------

test("student and source stay unmutated; demos matching the example are hidden from the teacher", async () => {
  const program = singleProgram((inputs) => ({ y: (inputs.x as number) * 2 }))
  const preExisting: Demo = { inputs: { x: 1 }, outputs: { y: 2 } }
  ;(program.predictors[0] as Predictor).demos = [preExisting]
  const compiled = await bootstrapFewShot(program, dataset([1]), {
    // Labeled pass installs demos on the teacher; the demo equal to the
    // in-flight example must be hidden during its own bootstrap run.
    maxLabeledDemos: 1,
    metric: exactMetric,
  })
  // Source program untouched (bootstrap works on copies).
  expect(program.predictors[0]?.demos).toEqual([preExisting])
  // Student was reset: its demos are freshly built, augmented first.
  expect(compiled.predictors[0]?.demos[0]?.augmented).toBe(true)
  expect(compiled).not.toBe(program)
})

test("errors are counted per attempt and rethrown at maxErrors", async () => {
  let calls = 0
  const program = singleProgram(() => {
    calls += 1
    throw new Error("boom")
  })
  // maxErrors 2: first failure is swallowed, second rethrows.
  await expect(
    bootstrapFewShot(program, dataset([1, 2, 3]), {
      maxErrors: 2,
      maxLabeledDemos: 0,
      metric: exactMetric,
    })
  ).rejects.toThrow("boom")
  expect(calls).toBe(2)
})

// --- GEPA composition ---------------------------------------------------------

test("bootstrap → gepa: demos ride the program, candidates stay instruction-only", async () => {
  const program = singleProgram((inputs) => ({ y: (inputs.x as number) * 2 }))
  const trainset = dataset([1, 2, 3, 4])
  const student = await bootstrapFewShot(program, trainset, {
    maxBootstrappedDemos: 2,
    maxLabeledDemos: 2,
    metric: exactMetric,
  })
  const installed = structuredClone(student.predictors[0]?.demos)
  const result = await gepa(student, trainset, {
    // The seed eval exhausts the budget, so the loop never runs.
    maxMetricCalls: 1,
    metric: (gold, prediction) => ({
      score: prediction?.y === gold.outputs.y ? 1 : 0,
    }),
    reflectionLM: () => Promise.resolve(""),
  })
  // Candidates are pure instruction maps.
  expect(result.candidates[0]).toEqual({ solve: "solve" })
  // The returned best program still carries the bootstrapped demos.
  expect(result.program.predictors[0]?.demos).toEqual(installed as Demo[])
})
