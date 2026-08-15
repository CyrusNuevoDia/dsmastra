import { expect, test } from "bun:test"

import type { Fields } from "#src/fields"
import {
  bootstrapFewShotPrompts,
  labeledFewShotPrompts,
} from "#src/optimizers/bootstrap"
import { gepaPrompts } from "#src/optimizers/gepa/index"
import { promptsOf } from "#src/optimizers/utils"
import type { Prompts } from "#src/optimizers/utils"
import type { Example, RunContext } from "#src/step"
import { fakeStep, fakeWorkflow, promptStep } from "#tests/unit/helpers"
import type { Call } from "#tests/unit/helpers"

const singleWorkflow = (
  fn: (
    inputData: Record<string, unknown>,
    ctx?: RunContext
  ) => Record<string, unknown>,
  log: Call[] = []
): ReturnType<typeof fakeWorkflow> => fakeWorkflow(fakeStep("solve", fn, log))

const dataset = (xs: number[]): Example[] =>
  xs.map((x) => ({ inputData: { x }, outputData: { y: x * 2 } }))

const exactMetric = (gold: Example, prediction: Fields | null) => ({
  score: prediction?.y === gold.outputData.y ? 1 : 0,
})

const solve = (prompts: Prompts) => promptStep(prompts, "solve")

// --- LabeledFewShot ----------------------------------------------------------

test("labeledFewShot installs k labeled examples on a reset copy", () => {
  const workflow = singleWorkflow((inputData) => ({ y: inputData.x }))
  const prompts = promptsOf(workflow)
  solve(prompts).examples = [{ inputData: { x: 0 }, outputData: { y: 0 } }]
  const trainingSet = dataset([1, 2, 3, 4, 5])
  const compiled = labeledFewShotPrompts(prompts, trainingSet, 3)
  const { examples } = solve(compiled)
  expect(examples.length).toBe(3)
  // Reset copy: the pre-existing example was cleared, not kept alongside.
  expect(examples.every((e) => (e.inputData.x as number) >= 1)).toBe(true)
  // The source prompts are untouched.
  expect(solve(prompts).examples.length).toBe(1)
})

test("labeledFewShot caps k at the trainingSet size and handles empty sets", () => {
  const workflow = singleWorkflow((inputData) => ({ y: inputData.x }))
  const prompts = promptsOf(workflow)
  expect(
    solve(labeledFewShotPrompts(prompts, dataset([1, 2]), 16)).examples.length
  ).toBe(2)
  expect(solve(labeledFewShotPrompts(prompts, [], 16)).examples.length).toBe(0)
})

// --- Bootstrap: example capture and acceptance --------------------------------

test("bootstrapped examples only come from metric-passing traces", async () => {
  // Only even x produces the right answer.
  const workflow = singleWorkflow((inputData) => ({
    y: (inputData.x as number) % 2 === 0 ? (inputData.x as number) * 2 : 0,
  }))
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([1, 2, 3, 4]),
    {
      maxLabeledExamples: 0,
      metric: exactMetric,
    }
  )
  const { examples } = solve(compiled)
  expect(
    examples.map((e) => e.inputData.x as number).toSorted((a, b) => a - b)
  ).toEqual([2, 4])
  expect(
    examples.map((e) => e.outputData.y as number).toSorted((a, b) => a - b)
  ).toEqual([4, 8])
})

test("without a threshold, only a score above zero is accepted", async () => {
  const workflow = singleWorkflow((inputData) => ({ y: inputData.x }))
  // Upstream DSPy tests `bool(metric_val)`, which counts a negative score as a
  // pass. We test `score > 0` instead, so a penalty score is a failure.
  const signedScore = (gold: Example) => ({
    score: (gold.inputData.x as number) - 2,
  })
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([1, 2, 3]),
    {
      maxLabeledExamples: 0,
      metric: signedScore,
    }
  )
  // x=1 scores -1 and x=2 scores 0; only x=3, scoring 1, is bootstrapped.
  expect(solve(compiled).examples.map((e) => e.inputData.x)).toEqual([3])
})

test("no metric accepts every trace", async () => {
  const workflow = singleWorkflow(() => ({ y: 0 }))
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([1, 2, 3]),
    {
      maxFewShotExamples: 2,
      maxLabeledExamples: 0,
    }
  )
  expect(solve(compiled).examples.length).toBe(2)
})

test("metricThreshold switches acceptance from truthiness to >= threshold", async () => {
  const workflow = singleWorkflow((inputData) => ({ y: inputData.x }))
  const partialCredit = (gold: Example) => ({
    score: (gold.inputData.x as number) === 1 ? 0.4 : 0.6,
  })
  // Without a threshold every score above zero is accepted.
  const loose = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([1, 2]),
    {
      maxLabeledExamples: 0,
      metric: partialCredit,
    }
  )
  expect(solve(loose).examples.length).toBe(2)
  // With threshold 0.5 only the 0.6-scoring example passes.
  const strict = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([1, 2]),
    {
      maxLabeledExamples: 0,
      metric: partialCredit,
      metricThreshold: 0.5,
    }
  )
  expect(solve(strict).examples.map((e) => e.inputData.x)).toEqual([2])
})

// --- Caps and labeled backfill -----------------------------------------------

test("maxFewShotExamples caps bootstrapped EXAMPLES; labeled backfill fills to maxLabeledExamples", async () => {
  const workflow = singleWorkflow((inputData) => ({
    y: (inputData.x as number) * 2,
  }))
  const trainingSet = dataset([1, 2, 3, 4, 5, 6])
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    trainingSet,
    {
      maxFewShotExamples: 2,
      maxLabeledExamples: 5,
      metric: exactMetric,
    }
  )
  const { examples } = solve(compiled)
  // Bootstrapped examples come first: the first two trainingSet examples
  // bootstrap, then the loop stops.
  const bootstrapped = examples.slice(0, 2)
  const labeled = examples.slice(2)
  expect(bootstrapped.map((e) => e.inputData.x)).toEqual([1, 2])
  // Backfill: min(maxLabeled - bootstrapped, |unbootstrapped|) = min(3, 4) = 3,
  // drawn only from the un-bootstrapped examples.
  expect(labeled.length).toBe(3)
  for (const example of labeled) {
    expect([3, 4, 5, 6]).toContain(example.inputData.x as number)
    expect(example.outputData.y).toBe((example.inputData.x as number) * 2)
  }
})

test("later steps backfill from the shrinking labeled pool (Python quirk)", async () => {
  const logA: Call[] = []
  const workflow = fakeWorkflow(
    fakeStep("a", (inputData) => ({ mid: inputData.x }), logA),
    fakeStep("b", (inputData) => ({ y: inputData.mid }))
  )
  const trainingSet = dataset([2, 4, 6, 8, 10]).map((e) => ({
    ...e,
    outputData: { y: e.inputData.x },
  }))
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    trainingSet,
    {
      maxFewShotExamples: 1,
      maxLabeledExamples: 3,
      metric: exactMetric,
    }
  )
  const examplesA = promptStep(compiled, "a").examples
  const examplesB = promptStep(compiled, "b").examples
  // One bootstrapped example leads; the rest are labeled backfill. Labeled
  // examples are raw trainingSet rows, so they carry a `y` output — the
  // bootstrapped trace for step a carries `mid` instead.
  const labeledOf = (examples: Example[] | undefined) =>
    (examples ?? [])
      .filter((e) => "y" in e.outputData)
      .map((e) => JSON.stringify(e))
  const labeledA = new Set(labeledOf(examplesA))
  expect(labeledOf(examplesA).length).toBe(2)
  // rawExamples is reassigned to each sample: b's labeled examples are a
  // subset of a's... except b's bootstrapped trace ALSO carries `y`, so drop
  // the leading bootstrapped entry first.
  const labeledB = labeledOf(examplesB?.slice(1))
  expect(labeledB.length).toBe(2)
  for (const example of labeledB) {
    expect(labeledA.has(example)).toBe(true)
  }
})

// --- maxRounds and the cache-bypass rollout ----------------------------------

test("maxRounds retries with a fresh rollout: seed=roundIdx, temperature=1", async () => {
  const log: Call[] = []
  // Succeeds only on the round-1 rollout (ctx.seed === 1).
  const workflow = singleWorkflow(
    (inputData, ctx) =>
      ctx?.seed === 1 ? { y: (inputData.x as number) * 2 } : { y: 0 },
    log
  )
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    dataset([7]),
    {
      maxLabeledExamples: 0,
      maxRounds: 3,
      metric: exactMetric,
    }
  )
  expect(solve(compiled).examples.length).toBe(1)
  expect(log.length).toBe(2)
  // Round 0: no override. Round 1: rollout id as seed, temperature 1.0.
  expect(log[0]?.ctx?.seed).toBeUndefined()
  expect(log[0]?.ctx?.temperature).toBeUndefined()
  expect(log[1]?.ctx?.seed).toBe(1)
  expect(log[1]?.ctx?.temperature).toBe(1)
})

test("teacherSettings override the teacher run; round overrides win on temperature", async () => {
  const log: Call[] = []
  const workflow = singleWorkflow(
    (inputData, ctx) =>
      ctx?.seed === 1 ? { y: (inputData.x as number) * 2 } : { y: 0 },
    log
  )
  await bootstrapFewShotPrompts(workflow, promptsOf(workflow), dataset([1]), {
    maxLabeledExamples: 0,
    maxRounds: 2,
    metric: exactMetric,
    teacherSettings: { model: "teacher-model" as never, temperature: 0.3 },
  })
  expect(log[0]?.ctx?.model).toBe("teacher-model" as never)
  expect(log[0]?.ctx?.temperature).toBe(0.3)
  expect(log[1]?.ctx?.temperature).toBe(1)
})

// --- Teacher/student separation and error accounting -------------------------

test("student and source stay unmutated; examples matching the example are hidden from the teacher", async () => {
  const workflow = singleWorkflow((inputData) => ({
    y: (inputData.x as number) * 2,
  }))
  const preExisting: Example = { inputData: { x: 1 }, outputData: { y: 2 } }
  const prompts = promptsOf(workflow)
  solve(prompts).examples = [preExisting]
  const compiled = await bootstrapFewShotPrompts(
    workflow,
    prompts,
    dataset([1]),
    {
      // Labeled pass installs examples on the teacher; the example equal to the
      // in-flight one must be hidden during its own bootstrap run.
      maxLabeledExamples: 1,
      metric: exactMetric,
    }
  )
  // Source prompts untouched (bootstrap works on copies).
  expect(solve(prompts).examples).toEqual([preExisting])
  // Student was reset and rebuilt: it got the bootstrapped trace example.
  expect(solve(compiled).examples.length).toBe(1)
  expect(compiled).not.toBe(prompts)
})

test("errors are counted per attempt and rethrown at maxErrors", async () => {
  let calls = 0
  const workflow = singleWorkflow(() => ({ y: 0 }))
  const failingMetric = () => {
    calls += 1
    throw new Error("boom")
  }
  // maxErrors 2: first failure is swallowed, second rethrows.
  await expect(
    bootstrapFewShotPrompts(workflow, promptsOf(workflow), dataset([1, 2, 3]), {
      maxErrors: 2,
      maxLabeledExamples: 0,
      metric: failingMetric,
    })
  ).rejects.toThrow("boom")
  expect(calls).toBe(2)
})

// --- GEPA composition ---------------------------------------------------------

test("bootstrap → gepa: examples ride the program, candidates stay description-only", async () => {
  const workflow = singleWorkflow((inputData) => ({
    y: (inputData.x as number) * 2,
  }))
  const trainingSet = dataset([1, 2, 3, 4])
  const student = await bootstrapFewShotPrompts(
    workflow,
    promptsOf(workflow),
    trainingSet,
    {
      maxFewShotExamples: 2,
      maxLabeledExamples: 2,
      metric: exactMetric,
    }
  )
  const installed = structuredClone(solve(student).examples)
  const result = await gepaPrompts(workflow, student, trainingSet, {
    // The seed eval exhausts the budget, so the loop never runs.
    maxMetricCalls: 1,
    metric: (gold, prediction) => ({
      score: prediction?.y === gold.outputData.y ? 1 : 0,
    }),
    reflectionModel: () => Promise.resolve(""),
  })
  // Candidates are pure description maps.
  expect(result.candidates[0]).toEqual({ solve: "solve" })
  // The returned best program still carries the bootstrapped examples.
  expect(solve(result.prompts).examples).toEqual(installed as Example[])
})
