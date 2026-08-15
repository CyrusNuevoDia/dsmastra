import { expect, test } from "bun:test"

import { createScorer } from "@mastra/core/evals"
import type { Trajectory } from "@mastra/core/evals"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { bootstrapFewShot } from "@/optimizers/bootstrap-few-shot"

import { fakeScorer, fakeStep } from "./helpers"

const fieldsSchema = z.record(z.string(), z.unknown())

/**
 * One tunable step whose outputs carry a `note` marker absent from the gold
 * outputData, so trace-harvested demos are distinguishable from labeled
 * backfill in the compiled examples.
 */
const makeWorkflow = () => {
  const step = fakeStep("solve", (inputData) => ({
    note: "t",
    y: (inputData.x as number) * 2,
  }))
  const workflow = createWorkflow({
    id: "bootstrap-wf",
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  })
    .then(step)
    .commit()
  return { step, workflow }
}

const trainingSet = [
  { inputData: { x: 1 }, outputData: { y: 2 } },
  { inputData: { x: 2 }, outputData: { y: 4 } },
]

const augmentedCount = (examples: { outputData: Record<string, unknown> }[]) =>
  examples.filter((example) => example.outputData.note === "t").length

/** A Mastra trajectory scorer scoring the rollout trace it receives. */
const gateScorer = (score: (trajectory: Trajectory) => number) =>
  createScorer({
    description: "trajectory gate",
    id: "gate",
    type: "trajectory",
  }).generateScore(({ run }) => score(run.output))

const firstStepOutput = (trajectory: Trajectory) => {
  const [step] = trajectory.steps
  return step?.stepType === "workflow_step" ? step.output : undefined
}

test("without a gate, the objective scorer gates teacher-trace acceptance", async () => {
  const { step, workflow } = makeWorkflow()
  let objectiveCalls = 0
  const objective = fakeScorer((gold, prediction) => {
    objectiveCalls += 1
    return prediction?.y === gold.outputData.y ? 1 : 0
  })

  const { score } = await bootstrapFewShot(workflow, {
    savePrompts: () => Promise.resolve(),
    scorer: objective,
    trainingSet,
  })

  // Both teacher rollouts match gold on y, so both traces become demos.
  expect(augmentedCount(step.examples)).toBe(2)
  expect(step.examples).toHaveLength(2)
  // Two bootstrap gate calls plus two final-evaluation calls.
  expect(objectiveCalls).toBe(4)
  expect(score).toBe(1)
})

test("a gate scores the trajectory and decides acceptance instead of the objective", async () => {
  const { step, workflow } = makeWorkflow()
  let objectiveCalls = 0
  const objective = fakeScorer(() => {
    objectiveCalls += 1
    return 1
  })
  const seen: Trajectory[] = []
  const gate = gateScorer((trajectory) => {
    seen.push(trajectory)
    return firstStepOutput(trajectory)?.y === 2 ? 1 : 0
  })

  const { score } = await bootstrapFewShot(workflow, {
    gate: { scorer: gate },
    savePrompts: () => Promise.resolve(),
    scorer: objective,
    trainingSet,
  })

  // Only x=1's trace (y === 2) passes the gate, even though the objective
  // scorer would have accepted everything; x=2 backfills as a labeled example.
  expect(augmentedCount(step.examples)).toBe(1)
  expect(step.examples).toHaveLength(2)
  // The objective scorer never gated — it only ran the final evaluation pass.
  expect(objectiveCalls).toBe(2)
  expect(score).toBe(1)
  // The gate received the engine-derived Trajectory: workflow_step entries
  // with status and output, plus the raw step results carrying each input.
  expect(seen).toHaveLength(2)
  expect(seen[0]).toMatchObject({
    steps: [
      {
        name: "solve",
        output: { note: "t", y: 2 },
        status: "success",
        stepId: "solve",
        stepType: "workflow_step",
      },
    ],
  })
  // The workflow input rides along under the engine's reserved "input" entry
  // (typed as a StepResult, but holding the raw input at run time).
  const workflowInput: unknown = seen[0]?.rawWorkflowResult?.stepResults.input
  expect(workflowInput).toEqual({ x: 1 })
})

test("non-tunable steps appear in the gate's trajectory but never as demos", async () => {
  const solve = fakeStep("solve", (inputData) => ({
    note: "t",
    y: (inputData.x as number) * 2,
  }))
  const plain = createStep({
    execute: ({ inputData }) =>
      Promise.resolve({ z: (inputData.y as number) + 1 }),
    id: "plain",
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  })
  const workflow = createWorkflow({
    id: "bootstrap-mixed-wf",
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  })
    .then(solve)
    .then(plain)
    .commit()
  const seen: Trajectory[] = []
  const gate = gateScorer((trajectory) => {
    seen.push(trajectory)
    return 1
  })

  await bootstrapFewShot(workflow, {
    gate: { scorer: gate },
    savePrompts: () => Promise.resolve(),
    scorer: fakeScorer(() => 1),
    trainingSet,
  })

  // Every engine-executed step shows up, in execution order — the plain
  // createStep included, even though only declareStep steps are tunable.
  expect(seen[0]?.steps.map((step) => step.name)).toEqual(["solve", "plain"])
  expect(seen[0]).toMatchObject({
    steps: [
      { status: "success", stepId: "solve" },
      {
        output: { z: 3 },
        status: "success",
        stepId: "plain",
        stepType: "workflow_step",
      },
    ],
  })
  // Harvesting is unchanged: the plain step's output never becomes a demo.
  expect(solve.examples).toHaveLength(2)
  for (const example of solve.examples) {
    expect(example.outputData.z).toBeUndefined()
  }
})

test("gate.threshold switches acceptance from truthiness to >= threshold", async () => {
  const zeroGate = () => gateScorer(() => 0)

  const strict = makeWorkflow()
  await bootstrapFewShot(strict.workflow, {
    gate: { scorer: zeroGate() },
    savePrompts: () => Promise.resolve(),
    scorer: fakeScorer(() => 1),
    trainingSet,
  })
  // Score 0 with no threshold: nothing above zero, nothing accepted.
  expect(augmentedCount(strict.step.examples)).toBe(0)

  const lenient = makeWorkflow()
  await bootstrapFewShot(lenient.workflow, {
    gate: { scorer: zeroGate(), threshold: 0 },
    savePrompts: () => Promise.resolve(),
    scorer: fakeScorer(() => 1),
    trainingSet,
  })
  // An explicit threshold of 0 accepts a 0 score (>=), unlike DSPy's
  // truthiness quirk where a falsy threshold is ignored.
  expect(augmentedCount(lenient.step.examples)).toBe(2)
})

test("a throwing gate counts toward maxErrors and rethrows at the cap", async () => {
  const { workflow } = makeWorkflow()
  let gateRuns = 0
  const gate = gateScorer(() => {
    gateRuns += 1
    throw new Error("gate boom")
  })

  await expect(
    bootstrapFewShot(workflow, {
      gate: { scorer: gate },
      maxErrors: 2,
      savePrompts: () => Promise.resolve(),
      scorer: fakeScorer(() => 1),
      trainingSet,
    })
  ).rejects.toThrow("gate boom")
  // First failure is swallowed and counted, the second hits the cap.
  expect(gateRuns).toBe(2)
})

test("combining gate with scoreThreshold is rejected", async () => {
  const { workflow } = makeWorkflow()
  await expect(
    bootstrapFewShot(workflow, {
      gate: { scorer: gateScorer(() => 1) },
      savePrompts: () => Promise.resolve(),
      scoreThreshold: 0.5,
      scorer: fakeScorer(() => 1),
      trainingSet,
    })
  ).rejects.toThrow("gate.threshold")
})
