import { expect, test } from "bun:test"

import { createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { loadPrompts } from "@/optimizers/utils"
import type { Prompts } from "@/optimizers/utils"

import { fakeStep } from "./helpers"

const makeWorkflow = () => {
  const step = fakeStep("solve", (inputData) => inputData)
  const workflow = createWorkflow({
    id: "wf",
    inputSchema: z.record(z.string(), z.unknown()),
    outputSchema: z.record(z.string(), z.unknown()),
  })
    .then(step)
    .commit()
  return { step, workflow }
}

const solvePrompts = {
  description: "tuned description",
  examples: [{ inputData: { x: 1 }, outputData: { y: 2 } }],
}

const prompts: Prompts = {
  steps: { solve: solvePrompts },
  version: 1,
}

test("loadPrompts applies saved state onto the workflow's live steps", () => {
  const { step, workflow } = makeWorkflow()
  const restored = loadPrompts(workflow, prompts)
  // In-place: the same workflow comes back and the caller's own step
  // reference carries the tuned state.
  expect(restored).toBe(workflow)
  expect(step.description).toBe("tuned description")
  expect(step.examples).toEqual(solvePrompts.examples)
})

test("prompts survive a JSON round-trip", () => {
  const { step, workflow } = makeWorkflow()
  // oxlint-disable-next-line unicorn/prefer-structured-clone -- the JSON round-trip IS what this test verifies
  const revived = JSON.parse(JSON.stringify(prompts)) as Prompts
  expect(revived).toEqual(prompts)
  loadPrompts(workflow, revived)
  expect(step.description).toBe("tuned description")
})

test("loadPrompts throws on step-id mismatch in either direction", () => {
  const { workflow } = makeWorkflow()
  expect(() => loadPrompts(workflow, { steps: {}, version: 1 })).toThrow(
    "missing: solve"
  )
  expect(() =>
    loadPrompts(workflow, {
      steps: { ...prompts.steps, stray: solvePrompts },
      version: 1,
    })
  ).toThrow("unknown: stray")
})

test("loadPrompts reaches tunable steps inside a parallel section", () => {
  const left = fakeStep("left", (inputData) => inputData)
  const right = fakeStep("right", (inputData) => inputData)
  const parallelWorkflow = createWorkflow({
    id: "wf-parallel",
    inputSchema: z.record(z.string(), z.unknown()),
    outputSchema: z.record(z.string(), z.unknown()),
  })
    .parallel([left, right])
    .commit()
  loadPrompts(parallelWorkflow, {
    steps: {
      left: { description: "tuned left", examples: [] },
      right: { description: "tuned right", examples: [] },
    },
    version: 1,
  })
  expect(left.description).toBe("tuned left")
  expect(right.description).toBe("tuned right")
})
