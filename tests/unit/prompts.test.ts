import { expect, test } from "bun:test"

import { z } from "zod"

import { loadPrompts } from "@/optimizers/utils"
import type { Prompts } from "@/optimizers/utils"
import { createWorkflow } from "@/workflow"

import { fakeStep } from "./helpers"

const makeWorkflow = () =>
  createWorkflow({
    id: "wf",
    inputSchema: z.object({}),
    outputSchema: z.object({}),
  })
    .then(fakeStep("solve", (inputData) => inputData))
    .commit()

const solvePrompts = {
  description: "tuned description",
  examples: [{ inputData: { x: 1 }, outputData: { y: 2 } }],
}

const prompts: Prompts = {
  steps: { solve: solvePrompts },
  version: 1,
}

test("loadPrompts applies saved state to a new workflow without mutating the source", () => {
  const workflow = makeWorkflow()
  const restored = loadPrompts(workflow, prompts)
  expect(restored).not.toBe(workflow)
  expect(restored.steps.solve?.description).toBe("tuned description")
  expect(restored.steps.solve?.examples).toEqual(solvePrompts.examples)
  // The source workflow's step is untouched.
  expect(workflow.steps.solve?.description).toBe("solve")
  expect(workflow.steps.solve?.examples).toEqual([])
})

test("prompts survive a JSON round-trip", () => {
  const workflow = makeWorkflow()
  // oxlint-disable-next-line unicorn/prefer-structured-clone -- the JSON round-trip IS what this test verifies
  const revived = JSON.parse(JSON.stringify(prompts)) as Prompts
  expect(revived).toEqual(prompts)
  const restored = loadPrompts(workflow, revived)
  expect(restored.steps.solve?.description).toBe("tuned description")
})

test("loadPrompts throws on step-id mismatch in either direction", () => {
  const workflow = makeWorkflow()
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
