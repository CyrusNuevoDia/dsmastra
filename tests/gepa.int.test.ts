import { expect, test } from "bun:test"
import { z } from "zod"
import { createWorkflow, declareStep, GEPA, optimize } from "@/index"
import { model } from "./_helpers"

const trainset = [
  { expected: { y: 2 }, input: { x: 1 } },
  { expected: { y: 4 }, input: { x: 2 } },
] as const

test("GEPA improves arithmetic accuracy", async () => {
  // Deliberately wrong prompt.
  const badStep = declareStep({
    id: "bad-math",
    inputSchema: z.object({ x: z.number() }),
    instructions: "Return y = x.", // should be x*2
    model,
    outputSchema: z.object({ y: z.number() }),
  })

  const wf = createWorkflow({
    id: "wf-bad",
    inputSchema: z.object({ x: z.number() }),
    outputSchema: z.object({ y: z.number() }),
  })
    .then(badStep)
    .commit()

  // Helper to measure accuracy
  const accuracy = async (workflow: typeof wf) => {
    const hits = await Promise.all(
      trainset.map(async (ex) => {
        const { y } = await workflow.steps["bad-math"].execute({
          inputData: ex.input,
        })
        return y === ex.expected.y
      })
    )
    return hits.filter(Boolean).length / trainset.length
  }

  const baseAcc = await accuracy(wf)
  expect(baseAcc).toBeLessThan(1)

  const tuned = await optimize(GEPA({ maxDemos: 2, maxSteps: 4 }), wf, {
    trainset,
  })

  const tunedAcc = await accuracy(tuned)
  expect(tunedAcc).toBe(1) // must be perfect on tiny set
})
