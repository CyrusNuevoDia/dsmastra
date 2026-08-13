import { expect, test } from "bun:test"
import { z } from "zod"
import { createWorkflow, declareStep, optimize, SIMBA } from "@/index"
import { model } from "./_helpers"

const trainset = [
  { expected: { sent: "positive" }, input: { text: "I love it" } },
  { expected: { sent: "negative" }, input: { text: "Hate this" } },
] as const

test("SIMBA improves sentiment prompt", async () => {
  const badStep = declareStep({
    id: "sentiment",
    inputSchema: z.object({ text: z.string() }),
    instructions:
      'Classify statement as "positive" or "negative", but return the inverse.', // wrong
    model,
    outputSchema: z.object({ sent: z.enum(["positive", "negative"]) }),
  })

  const wf = createWorkflow({
    id: "wf-sentiment",
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ sent: z.enum(["positive", "negative"]) }),
  })
    .then(badStep)
    .commit()

  const score = async (workflow: typeof wf) => {
    const hits = await Promise.all(
      trainset.map(async (ex) => {
        const { sent } = await workflow.steps.sentiment.execute({
          inputData: ex.input,
        })
        return sent === ex.expected.sent
      })
    )
    return hits.filter(Boolean).length
  }

  const before = await score(wf)
  expect(before).toBeLessThan(trainset.length)

  const tuned = await optimize(SIMBA({ maxDemos: 2, maxSteps: 6 }), wf, {
    trainset,
  })

  const after = await score(tuned)
  expect(after).toBe(trainset.length)
})
