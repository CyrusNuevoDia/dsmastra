import { expect, test } from "bun:test"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import { createWorkflow, declareStep, optimize, SIMBA } from "@/index"
import type { Metric } from "@/simba"

// A temperature-capable (non-reasoning) model: SIMBA's rollout sampling and
// the rule strategy both need it.
const model = openai("gpt-4.1-mini")

// The convention the instructions fail to mention: answers must be spelled
// out as words. The "Spell out:" examples succeed anyway, the plain ones land
// on partial credit, so minibatches carry the score spread SIMBA climbs.
const trainset = [
  { expected: { answer: "four" }, input: { question: "Spell out: 2+2" } },
  { expected: { answer: "six" }, input: { question: "Spell out: 3+3" } },
  { expected: { answer: "five" }, input: { question: "2+3" } },
  { expected: { answer: "seven" }, input: { question: "10-3" } },
  { expected: { answer: "two" }, input: { question: "1+1" } },
  { expected: { answer: "nine" }, input: { question: "4+5" } },
] as const

const numbers: Record<string, number> = {
  five: 5,
  four: 4,
  nine: 9,
  seven: 7,
  six: 6,
  two: 2,
}

// Graded: 1.0 for the spelled-out answer, 0.5 for the right number in the
// wrong format, 0 otherwise.
const metric: Metric = (example, prediction) => {
  const answer = String(prediction?.answer ?? "")
    .trim()
    .toLowerCase()
  const expected = example.expected.answer as string
  if (answer === expected) {
    return 1
  }
  if (Number.parseFloat(answer) === numbers[expected]) {
    return { feedback: "Numerically right, but not spelled out.", score: 0.5 }
  }
  return 0
}

test(
  "SIMBA teaches the answer-format convention",
  async () => {
    const step = declareStep({
      id: "math",
      inputSchema: z.object({ question: z.string() }),
      instructions: "Answer the arithmetic question.", // underspecified: format unstated
      model,
      outputSchema: z.object({ answer: z.string() }),
      temperature: 0,
    })

    const wf = createWorkflow({
      id: "wf-math",
      inputSchema: z.object({ question: z.string() }),
      outputSchema: z.object({ answer: z.string() }),
    })
      .then(step)
      .commit()

    const score = async (workflow: typeof wf) => {
      const scores = await Promise.all(
        trainset.map(async (ex) => {
          const prediction = await workflow.steps.math.execute({
            inputData: ex.input,
          })
          const result = await metric(ex, prediction)
          return typeof result === "number" ? result : result.score
        })
      )
      return scores.reduce((acc, s) => acc + s, 0) / trainset.length
    }

    const before = await score(wf)
    expect(before).toBeLessThan(1)

    const tuned = await optimize(
      SIMBA({
        bsize: trainset.length,
        maxDemos: 2,
        maxSteps: 3,
        metric,
        numCandidates: 3,
        seed: 0,
      }),
      wf,
      { trainset }
    )

    const after = await score(tuned)
    console.log(`SIMBA int test: before=${before}, after=${after}`)
    expect(after).toBeGreaterThan(before)
  },
  { timeout: 300_000 }
)
