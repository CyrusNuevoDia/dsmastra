import { expect, test } from "bun:test"

import { openai } from "@ai-sdk/openai"
import { createScorer } from "@mastra/core/evals"
import { createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { declareStep, simba } from "@/index"

// A temperature-capable (non-reasoning) model: SIMBA's rollout sampling and
// the rule strategy both need it.
const model = openai("gpt-4.1-mini")

// The convention the description fails to mention: answers must be spelled
// out as words. The "Spell out:" examples succeed anyway, the plain ones land
// on partial credit, so minibatches carry the score spread SIMBA climbs.
const trainingSet = [
  { inputData: { question: "Spell out: 2+2" }, outputData: { answer: "four" } },
  { inputData: { question: "Spell out: 3+3" }, outputData: { answer: "six" } },
  { inputData: { question: "2+3" }, outputData: { answer: "five" } },
  { inputData: { question: "10-3" }, outputData: { answer: "seven" } },
  { inputData: { question: "1+1" }, outputData: { answer: "two" } },
  { inputData: { question: "4+5" }, outputData: { answer: "nine" } },
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
const grade = (
  expected: string,
  prediction: Record<string, unknown> | undefined
): { reason?: string; score: number } => {
  const answer = String(prediction?.answer ?? "")
    .trim()
    .toLowerCase()
  if (answer === expected) {
    return { score: 1 }
  }
  if (Number(answer) === numbers[expected]) {
    return { reason: "Numerically right, but not spelled out.", score: 0.5 }
  }
  return { score: 0 }
}

const spelledOutScorer = createScorer({
  description: "Graded exact match on the spelled-out answer.",
  id: "spelled-out",
})
  .generateScore(
    ({ run }) =>
      grade(
        (run.groundTruth as { answer: string }).answer,
        run.output as Record<string, unknown>
      ).score
  )
  .generateReason(
    ({ run, score }) =>
      grade(
        (run.groundTruth as { answer: string }).answer,
        run.output as Record<string, unknown>
      ).reason ?? `This trajectory got a score of ${score}.`
  )

test(
  "SIMBA teaches the answer-format convention",
  async () => {
    const step = declareStep({
      // Deliberately underspecified: the output format is left unstated.
      description: "Answer the arithmetic question.",
      id: "math",
      inputSchema: z.object({ question: z.string() }),
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

    // The optimizer tunes the caller's own step in place, so scoring reads
    // the live step before and after.
    const score = async () => {
      const scores = await Promise.all(
        trainingSet.map(async (ex) => {
          const prediction = await step.execute({
            inputData: ex.inputData,
          })
          return grade(ex.outputData.answer, prediction).score
        })
      )
      return scores.reduce((acc: number, s) => acc + s, 0) / trainingSet.length
    }

    const before = await score()
    expect(before).toBeLessThan(1)

    await simba(wf, {
      batchSize: trainingSet.length,
      candidates: 3,
      maxFewShotExamples: 2,
      maxSteps: 3,
      savePrompts: () => Promise.resolve(),
      scorer: spelledOutScorer,
      seed: 0,
      trainingSet,
    })

    const after = await score()
    console.log(`SIMBA int test: before=${before}, after=${after}`)
    expect(after).toBeGreaterThan(before)
  },
  { timeout: 300_000 }
)
