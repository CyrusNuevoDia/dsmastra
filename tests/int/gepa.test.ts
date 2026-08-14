import { expect, test } from "bun:test"

import { createScorer } from "@mastra/core/evals"
import { createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { declareStep, gepa } from "@/index"
import type { Example } from "@/index"

import { model } from "../_helpers"

const trainingSet: Example[] = [1, 2, 3, 5, 8, 13].map((x) => ({
  inputData: { x },
  outputData: { y: x * 2 },
}))

const BAD_DESCRIPTION = "Return y = x."

test(
  "GEPA improves arithmetic accuracy",
  async () => {
    const step = declareStep({
      // BAD_DESCRIPTION asks for the wrong operation; the target is x*2.
      description: BAD_DESCRIPTION,
      id: "math",
      inputSchema: z.object({ x: z.number() }),
      model,
      outputSchema: z.object({ y: z.number() }),
    })
    const workflow = createWorkflow({
      id: "wf-math",
      inputSchema: z.object({ x: z.number() }),
      outputSchema: z.object({ y: z.number() }),
    })
      .then(step)
      .commit()

    const doubling = createScorer({
      description: "Exact match on y, with corrective feedback.",
      id: "doubling",
    })
      .generateScore(({ run }) => {
        const gold = run.groundTruth as { y: number }
        const prediction = run.output as { y?: number } | undefined
        return prediction?.y === gold.y ? 1 : 0
      })
      .generateReason(({ run }) => {
        const input = run.input as { x: number }
        const gold = run.groundTruth as { y: number }
        const prediction = run.output as { y?: number } | undefined
        return prediction?.y === gold.y
          ? `Correct: for x=${input.x}, y=${gold.y}.`
          : `Wrong: for x=${input.x} the answer should be y=${gold.y}, but the assistant returned y=${prediction?.y}.`
      })

    const { score } = await gepa(workflow, {
      // Few-shot pre-pass first, then description evolution — one call.
      maxFewShotExamples: 2,
      maxScorerCalls: 60,
      reflectionModel: model,
      savePrompts: () => Promise.resolve(),
      scorer: doubling,
      seed: 0,
      trainingSet,
    })

    // Tuning is in place: the caller's own step reference carries the result.
    const tunedStep = step
    console.log(`\n=== BEFORE ===\n${BAD_DESCRIPTION}`)
    console.log(`\n=== AFTER (score ${score}) ===\n${tunedStep.description}`)
    console.log("\n=== FEW-SHOT EXAMPLES (max 2) ===")
    console.log(tunedStep.examples)
    expect(tunedStep.examples.length <= 2).toBe(true)

    // The pre-pass alone may already perfect the program (its examples teach
    // the doubling pattern), so the seed can legitimately score 1 and GEPA
    // accepts no child. Only the end state is asserted.
    expect(score).toBe(1)

    // The tuned workflow actually behaves correctly.
    const outputData = await Promise.all(
      trainingSet.map(
        (example) =>
          tunedStep.execute({
            inputData: example.inputData as never,
          }) as Promise<{ y: number }>
      )
    )
    for (const [i, example] of trainingSet.entries()) {
      expect(outputData[i]?.y).toBe(example.outputData.y as number)
    }
  },
  { timeout: 300_000 }
)
