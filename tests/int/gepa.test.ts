import { expect, test } from "bun:test"

import { z } from "zod"

import { createWorkflow, declareStep, gepa } from "@/index"
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

    const { score, workflow: tuned } = await gepa(workflow, {
      // Few-shot pre-pass first, then description evolution — one call.
      maxFewShotExamples: 2,
      maxMetricCalls: 60,
      metric: (gold, prediction) => ({
        feedback:
          prediction?.y === gold.outputData.y
            ? `Correct: for x=${gold.inputData.x}, y=${gold.outputData.y}.`
            : `Wrong: for x=${gold.inputData.x} the answer should be y=${gold.outputData.y}, but the assistant returned y=${prediction?.y}.`,
        score: prediction?.y === gold.outputData.y ? 1 : 0,
      }),
      reflectionModel: model,
      savePrompts: () => Promise.resolve(),
      seed: 0,
      trainingSet,
    })

    const tunedStep = tuned.steps.math
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
