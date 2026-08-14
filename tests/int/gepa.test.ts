import { expect, test } from "bun:test"

import { z } from "zod"

import { bootstrapFewShot } from "@/bootstrap"
import { gepa } from "@/gepa"
import { declarePredictor } from "@/predictor"
import { createProgram } from "@/program"
import type { Example } from "@/simba"

import { model } from "../_helpers"

const trainset: Example[] = [1, 2, 3, 5, 8, 13].map((x) => ({
  inputs: { x },
  outputs: { y: x * 2 },
}))

const BAD_INSTRUCTIONS = "Return y = x."

test(
  "GEPA improves arithmetic accuracy",
  async () => {
    const predictor = declarePredictor({
      inputSchema: z.object({ x: z.number() }),
      // BAD_INSTRUCTIONS asks for the wrong operation; the target is x*2.
      instructions: BAD_INSTRUCTIONS,
      model,
      name: "math",
      outputSchema: z.object({ y: z.number() }),
    })
    const program = createProgram({
      forward: (call, input: Record<string, unknown>) => call("math", input),
      predictors: [predictor],
    })

    // DSPy-style composition: BootstrapFewShot installs demos first, then
    // GEPA evolves instructions on the demo-carrying program.
    const student = await bootstrapFewShot(program, trainset, {
      maxBootstrappedDemos: 2,
      maxLabeledDemos: 2,
      metric: (gold, prediction) => ({
        score: prediction?.y === gold.outputs.y ? 1 : 0,
      }),
    })

    const result = await gepa(student, trainset, {
      maxMetricCalls: 60,
      metric: (gold, prediction) => ({
        feedback:
          prediction?.y === gold.outputs.y
            ? `Correct: for x=${gold.inputs.x}, y=${gold.outputs.y}.`
            : `Wrong: for x=${gold.inputs.x} the answer should be y=${gold.outputs.y}, but the assistant returned y=${prediction?.y}.`,
        score: prediction?.y === gold.outputs.y ? 1 : 0,
      }),
      reflectionLM: model,
      seed: 0,
    })

    const before = result.candidates[0]?.math
    const after = result.candidates[result.bestIdx]?.math
    console.log(`\n=== BEFORE (score ${result.valAggregateScores[0]}) ===`)
    console.log(before)
    console.log(
      `\n=== AFTER (score ${result.valAggregateScores[result.bestIdx]}) ===`
    )
    console.log(after)
    console.log("\n=== BOOTSTRAPPED DEMOS (max 2) ===")
    console.log(result.program.predictors[0]?.demos)
    expect((result.program.predictors[0]?.demos.length ?? 0) <= 2).toBe(true)

    // The bootstrap pre-pass alone may already perfect the program (its demos
    // teach the doubling pattern), so the seed can legitimately score 1 and
    // GEPA accepts no child. Only the end state is asserted.
    expect(result.valAggregateScores[result.bestIdx]).toBe(1)

    // The tuned program actually behaves correctly.
    const tuned = result.program
    const outputs = await Promise.all(
      trainset.map(
        (example) =>
          tuned.run(example.inputs as never) as Promise<{ y: number }>
      )
    )
    for (const [i, example] of trainset.entries()) {
      expect(outputs[i]?.y).toBe(example.outputs.y as number)
    }
  },
  { timeout: 300_000 }
)
