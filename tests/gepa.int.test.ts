import { expect, test } from "bun:test"
import { z } from "zod"
import { gepa } from "@/gepa"
import { declarePredictor } from "@/predictor"
import { createProgram } from "@/program"
import type { Example } from "@/simba"
import { model } from "./_helpers"

const trainset: Example[] = [1, 2, 3, 5, 8, 13].map((x) => ({
  expected: { y: x * 2 },
  input: { x },
}))

const BAD_INSTRUCTIONS = "Return y = x."

test(
  "GEPA improves arithmetic accuracy",
  async () => {
    const predictor = declarePredictor({
      inputSchema: z.object({ x: z.number() }),
      instructions: BAD_INSTRUCTIONS, // should be x*2
      model,
      name: "math",
      outputSchema: z.object({ y: z.number() }),
    })
    const program = createProgram({
      forward: (call, input: Record<string, unknown>) => call("math", input),
      predictors: [predictor],
    })

    const result = await gepa(program, trainset, {
      maxMetricCalls: 60,
      metric: (gold, prediction) => ({
        feedback:
          prediction?.y === gold.expected.y
            ? `Correct: for x=${gold.input.x}, y=${gold.expected.y}.`
            : `Wrong: for x=${gold.input.x} the answer should be y=${gold.expected.y}, but the assistant returned y=${prediction?.y}.`,
        score: prediction?.y === gold.expected.y ? 1 : 0,
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

    expect(result.valAggregateScores[0]).toBeLessThan(1)
    expect(result.bestIdx).toBeGreaterThan(0)
    expect(result.valAggregateScores[result.bestIdx]).toBe(1)

    // The tuned program actually behaves correctly.
    const tuned = result.program
    const outputs = await Promise.all(
      trainset.map(
        (example) => tuned.run(example.input as never) as Promise<{ y: number }>
      )
    )
    for (const [i, example] of trainset.entries()) {
      expect(outputs[i]?.y).toBe(example.expected.y as number)
    }
  },
  { timeout: 300_000 }
)
