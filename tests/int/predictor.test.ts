import { expect, test } from "bun:test"

import { z } from "zod"

import { declareStep } from "@/predictor"

import { model } from "../_helpers"

test("declareStep → doubles number (live)", async () => {
  const step = declareStep({
    id: "double",
    inputSchema: z.object({ x: z.number().int().min(0).max(5) }),
    instructions: "Return y = x * 2.",
    model,
    outputSchema: z.object({ y: z.number() }),
  })

  const { y } = await step.execute({ inputData: { x: 3 } })
  expect(y).toBe(6)
})
