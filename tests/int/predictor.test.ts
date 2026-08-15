import { expect, test } from "bun:test"

import { z } from "zod"

import { declareStep } from "#src/step"
import { model } from "#tests/_helpers"

test("declareStep → doubles number (live)", async () => {
  const step = declareStep({
    description: "Return y = x * 2.",
    id: "double",
    inputSchema: z.object({ x: z.number().int().min(0).max(5) }),
    model,
    outputSchema: z.object({ y: z.number() }),
  })

  const { y } = await step.execute({ inputData: { x: 3 } })
  expect(y).toBe(6)
})
