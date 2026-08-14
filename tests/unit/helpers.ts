import { z } from "zod"

import type { RunContext, AnyTunableStep } from "@/step"

export type Call = { ctx?: RunContext; inputData: Record<string, unknown> }

/**
 * Deterministic fake step: `fn` maps inputData (+ctx) to outputData, calls are
 * logged, clones share the log and behavior but copy examples/description.
 */
export const fakeStep = (
  id: string,
  fn: (
    inputData: Record<string, unknown>,
    ctx?: RunContext
  ) => Record<string, unknown>,
  log: Call[] = []
): AnyTunableStep => {
  const step: AnyTunableStep = {
    clone: () => {
      const cloned = fakeStep(id, fn, log)
      cloned.description = step.description
      cloned.examples = structuredClone(step.examples)
      return cloned
    },
    description: "solve",
    examples: [],
    execute: ({ inputData }, ctx) => {
      log.push({ ctx, inputData })
      const outputData = fn(inputData, ctx)
      ctx?.trace?.push({ inputData, outputData, stepId: id })
      return Promise.resolve(outputData)
    },
    id,
    inputSchema: z.record(z.string(), z.unknown()),
    model: "stub" as never,
    outputSchema: z.record(z.string(), z.unknown()),
    settings: {},
  }
  return step
}
