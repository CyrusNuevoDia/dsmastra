import { createScorer } from "@mastra/core/evals"
import type { RequestContext } from "@mastra/core/request-context"
import { createStep } from "@mastra/core/workflows"
import { z } from "zod"

import type { Example, RunContext } from "@/step"
import { RUN_CONTEXT_KEY } from "@/step"

export type Call = { ctx?: RunContext; inputData: Record<string, unknown> }

type Gold = {
  inputData: Record<string, unknown>
  outputData: Record<string, unknown>
}

/**
 * A Mastra scorer wrapping a plain (gold, prediction) => score function —
 * the test-side stand-in for a user's createScorer definition.
 */
export const fakeScorer = (
  score: (
    gold: Gold,
    prediction: Record<string, unknown> | undefined
  ) => number,
  id = "test-scorer"
) =>
  createScorer({ description: "deterministic test scorer", id }).generateScore(
    ({ run }) =>
      score(
        {
          inputData: run.input as Gold["inputData"],
          outputData: run.groundTruth as Gold["outputData"],
        },
        run.output as Record<string, unknown> | undefined
      )
  )

const fieldsSchema = z.record(z.string(), z.unknown())

/**
 * Deterministic fake step: `fn` maps inputData (+ctx) to outputData, calls are
 * logged, clones share the log and behavior but copy examples/description.
 * Built on a real Mastra `createStep` and carrying the tunable prompt state
 * alongside, exactly like `declareStep`.
 */
export const fakeStep = (
  id: string,
  fn: (
    inputData: Record<string, unknown>,
    ctx?: RunContext
  ) => Record<string, unknown>,
  log: Call[] = []
) => {
  const execute = (
    params: {
      inputData: Record<string, unknown>
      requestContext?: RequestContext
    },
    directCtx?: RunContext
  ) => {
    const { inputData } = params
    // Engine invocations carry the RunContext in the request context, direct
    // calls pass it as the second argument — same contract as declareStep.
    const ctx =
      directCtx ??
      (params.requestContext?.get(RUN_CONTEXT_KEY) as RunContext | undefined)
    log.push({ ctx, inputData })
    const outputData = fn(inputData, ctx)
    ctx?.trace?.push({ inputData, outputData, stepId: id })
    return Promise.resolve(outputData)
  }
  const step = createStep({
    execute,
    id,
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  })
  const tunable = Object.assign(step, {
    clone: () => {
      const cloned = fakeStep(id, fn, log)
      cloned.description = tunable.description
      cloned.examples = structuredClone(tunable.examples)
      return cloned
    },
    description: "solve",
    examples: [] as Example[],
    execute,
    inputSchema: fieldsSchema,
    model: "stub" as never,
    outputSchema: fieldsSchema,
    settings: {},
  })
  return tunable
}
