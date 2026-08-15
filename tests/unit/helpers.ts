import { createScorer } from "@mastra/core/evals"
import type { RequestContext } from "@mastra/core/request-context"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import type { AnyWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import type { Prompts } from "../../src/optimizers/utils"
import type { Example, RunContext } from "../../src/step"
import { RUN_CONTEXT_KEY } from "../../src/step"

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

/** Read a named prompt entry while keeping fixture assertions non-optional. */
export const promptStep = (prompts: Prompts, id: string) => {
  const step = prompts.steps[id]
  if (!step) {
    throw new Error(`Prompts lost step ${id}`)
  }
  return step
}

/**
 * Deterministic fake step: `fn` maps inputData (+ctx) to outputData, calls are
 * logged, and built on a real Mastra `createStep` with declarative prompt state
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
  const declarative = Object.assign(step, {
    description: "solve",
    examples: [] as Example[],
    execute,
    inputSchema: fieldsSchema,
    model: "stub" as never,
    outputSchema: fieldsSchema,
    settings: {},
  })
  return declarative
}

/** A sequential workflow over fake declarative steps for engine-path tests. */
export const fakeWorkflow = (
  first: ReturnType<typeof fakeStep>,
  second?: ReturnType<typeof fakeStep>
) => {
  /* oxlint-disable promise/prefer-await-to-then -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  const workflow = createWorkflow({
    id: `${first.id}-workflow`,
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  }).then(first)
  const built = second ? workflow.then(second) : workflow
  /* oxlint-enable promise/prefer-await-to-then */
  return built.commit()
}

/** Run a factory-built optimizer workflow to completion, rethrowing failures. */
export const runOptimizer = async (
  optimizer: AnyWorkflow
): Promise<{ candidates: [Prompts, { score: number }][]; score: number }> => {
  const run = await optimizer.createRun()
  const result = await run.start({ inputData: {} })
  if (result.status === "failed") {
    throw result.error
  }
  if (result.status !== "success") {
    throw new Error(`Optimizer run ended with status ${result.status}`)
  }
  return result.result as {
    candidates: [Prompts, { score: number }][]
    score: number
  }
}
