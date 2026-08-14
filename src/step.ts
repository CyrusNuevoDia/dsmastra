import type { RequestContext } from "@mastra/core/request-context"
import type { InferPublicSchema } from "@mastra/core/schema"
import { createStep } from "@mastra/core/workflows"
import type { Step } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { generateText, Output } from "ai"
import type { z } from "zod"

import type { FieldSchema, Fields } from "@/fields"
import { renderPrompt } from "@/prompting"

/**
 * A training example: inputData plus the expected output fields. Doubles as a
 * few-shot example rendered into the prompt between the description and the
 * live input — there is no separate demo type.
 */
export type Example<TInput = Fields, TOutput = Fields> = {
  inputData: TInput
  outputData: TOutput
}

export type TraceStep = {
  inputData: Fields
  outputData: Fields
  stepId: string
}

/**
 * The request-context key rollouts use to hand a RunContext to steps executed
 * by Mastra's engine.
 */
export const RUN_CONTEXT_KEY = "dsmastra"

/** A rollout's Mastra trace linkage, for attaching scores to it in Studio. */
export type ScoreTarget = {
  spanId?: string
  traceId?: string
}

/** Per-run overrides and trace capture, threaded through a whole program run. */
export type RunContext = {
  model?: LanguageModel
  seed?: number
  /** Written by the engine runner after a successful run — see workflowToProgram. */
  target?: ScoreTarget
  temperature?: number
  trace?: TraceStep[]
}

/** AI SDK call settings supported first-class on a step, forwarded verbatim to generateText. */
export type StepSettings = {
  abortSignal?: AbortSignal
  frequencyPenalty?: number
  headers?: Record<string, string | undefined>
  maxOutputTokens?: number
  maxRetries?: number
  presencePenalty?: number
  seed?: number
  stopSequences?: string[]
  temperature?: number
  topK?: number
  topP?: number
}

export type StepConfig<
  TStepId extends string,
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
> = StepSettings & {
  /** The step's instruction text — this is what the optimizers tune. */
  description: string
  examples?: Example[]
  id: TStepId
  inputSchema: TInputSchema
  model: LanguageModel
  outputSchema: TOutputSchema
  /** Mastra live-eval scorers, forwarded to the step verbatim. Optimizer
   * rollouts disable these (they run with `disableScorers`) and score through
   * the optimizer's own scorer instead. */
  scorers?: Step["scorers"]
}

/**
 * A Mastra step that is also the unit the optimizers tune: `description` and
 * `examples` are mutable prompt state, everything else is fixed config.
 *
 * `execute` is declared with method syntax on purpose. Mastra's own execute
 * takes a much larger params object (runId, mastra, request context…) that it
 * supplies when running the step inside a workflow, while callers here only
 * ever pass `inputData` — and the optimizers additionally thread a RunContext
 * for per-rollout model/seed/temperature overrides and trace capture, which
 * Mastra simply never passes. Method syntax makes the parameter bivariant, so
 * the real Mastra step satisfies this narrower view without an assertion.
 */
export type TunableStep<
  TStepId extends string = string,
  TInputSchema extends FieldSchema = FieldSchema,
  TOutputSchema extends FieldSchema = FieldSchema,
> = {
  clone: () => TunableStep<TStepId, TInputSchema, TOutputSchema>
  description: string
  examples: Example[]
  // oxlint-disable-next-line typescript/method-signature-style -- bivariance is the point; see the doc comment above
  execute(
    params: { inputData: z.infer<TInputSchema> },
    ctx?: RunContext
  ): Promise<z.infer<TOutputSchema>>
  id: TStepId
  inputSchema: TInputSchema
  model: LanguageModel
  outputSchema: TOutputSchema
  scorers?: Step["scorers"]
  settings: StepSettings
}

export type AnyTunableStep = TunableStep

export const declareStep = <
  TStepId extends string,
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
>(
  config: StepConfig<TStepId, TInputSchema, TOutputSchema>
): TunableStep<TStepId, TInputSchema, TOutputSchema> => {
  const {
    description,
    examples,
    id,
    inputSchema,
    model,
    outputSchema,
    scorers,
    ...settings
  } = config

  // Assigned at the bottom of this function; `execute` only dereferences it at
  // call time, after construction, so tuned description/examples are read live.
  // oxlint-disable-next-line prefer-const -- assigned once at the bottom, after `execute` (which closes over it) is defined
  let tunable: TunableStep<TStepId, TInputSchema, TOutputSchema>

  const execute = async (
    { inputData }: { inputData: z.infer<TInputSchema> },
    ctx?: RunContext
  ): Promise<z.infer<TOutputSchema>> => {
    const generated = await generateText({
      ...tunable.settings,
      model: ctx?.model ?? tunable.model,
      output: Output.object({ schema: tunable.outputSchema }),
      prompt: renderPrompt(tunable.description, tunable.examples, inputData),
      seed: ctx?.seed ?? tunable.settings.seed,
      temperature: ctx?.temperature ?? tunable.settings.temperature,
    })
    // The AI SDK types `output` through a conditional it can't collapse while
    // the schema is still generic, so re-parse instead of asserting: this both
    // recovers the precise output type and re-checks what the model returned.
    const outputData = tunable.outputSchema.parse(generated.output)
    ctx?.trace?.push({ inputData, outputData, stepId: tunable.id })
    return outputData
  }

  // One execute serves both callers. Mastra's engine passes its own params
  // object, recognizable by the requestContext it always carries — there the
  // input is re-parsed and the RunContext (rollout overrides + trace capture)
  // is read from the reserved request-context key. The optimizers call with
  // bare inputData and an explicit RunContext.
  const runStep = (
    params: {
      inputData: z.infer<TInputSchema>
      requestContext?: RequestContext
    },
    ctx?: RunContext
  ): Promise<z.infer<TOutputSchema>> => {
    if (params.requestContext) {
      // SAFETY: only dsmastra's own rollout executor writes this key, and it
      // always writes a RunContext (or nothing at all).
      const engineCtx = params.requestContext.get(RUN_CONTEXT_KEY) as
        | RunContext
        | undefined
      return execute(
        { inputData: inputSchema.parse(params.inputData) },
        engineCtx
      )
    }
    return execute(params, ctx)
  }

  const step = createStep({
    description,
    // SAFETY: `runStep` resolves through `execute`, which parses its result
    // with `config.outputSchema` — the very schema Mastra derives its expected
    // step output from. Both sides agree on the runtime shape; the assertion
    // only bridges zod's `output<T>` and Mastra's `InferPublicSchema<T>`, two
    // spellings of that same type that TS cannot equate while `T` is generic.
    execute: async ({ inputData, requestContext }) =>
      // SAFETY: the engine's inputData is typed via InferPublicSchema, zod's
      // own output type under another name; runStep re-parses it with the
      // schema before use, so the assertion never outruns validation.
      (await runStep({
        inputData: inputData as z.infer<TInputSchema>,
        requestContext,
      })) as InferPublicSchema<TOutputSchema>,
    id,
    inputSchema,
    outputSchema,
    scorers,
  })

  // Mastra's step stores the schemas re-wrapped as standard-schema objects and
  // types `execute` for in-workflow invocation. Overwriting them restores the
  // library's own view of the step without disturbing what Mastra runs, and
  // the tuning fields (description, examples, settings, clone) ride alongside.
  tunable = Object.assign(step, {
    clone: () =>
      declareStep({
        ...tunable.settings,
        description: tunable.description,
        examples: tunable.examples,
        id,
        inputSchema,
        model: tunable.model,
        outputSchema,
        scorers: tunable.scorers,
      }),
    description,
    examples: structuredClone(examples ?? []),
    execute: runStep,
    inputSchema,
    model,
    outputSchema,
    scorers,
    settings,
  })

  return tunable
}
