import type { InferPublicSchema } from "@mastra/core/schema"
import { createStep } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { generateText, Output } from "ai"
import type { z } from "zod"

import type { FieldSchema, Fields } from "@/fields"

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

/** Per-run overrides and trace capture, threaded through a whole program run. */
export type RunContext = {
  model?: LanguageModel
  seed?: number
  temperature?: number
  trace?: TraceStep[]
}

/** AI SDK call settings supported first-class on a step, forwarded verbatim to generateObject. */
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
  settings: StepSettings
}

export type AnyTunableStep = TunableStep

const renderPrompt = (
  description: string,
  examples: Example[],
  inputData: Fields
): string => {
  const parts = [description]
  for (const example of examples) {
    parts.push(
      `Example:\nInput:\n${JSON.stringify(example.inputData)}\nOutput:\n${JSON.stringify(example.outputData)}`
    )
  }
  parts.push(`Input:\n${JSON.stringify(inputData)}`)
  return parts.join("\n\n")
}

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

  const step = createStep({
    description,
    execute: async ({ inputData }) => {
      const outputData = await execute({
        inputData: inputSchema.parse(inputData),
      })
      // SAFETY: `outputData` came back from `execute`, which parses it through
      // `config.outputSchema` — the very schema Mastra derives its expected step
      // output from. Both sides agree on the runtime shape; the assertion only
      // bridges zod's `output<T>` and Mastra's `InferPublicSchema<T>`, two
      // spellings of that same type that TS cannot equate while `T` is generic.
      return outputData as InferPublicSchema<TOutputSchema>
    },
    id,
    inputSchema,
    outputSchema,
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
      }),
    description,
    examples: structuredClone(examples ?? []),
    execute,
    inputSchema,
    model,
    outputSchema,
    settings,
  })

  return tunable
}
