import type { InferPublicSchema } from "@mastra/core/schema"
import { createStep } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { generateObject } from "ai"
import type { z } from "zod"

import type { FieldSchema, Fields } from "@/fields"

/** A few-shot example rendered into the prompt between instructions and input. */
export type Demo = {
  augmented?: boolean
  inputs: Fields
  outputs: Fields
}

export type TraceStep = {
  inputs: Fields
  outputs: Fields
  predictorName: string
}

/** Per-run overrides and trace capture, threaded through a whole program run. */
export type RunContext = {
  model?: LanguageModel
  seed?: number
  temperature?: number
  trace?: TraceStep[]
}

export type PredictorConfig<
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
> = {
  demos?: Demo[]
  inputSchema: TInputSchema
  instructions: string
  model: LanguageModel
  name: string
  outputSchema: TOutputSchema
  seed?: number
  temperature?: number
}

export type Predictor<
  TInputSchema extends FieldSchema = FieldSchema,
  TOutputSchema extends FieldSchema = FieldSchema,
> = {
  // oxlint-disable-next-line typescript/method-signature-style -- method syntax keeps params bivariant so any Predictor is assignable to AnyPredictor
  call(
    inputs: z.infer<TInputSchema>,
    ctx?: RunContext
  ): Promise<z.infer<TOutputSchema>>
  clone: () => Predictor<TInputSchema, TOutputSchema>
  demos: Demo[]
  inputSchema: TInputSchema
  instructions: string
  model: LanguageModel
  name: string
  outputSchema: TOutputSchema
  seed?: number
  temperature?: number
}

export type AnyPredictor = Predictor

const renderPrompt = (
  instructions: string,
  demos: Demo[],
  inputs: Fields
): string => {
  const parts = [instructions]
  for (const demo of demos) {
    parts.push(
      `Example:\nInput:\n${JSON.stringify(demo.inputs)}\nOutput:\n${JSON.stringify(demo.outputs)}`
    )
  }
  parts.push(`Input:\n${JSON.stringify(inputs)}`)
  return parts.join("\n\n")
}

export const declarePredictor = <
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
>(
  config: PredictorConfig<TInputSchema, TOutputSchema>
): Predictor<TInputSchema, TOutputSchema> => {
  const predictor: Predictor<TInputSchema, TOutputSchema> = {
    call: async (inputs, ctx) => {
      const generated = await generateObject({
        model: ctx?.model ?? predictor.model,
        prompt: renderPrompt(predictor.instructions, predictor.demos, inputs),
        schema: predictor.outputSchema,
        seed: ctx?.seed ?? predictor.seed,
        temperature: ctx?.temperature ?? predictor.temperature,
      })
      // The AI SDK types `object` through a conditional it can't collapse while
      // the schema is still generic, so re-parse instead of asserting: this both
      // recovers the precise output type and re-checks what the model returned.
      const outputs = predictor.outputSchema.parse(generated.object)
      ctx?.trace?.push({
        inputs,
        outputs,
        predictorName: predictor.name,
      })
      return outputs
    },
    clone: () =>
      // declarePredictor already structuredClones the demos it receives.
      declarePredictor({
        demos: predictor.demos,
        inputSchema: predictor.inputSchema,
        instructions: predictor.instructions,
        model: predictor.model,
        name: predictor.name,
        outputSchema: predictor.outputSchema,
        seed: predictor.seed,
        temperature: predictor.temperature,
      }),
    demos: structuredClone(config.demos ?? []),
    inputSchema: config.inputSchema,
    instructions: config.instructions,
    model: config.model,
    name: config.name,
    outputSchema: config.outputSchema,
    seed: config.seed,
    temperature: config.temperature,
  }
  return predictor
}

type DeclareStepConfig<
  TStepId extends string,
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
> = {
  id: TStepId
  inputSchema: TInputSchema
  instructions: string
  model: LanguageModel
  outputSchema: TOutputSchema
  seed?: number
  temperature?: number
}

/**
 * A Mastra step that also carries the predictor driving it, so the optimizers can
 * find and tune it.
 *
 * `execute` is declared with method syntax on purpose. Mastra's own execute takes
 * a much larger params object (runId, mastra, request context…) that it supplies
 * when running the step inside a workflow, while callers here only ever pass
 * `inputData`. Method syntax makes the parameter bivariant, so the real Mastra
 * step satisfies this narrower view without an assertion — the same reason
 * `Predictor.call` above is written this way.
 */
export type DeclaredStep<
  TStepId extends string,
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
> = {
  // oxlint-disable-next-line typescript/method-signature-style -- bivariance is the point; see the doc comment above
  execute(params: {
    inputData: z.infer<TInputSchema>
  }): Promise<z.infer<TOutputSchema>>
  id: TStepId
  inputSchema: TInputSchema
  outputSchema: TOutputSchema
  predictor: Predictor<TInputSchema, TOutputSchema>
}

export const declareStep = <
  TStepId extends string,
  TInputSchema extends FieldSchema,
  TOutputSchema extends FieldSchema,
>(
  config: DeclareStepConfig<TStepId, TInputSchema, TOutputSchema>
): DeclaredStep<TStepId, TInputSchema, TOutputSchema> => {
  const { instructions, model, temperature, seed, ...stepConfig } = config
  const predictor = declarePredictor({
    inputSchema: config.inputSchema,
    instructions,
    model,
    name: config.id,
    outputSchema: config.outputSchema,
    seed,
    temperature,
  })

  // The predictor is the real implementation; `execute` just adapts it to the
  // step-shaped call Mastra makes. It is defined here rather than inline so the
  // same function object is both what Mastra runs and what `DeclaredStep`
  // describes — Mastra types `inputData` through its own `InferPublicSchema`,
  // which TS won't equate with zod's `output` while the schema is generic, so
  // parsing bridges the two spellings instead of asserting between them.
  const execute = ({ inputData }: { inputData: z.infer<TInputSchema> }) =>
    predictor.call(inputData)

  const step = createStep({
    ...stepConfig,
    execute: async ({ inputData }) => {
      const outputs = await predictor.call(config.inputSchema.parse(inputData))
      // SAFETY: `outputs` came back from `predictor.call`, which parses it through
      // `config.outputSchema` — the very schema Mastra derives its expected step
      // output from. Both sides agree on the runtime shape; the assertion only
      // bridges zod's `output<T>` and Mastra's `InferPublicSchema<T>`, two
      // spellings of that same type that TS cannot equate while `T` is generic.
      return outputs as InferPublicSchema<TOutputSchema>
    },
  })

  // Mastra's step stores the schemas re-wrapped as standard-schema objects and
  // types `execute` for in-workflow invocation. Overwriting all three restores the
  // library's own view of the step without disturbing what Mastra runs.
  return Object.assign(step, {
    execute,
    inputSchema: config.inputSchema,
    outputSchema: config.outputSchema,
    predictor,
  })
}
