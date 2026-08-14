import { createStep } from "@mastra/core/workflows"
import { generateObject, type LanguageModel } from "ai"
import { z } from "zod"

/** A few-shot example rendered into the prompt between instructions and input. */
export type Demo = {
  augmented?: boolean
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
}

export type TraceStep = {
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
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
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
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
  TInputSchema extends z.ZodType = z.ZodType,
  TOutputSchema extends z.ZodType = z.ZodType,
> = {
  // biome-ignore lint/style/useConsistentMethodSignatures: method syntax keeps params bivariant so any Predictor is assignable to AnyPredictor
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

export type SchemaProperty = {
  description?: string
  type?: string
}

/** Top-level properties of a zod object schema via JSON-schema conversion; {} when the schema can't convert. */
export function schemaProperties(
  schema: z.ZodType | undefined
): Record<string, SchemaProperty> {
  if (!schema) {
    return {}
  }
  try {
    const jsonSchema = z.toJSONSchema(schema) as {
      properties?: Record<string, SchemaProperty>
    }
    return jsonSchema.properties ?? {}
  } catch {
    return {}
  }
}

function renderPrompt(
  instructions: string,
  demos: Demo[],
  inputs: unknown
): string {
  const parts = [instructions]
  for (const demo of demos) {
    parts.push(
      `Example:\nInput:\n${JSON.stringify(demo.inputs)}\nOutput:\n${JSON.stringify(demo.outputs)}`
    )
  }
  parts.push(`Input:\n${JSON.stringify(inputs)}`)
  return parts.join("\n\n")
}

export function declarePredictor<
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
>(
  config: PredictorConfig<TInputSchema, TOutputSchema>
): Predictor<TInputSchema, TOutputSchema> {
  const predictor: Predictor<TInputSchema, TOutputSchema> = {
    call: async (inputs, ctx) => {
      const { object } = await generateObject({
        model: ctx?.model ?? predictor.model,
        prompt: renderPrompt(predictor.instructions, predictor.demos, inputs),
        schema: predictor.outputSchema,
        seed: ctx?.seed ?? predictor.seed,
        temperature: ctx?.temperature ?? predictor.temperature,
      })
      ctx?.trace?.push({
        inputs: inputs as Record<string, unknown>,
        outputs: object as Record<string, unknown>,
        predictorName: predictor.name,
      })
      return object as z.infer<TOutputSchema>
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
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
> = {
  id: TStepId
  inputSchema: TInputSchema
  instructions: string
  model: LanguageModel
  outputSchema: TOutputSchema
  seed?: number
  temperature?: number
}

type DeclaredStep<
  TStepId extends string,
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
> = {
  execute: (params: {
    inputData: z.infer<TInputSchema>
  }) => Promise<z.infer<TOutputSchema>>
  id: TStepId
  inputSchema: TInputSchema
  outputSchema: TOutputSchema
  predictor: Predictor<TInputSchema, TOutputSchema>
}

export function declareStep<
  TStepId extends string,
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
>(
  config: DeclareStepConfig<TStepId, TInputSchema, TOutputSchema>
): DeclaredStep<TStepId, TInputSchema, TOutputSchema> {
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

  const step = createStep({
    ...stepConfig,
    // `never` satisfies Mastra's inferred execute return type without `any`.
    execute: async ({ inputData }) =>
      (await predictor.call(inputData as never)) as never,
  }) as unknown as DeclaredStep<TStepId, TInputSchema, TOutputSchema>
  step.predictor = predictor
  return step
}
