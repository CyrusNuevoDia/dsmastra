import { createStep } from "@mastra/core/workflows"
import { generateObject, type LanguageModel } from "ai"
import type { z } from "zod"

interface DeclareStepConfig<
  TStepId extends string,
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
> {
  id: TStepId
  inputSchema: TInputSchema
  instructions: string
  model: LanguageModel
  outputSchema: TOutputSchema
  seed?: number
  temperature?: number
}

interface DeclaredStep<
  TStepId extends string,
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
> {
  execute: (params: {
    inputData: z.infer<TInputSchema>
  }) => Promise<z.infer<TOutputSchema>>
  id: TStepId
  inputSchema: TInputSchema
  outputSchema: TOutputSchema
}

export function declareStep<
  TStepId extends string,
  TInputSchema extends z.ZodType,
  TOutputSchema extends z.ZodType,
>(
  config: DeclareStepConfig<TStepId, TInputSchema, TOutputSchema>
): DeclaredStep<TStepId, TInputSchema, TOutputSchema> {
  const { instructions, model, temperature, seed, ...stepConfig } = config

  return createStep({
    ...stepConfig,
    execute: async ({ inputData }) => {
      const { object } = await generateObject({
        model,
        prompt: `${instructions}\n\nInput:\n${JSON.stringify(inputData)}`,
        schema: config.outputSchema,
        seed,
        temperature,
      })

      // `never` satisfies Mastra's inferred execute return type without `any`.
      return object as never
    },
  }) as unknown as DeclaredStep<TStepId, TInputSchema, TOutputSchema>
}
