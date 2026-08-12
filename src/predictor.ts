import { createStep } from "@mastra/core/workflows";
import { generateObject, type LanguageModel } from "ai";
import { z } from "zod";

interface DeclareStepConfig<
	TStepId extends string,
	TInputSchema extends z.ZodType,
	TOutputSchema extends z.ZodType,
> {
	id: TStepId;
	instructions: string;
	inputSchema: TInputSchema;
	outputSchema: TOutputSchema;
	model: LanguageModel;
	temperature?: number;
	seed?: number;
}

interface DeclaredStep<
	TStepId extends string,
	TInputSchema extends z.ZodType,
	TOutputSchema extends z.ZodType,
> {
	id: TStepId;
	inputSchema: TInputSchema;
	outputSchema: TOutputSchema;
	execute(params: { inputData: z.infer<TInputSchema> }): Promise<z.infer<TOutputSchema>>;
}

export function declareStep<
	TStepId extends string,
	TInputSchema extends z.ZodType,
	TOutputSchema extends z.ZodType,
>(
	config: DeclareStepConfig<TStepId, TInputSchema, TOutputSchema>,
): DeclaredStep<TStepId, TInputSchema, TOutputSchema> {
	const { instructions, model, temperature, seed, ...stepConfig } = config;

	return createStep({
		...stepConfig,
		execute: async ({ inputData }) => {
			const { object } = await generateObject({
				model,
				schema: config.outputSchema,
				temperature,
				seed,
				prompt: `${instructions}\n\nInput:\n${JSON.stringify(inputData)}`,
			});

			return object as any;
		},
	}) as unknown as DeclaredStep<TStepId, TInputSchema, TOutputSchema>;
}
