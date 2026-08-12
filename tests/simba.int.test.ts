import { test, expect } from "bun:test";
import { z } from "zod";
import { declareStep, createWorkflow, optimize, SIMBA } from "../src";
import { model } from "./_helpers";

const trainset = [
	{ input: { text: "I love it" }, expected: { sent: "positive" } },
	{ input: { text: "Hate this" }, expected: { sent: "negative" } },
] as const;

test("SIMBA improves sentiment prompt", async () => {
	const badStep = declareStep({
		id: "sentiment",
		instructions:
			'Classify statement as "positive" or "negative", but RETURN THE SAME TEXT.', // wrong
		inputSchema: z.object({ text: z.string() }),
		outputSchema: z.object({ sent: z.enum(["positive", "negative"]) }),
		model,
	});

	const wf = createWorkflow({
		id: "wf-sentiment",
		inputSchema: z.object({ text: z.string() }),
		outputSchema: z.object({ sent: z.enum(["positive", "negative"]) }),
	})
		.then(badStep)
		.commit();

	const score = async (workflow: any) => {
		let ok = 0;
		for (const ex of trainset) {
			const { sent } = await workflow.steps.sentiment.execute({
				inputData: ex.input,
			});
			if (sent === ex.expected.sent) ok++;
		}
		return ok;
	};

	const before = await score(wf);
	expect(before).toBeLessThan(trainset.length);

	const tuned = await optimize(SIMBA({ maxSteps: 6, maxDemos: 2 }), wf, {
		trainset,
	});

	const after = await score(tuned);
	expect(after).toBe(trainset.length);
});
