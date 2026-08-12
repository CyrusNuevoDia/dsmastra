import { test, expect } from "bun:test";
import { z } from "zod";
import { declareStep, createWorkflow, optimize, GEPA } from "../src";
import { model } from "./_helpers";

const trainset = [
	{ input: { x: 1 }, expected: { y: 2 } },
	{ input: { x: 2 }, expected: { y: 4 } },
] as const;

test("GEPA improves arithmetic accuracy", async () => {
	// Deliberately wrong prompt.
	const badStep = declareStep({
		id: "bad-math",
		instructions: "Return y = x.", // should be x*2
		inputSchema: z.object({ x: z.number() }),
		outputSchema: z.object({ y: z.number() }),
		model,
	});

	const wf = createWorkflow({
		id: "wf-bad",
		inputSchema: z.object({ x: z.number() }),
		outputSchema: z.object({ y: z.number() }),
	})
		.then(badStep)
		.commit();

	// Helper to measure accuracy
	const accuracy = async (workflow: any) => {
		let ok = 0;
		for (const ex of trainset) {
			const { y } = await workflow.steps["bad-math"].execute({
				inputData: ex.input,
			});
			if (y === ex.expected.y) ok++;
		}
		return ok / trainset.length;
	};

	const baseAcc = await accuracy(wf);
	expect(baseAcc).toBeLessThan(1);

	const tuned = await optimize(GEPA({ maxSteps: 4, maxDemos: 2 }), wf, {
		trainset,
	});

	const tunedAcc = await accuracy(tuned);
	expect(tunedAcc).toBe(1); // must be perfect on tiny set
});
