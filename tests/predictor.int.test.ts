import { expect, test } from "bun:test";
import { z } from "zod";
import { declareStep } from "../src/predictor";
import { model } from "./_helpers";

test("declareStep → doubles number (live)", async () => {
	const step = declareStep({
		id: "double",
		instructions: "Return y = x * 2.",
		inputSchema: z.object({ x: z.number().int().min(0).max(5) }),
		outputSchema: z.object({ y: z.number() }),
		model,
	});

	const { y } = await step.execute({ inputData: { x: 3 } });
	expect(y).toBe(6);
});
