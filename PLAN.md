Below are only the test sources expected by the plan.
Place them under tests/ (Bun will pick them up automatically).

⸻

tests/helpers/openai.ts

import { openai } from '@ai-sdk/openai';

/** Fast, cheap model for CI. */
export const fastModel = () => openai('gpt-3.5-turbo-0125');

/** Guard: skip integration tests when API key is missing. */
export const hasOpenAIKey = Boolean(process.env.OPENAI_API_KEY);


⸻

tests/predictor.int.test.ts

import { test, expect } from 'bun:test';
import { z } from 'zod';
import { declareStep } from '../src/predictor';
import { fastModel, hasOpenAIKey } from './helpers/openai';

(hasOpenAIKey ? test : test.skip)('declareStep → doubles number (live)', async () => {
  const step = declareStep({
    id: 'double',
    instructions: 'Return y = x * 2.',
    inputSchema:  z.object({ x: z.number().int().min(0).max(5) }),
    outputSchema: z.object({ y: z.number() }),
    model: fastModel(),
    temperature: 0,
    seed: 42,
  });

  const { y } = await step.execute({ inputData: { x: 3 } });
  expect(y).toBe(6);
});


⸻

tests/miprov2.int.test.ts

import { test, expect } from 'bun:test';
import { z } from 'zod';
import {
  declareStep,
  createWorkflow,
  optimize,
  MIPROv2,
} from '../src';
import { fastModel, hasOpenAIKey } from './helpers/openai';

const trainset = [
  { input: { x: 1 }, expected: { y: 2 } },
  { input: { x: 2 }, expected: { y: 4 } },
] as const;

(hasOpenAIKey ? test : test.skip)('MIPROv2 improves arithmetic accuracy', async () => {
  // Deliberately wrong prompt.
  const badStep = declareStep({
    id: 'bad-math',
    instructions: 'Return y = x.',      // should be x*2
    inputSchema:  z.object({ x: z.number() }),
    outputSchema: z.object({ y: z.number() }),
    model: fastModel(),
    temperature: 0,
    seed: 42,
  });

  const wf = createWorkflow({
    id: 'wf-bad',
    inputSchema: z.object({ x: z.number() }),
    outputSchema: z.object({ y: z.number() }),
  }).then(badStep).commit();

  // Helper to measure accuracy
  const accuracy = async (workflow: any) => {
    let ok = 0;
    for (const ex of trainset) {
      const { y } = await workflow.steps['bad-math'].execute({ inputData: ex.input });
      if (y === ex.expected.y) ok++;
    }
    return ok / trainset.length;
  };

  const baseAcc = await accuracy(wf);
  expect(baseAcc).toBeLessThan(1);

  const tuned = await optimize(
    MIPROv2({ maxSteps: 4, maxDemos: 2 }),
    wf,
    { trainset }
  );

  const tunedAcc = await accuracy(tuned);
  expect(tunedAcc).toBe(1);           // must be perfect on tiny set
});


⸻

tests/simba.int.test.ts

import { test, expect } from 'bun:test';
import { z } from 'zod';
import {
  declareStep,
  createWorkflow,
  optimize,
  SIMBA,
} from '../src';
import { fastModel, hasOpenAIKey } from './helpers/openai';

const trainset = [
  { input: { text: 'I love it' },  expected: { sent: 'positive' } },
  { input: { text: 'Hate this' },  expected: { sent: 'negative' } },
] as const;

(hasOpenAIKey ? test : test.skip)('SIMBA improves sentiment prompt', async () => {
  const badStep = declareStep({
    id: 'sentiment',
    instructions: 'Classify statement as “positive” or “negative”, but RETURN THE SAME TEXT.', // wrong
    inputSchema:  z.object({ text: z.string() }),
    outputSchema: z.object({ sent: z.enum(['positive','negative']) }),
    model: fastModel(),
    temperature: 0,
    seed: 42,
  });

  const wf = createWorkflow({
    id: 'wf-sentiment',
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ sent: z.enum(['positive','negative']) }),
  }).then(badStep).commit();

  const score = async (workflow: any) => {
    let ok = 0;
    for (const ex of trainset) {
      const { sent } = await workflow.steps.sentiment.execute({ inputData: ex.input });
      if (sent === ex.expected.sent) ok++;
    }
    return ok;
  };

  const before = await score(wf);
  expect(before).toBeLessThan(trainset.length);

  const tuned = await optimize(
    SIMBA({ maxSteps: 6, maxDemos: 2 }),
    wf,
    { trainset }
  );

  const after = await score(tuned);
  expect(after).toBe(trainset.length);
});


⸻

Notes for the implementation agent
	•	All tests really hit OpenAI; keep them deterministic (temperature:0, seed:42).
	•	They skip automatically if OPENAI_API_KEY is missing → unit CI still passes.
	•	createWorkflow().then(step).commit() must expose the step via workflow.steps[stepId] for direct execution in tests (mirrors Mastra runtime object).
	•	optimize() is expected to mutate and return a clone of the workflow with tuned prompts.
