import { expect, test } from "bun:test"

import { createStep, createWorkflow } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { MockLanguageModelV4 } from "ai/test"
import { z } from "zod"

import { createLabeledFewShotWorkflow } from "#src/optimizers/labeled-few-shot"
import { loadPrompts, promptsOf, runWith } from "#src/optimizers/utils"
import type { TraceStep } from "#src/step"
import { declareStep } from "#src/step"
import {
  fakeScorer,
  fakeStep,
  promptStep,
  runOptimizer,
} from "#tests/unit/helpers"

const usage = {
  inputTokens: {
    cacheRead: undefined,
    cacheWrite: undefined,
    noCache: 10,
    total: 10,
  },
  outputTokens: { reasoning: undefined, text: 20, total: 20 },
}

const mockModel = (respond: (promptText: string) => string): LanguageModel =>
  new MockLanguageModelV4({
    doGenerate: (options) => {
      const promptText = options.prompt
        .flatMap((message) =>
          typeof message.content === "string"
            ? [message.content]
            : message.content.map((part) =>
                part.type === "text" ? part.text : ""
              )
        )
        .join("\n")
      return Promise.resolve({
        content: [{ text: respond(promptText), type: "text" as const }],
        finishReason: { raw: undefined, unified: "stop" as const },
        usage,
        warnings: [],
      })
    },
  }) as LanguageModel

test("a tuned workflow runs through Mastra's engine with the tuned prompt state", async () => {
  const prompts: string[] = []
  const step = declareStep({
    description: "Classify the sentiment.",
    id: "classify",
    inputSchema: z.object({ text: z.string() }),
    model: mockModel((promptText) => {
      prompts.push(promptText)
      return JSON.stringify({ label: "pos" })
    }),
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })
  const workflow = createWorkflow({
    id: "sentiment",
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })
    .then(step)
    .commit()

  loadPrompts(workflow, {
    steps: {
      classify: {
        description: "TUNED: trust the stated sentiment.",
        examples: [],
      },
    },
    version: 1,
  })

  const run = await workflow.createRun()
  const result = await run.start({ inputData: { text: "great stuff" } })

  expect(result.status).toBe("success")
  if (result.status === "success") {
    expect(result.result).toEqual({ label: "pos" })
  }
  // The engine executed the step with the tuned description in the prompt.
  expect(prompts.join("\n")).toContain("TUNED: trust the stated sentiment.")
})

test("engine rollouts honor RunContext overrides and capture per-step traces", async () => {
  const seen: { seed: number | undefined; temperature: number | undefined }[] =
    []
  const step = declareStep({
    description: "Classify the sentiment.",
    id: "classify",
    inputSchema: z.object({ text: z.string() }),
    model: new MockLanguageModelV4({
      doGenerate: (options) => {
        seen.push({ seed: options.seed, temperature: options.temperature })
        return Promise.resolve({
          content: [
            { text: JSON.stringify({ label: "pos" }), type: "text" as const },
          ],
          finishReason: { raw: undefined, unified: "stop" as const },
          usage,
          warnings: [],
        })
      },
    }) as LanguageModel,
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })
  const workflow = createWorkflow({
    id: "override-wf",
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })
    .then(step)
    .commit()

  const prompts = promptsOf(workflow)
  const trace: TraceStep[] = []
  const output = await runWith(
    workflow,
    prompts,
    { text: "great" },
    { seed: 9, temperature: 0.7, trace }
  )

  expect(output).toEqual({ label: "pos" })
  expect(seen).toEqual([{ seed: 9, temperature: 0.7 }])
  expect(trace).toEqual([
    {
      inputData: { text: "great" },
      outputData: { label: "pos" },
      stepId: "classify",
    },
  ])
})

test("engine rollouts run parallel graphs; tuning lands on declarative steps only", async () => {
  const double = fakeStep("double", (inputData) => ({
    y: (inputData.x as number) * 2,
  }))
  const inc = fakeStep("inc", (inputData) => ({
    y: (inputData.x as number) + 1,
  }))
  const merge = createStep({
    execute: ({ inputData }) => {
      const branches = inputData as Record<string, { y: number }>
      return Promise.resolve({
        sum: (branches.double?.y ?? 0) + (branches.inc?.y ?? 0),
      })
    },
    id: "merge",
    inputSchema: z.record(z.string(), z.unknown()),
    outputSchema: z.record(z.string(), z.unknown()),
  })
  const workflow = createWorkflow({
    id: "parallel-wf",
    inputSchema: z.record(z.string(), z.unknown()),
    outputSchema: z.record(z.string(), z.unknown()),
  })
    .parallel([double, inc])
    .then(merge)
    .commit()

  // A rollout through the engine executes the parallel section correctly.
  const trace: TraceStep[] = []
  const output = await runWith(
    workflow,
    promptsOf(workflow),
    { x: 3 },
    { trace }
  )
  expect(output).toEqual({ sum: 10 })
  expect(trace.map((t) => t.stepId).toSorted()).toEqual(["double", "inc"])

  // Optimizing tunes the declarative steps in place; the merge step is untouched.
  const { candidates, score } = await runOptimizer(
    createLabeledFewShotWorkflow(workflow, {
      maxFewShotExamples: 1,
      savePrompts: () => Promise.resolve(),
      scorer: fakeScorer((gold, prediction) =>
        prediction?.sum === gold.outputData.sum ? 1 : 0
      ),
      trainingSet: [{ inputData: { x: 1 }, outputData: { sum: 4 } }],
    })
  )
  expect(double.examples).toHaveLength(1)
  expect(inc.examples).toHaveLength(1)
  // The compiled workflow was evaluated: x=1 → 2 + 2 = 4, exact match.
  expect(score).toBe(1)
  const [snapshot, { score: candidateScore }] = candidates[0] ?? [
    { steps: {}, version: 1 },
    { score: Number.NaN },
  ]
  expect(candidateScore).toBe(score)
  expect(Object.keys(snapshot.steps).toSorted()).toEqual(["double", "inc"])
})

test("concurrent candidates never observe mixed prompt state", async () => {
  // The step records the LIVE description it sees while executing — if two
  // candidates' rollouts interleaved on the shared step, a rollout would
  // observe the other candidate's description.
  const observed: { description: string; tag: string }[] = []
  // oxlint-disable-next-line prefer-const -- assigned right below; the fn closure needs the binding
  let probe: ReturnType<typeof fakeStep>
  probe = fakeStep("probe", (inputData) => {
    observed.push({
      description: probe.description,
      tag: inputData.tag as string,
    })
    return { tag: inputData.tag }
  })
  const workflow = createWorkflow({
    id: "isolation-wf",
    inputSchema: z.record(z.string(), z.unknown()),
    outputSchema: z.record(z.string(), z.unknown()),
  })
    .then(probe)
    .commit()

  const base = promptsOf(workflow)
  const candidateA = structuredClone(base)
  const candidateB = structuredClone(base)
  promptStep(candidateA, "probe").description = "CANDIDATE-A"
  promptStep(candidateB, "probe").description = "CANDIDATE-B"

  // Launch both candidates' batches concurrently: the gate must serialize
  // them so every rollout sees exactly its own candidate's state.
  await Promise.all([
    ...["a1", "a2", "a3"].map((tag) => runWith(workflow, candidateA, { tag })),
    ...["b1", "b2", "b3"].map((tag) => runWith(workflow, candidateB, { tag })),
  ])

  expect(observed).toHaveLength(6)
  for (const { description, tag } of observed) {
    expect(description).toBe(
      tag.startsWith("a") ? "CANDIDATE-A" : "CANDIDATE-B"
    )
  }
})
