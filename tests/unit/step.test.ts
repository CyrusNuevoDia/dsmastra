import { expect, test } from "bun:test"

import type { LanguageModelV4CallOptions } from "@ai-sdk/provider"
import { createWorkflow } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { MockLanguageModelV4 } from "ai/test"
import { z } from "zod"

import { loadPrompts, promptsOf } from "#src/optimizers/utils"
import { declareStep } from "#src/step"
import { promptStep } from "#tests/unit/helpers"

const usage = {
  inputTokens: {
    cacheRead: undefined,
    cacheWrite: undefined,
    noCache: 10,
    total: 10,
  },
  outputTokens: { reasoning: undefined, text: 20, total: 20 },
}

test("declareStep forwards AI SDK call settings to generateText", async () => {
  const seen: LanguageModelV4CallOptions[] = []
  const model = new MockLanguageModelV4({
    doGenerate: (options) => {
      seen.push(options)
      return Promise.resolve({
        content: [{ text: JSON.stringify({ y: 2 }), type: "text" as const }],
        finishReason: { raw: undefined, unified: "stop" as const },
        usage,
        warnings: [],
      })
    },
  }) as LanguageModel

  const step = declareStep({
    description: "Double the input.",
    id: "double",
    inputSchema: z.object({ x: z.number() }),
    maxOutputTokens: 99,
    model,
    outputSchema: z.object({ y: z.number() }),
    seed: 7,
    temperature: 0.3,
    topK: 5,
    topP: 0.9,
  })

  const outputData = await step.execute({ inputData: { x: 1 } })
  expect(outputData).toEqual({ y: 2 })
  const options = seen[0] as LanguageModelV4CallOptions
  expect(options.temperature).toBe(0.3)
  expect(options.seed).toBe(7)
  expect(options.topP).toBe(0.9)
  expect(options.topK).toBe(5)
  expect(options.maxOutputTokens).toBe(99)
})

test("tuned description and examples reach the rendered prompt; prompt snapshots isolate them", async () => {
  const prompts: string[] = []
  const model = new MockLanguageModelV4({
    doGenerate: (options) => {
      const text = options.prompt
        .flatMap((message) =>
          typeof message.content === "string"
            ? [message.content]
            : message.content.map((part) =>
                part.type === "text" ? part.text : ""
              )
        )
        .join("\n")
      prompts.push(text)
      return Promise.resolve({
        content: [{ text: JSON.stringify({ y: 4 }), type: "text" as const }],
        finishReason: { raw: undefined, unified: "stop" as const },
        usage,
        warnings: [],
      })
    },
  }) as LanguageModel

  const step = declareStep({
    description: "Original.",
    id: "double",
    inputSchema: z.object({ x: z.number() }),
    model,
    outputSchema: z.object({ y: z.number() }),
  })

  // Tuning mutates the step's own fields; execute reads them live.
  step.description = "Tuned: double it."
  step.examples.push({ inputData: { x: 1 }, outputData: { y: 2 } })
  await step.execute({ inputData: { x: 2 } })
  expect(prompts[0]).toContain("Tuned: double it.")
  expect(prompts[0]).toContain('{"x":1}')

  const workflow = createWorkflow({
    id: "double-workflow",
    inputSchema: z.object({ x: z.number() }),
    outputSchema: z.object({ y: z.number() }),
  })
    .then(step)
    .commit()
  const snapshot = promptsOf(workflow)
  const isolated = structuredClone(snapshot)
  expect(promptStep(isolated, "double").description).toBe("Tuned: double it.")
  expect(promptStep(isolated, "double").examples).toEqual(step.examples)
  promptStep(isolated, "double").description = "Diverged."
  promptStep(isolated, "double").examples.pop()
  loadPrompts(workflow, snapshot)
  expect(step.description).toBe("Tuned: double it.")
  expect(step.examples).toHaveLength(1)
})
