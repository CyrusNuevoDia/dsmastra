import { describe, expect, test } from "bun:test"

import { createWorkflow } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { MockLanguageModelV4 } from "ai/test"
import { z } from "zod"

import type { Fields } from "@/fields"
import {
  appendAnExample,
  appendARule,
  dropExamples,
  makeBuckets,
  percentile,
  simba,
  softmaxSample,
  topKPlusBaseline,
} from "@/optimizers/simba"
import type { Bucket, Rollout } from "@/optimizers/simba"
import type { Prompts } from "@/optimizers/utils"
import type { Example, Program } from "@/program"
import { createProgram } from "@/program"
import { createRNG, samplePoisson } from "@/random"
import { declareStep } from "@/step"

import { fakeScorer } from "./helpers"

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

const deadModel = mockModel(() => {
  throw new Error("LM should not be called")
})

const makeRollout = (
  score: number,
  overrides: Partial<Rollout> = {}
): Rollout => ({
  example: { inputData: {}, outputData: {} },
  outputMetadata: {},
  prediction: {},
  score,
  trace: [],
  ...overrides,
})

const bucketOf = (scores: number[]): Bucket => {
  const rollouts = scores
    .map((score) => makeRollout(score))
    .toSorted((a, b) => b.score - a.score)
  const max = (rollouts[0] as Rollout).score
  const min = (rollouts.at(-1) as Rollout).score
  const avg = scores.reduce((acc, s) => acc + s, 0) / scores.length
  return {
    maxScore: max,
    maxToAvgGap: max - avg,
    maxToMinGap: max - min,
    rollouts,
  }
}

const classifyStep = (description: string, model: LanguageModel = deadModel) =>
  declareStep({
    description,
    id: "classify",
    inputSchema: z.object({ text: z.string() }),
    model,
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })

const makeProgram = (
  description: string,
  model: LanguageModel = deadModel
): Program =>
  createProgram({
    forward: (call, inputData: Fields) => call("classify", inputData),
    steps: [classifyStep(description, model)],
  })

describe("topKPlusBaseline", () => {
  test("takes top k by average, ties toward lower index", () => {
    // ties between 1 and 3 break toward 1; slot for 3 is overwritten by the baseline
    expect(topKPlusBaseline([0, 0.5, 0.9, 0.5], 3)).toEqual([2, 1, 0])
    expect(topKPlusBaseline([0.1, 0.5, 0.9, 0.5, 0], 4)).toEqual([2, 1, 3, 0])
  })

  test("forces the baseline into the last slot when absent", () => {
    expect(topKPlusBaseline([0, 0.9, 0.8, 0.7], 2)).toEqual([1, 0])
  })

  test("k=1 collapses to the baseline", () => {
    expect(topKPlusBaseline([0, 0.9], 1)).toEqual([0])
  })

  test("dedupes preserving order and may return fewer than k", () => {
    const result = topKPlusBaseline([0, 0.9], 5)
    expect(result).toEqual([1, 0])
    expect(new Set(result).size).toBe(result.length)
  })
})

describe("softmaxSample", () => {
  test("throws on empty index list", () => {
    expect(() => softmaxSample(createRNG(0), [], [], 0.2)).toThrow()
  })

  test("falls back to uniform when the weight sum underflows to 0", () => {
    const avg = [-1e9, -1e9, -1e9]
    const picks = new Set<number>()
    const rng = createRNG(0)
    for (let i = 0; i < 100; i += 1) {
      picks.add(softmaxSample(rng, [0, 1, 2], avg, 0.2))
    }
    expect(picks).toEqual(new Set([0, 1, 2]))
  })

  test("overwhelmingly prefers a much stronger program at low temperature", () => {
    const rng = createRNG(1)
    for (let i = 0; i < 50; i += 1) {
      expect(softmaxSample(rng, [0, 1], [0, 10], 0.2)).toBe(1)
    }
  })
})

describe("percentile", () => {
  test("uses NumPy-style linear interpolation", () => {
    expect(percentile([0, 1], 10)).toBeCloseTo(0.1)
    expect(percentile([0, 1], 90)).toBeCloseTo(0.9)
    expect(percentile([1, 2, 3, 4], 50)).toBeCloseTo(2.5)
  })
})

describe("samplePoisson", () => {
  test("lambda 0 always yields 0", () => {
    const rng = createRNG(0)
    expect(samplePoisson(rng, 0)).toBe(0)
  })

  test("mean approximates lambda", () => {
    const rng = createRNG(0)
    let total = 0
    const draws = 2000
    for (let i = 0; i < draws; i += 1) {
      total += samplePoisson(rng, 2)
    }
    expect(total / draws).toBeGreaterThan(1.7)
    expect(total / draws).toBeLessThan(2.3)
  })
})

describe("makeBuckets", () => {
  test("groups model-major rollouts by example with stride batchSize", () => {
    // 3 models x 2 examples, model-major: [m0e0, m0e1, m1e0, m1e1, m2e0, m2e1]
    const scores = [0.5, 0.1, 0.7, 0.9, 0.6, 0.2]
    const rollouts = scores.map((score, i) =>
      makeRollout(score, {
        example: { inputData: { i: i % 2 }, outputData: {} },
      })
    )
    const buckets = makeBuckets(rollouts, 2)
    expect(buckets).toHaveLength(2)
    // example 1 (scores 0.1, 0.9, 0.2) has gap 0.8 > example 0's 0.2, so it's first
    const first = buckets[0] as Bucket
    expect(first.rollouts.map((r) => r.score)).toEqual([0.9, 0.2, 0.1])
    expect(first.maxToMinGap).toBeCloseTo(0.8)
    expect(first.rollouts.every((r) => r.example.inputData.i === 1)).toBe(true)
    const second = buckets[1] as Bucket
    expect(second.rollouts.map((r) => r.score)).toEqual([0.7, 0.6, 0.5])
  })

  test("orders by max score then max-to-avg gap when gaps tie", () => {
    // Both examples have max-to-min gap 0.4; example 0 has higher max.
    const rollouts = [0.9, 0.6, 0.5, 0.2].map((score, i) =>
      makeRollout(score, {
        example: { inputData: { i: i % 2 }, outputData: {} },
      })
    )
    const buckets = makeBuckets(rollouts, 2)
    expect((buckets[0] as Bucket).maxScore).toBeCloseTo(0.9)
  })

  test("copies rollout records so strategies cannot mutate shared state", () => {
    const rollouts = [makeRollout(1), makeRollout(0)]
    const buckets = makeBuckets(rollouts, 1)
    ;((buckets[0] as Bucket).rollouts[0] as Rollout).score = 42
    expect(rollouts.map((r) => r.score).toSorted((a, b) => a - b)).toEqual([
      0, 1,
    ])
  })
})

describe("dropExamples", () => {
  const programWithExamples = (count: number): Program => {
    const program = makeProgram("classify")
    for (const step of program.steps) {
      step.examples = Array.from({ length: count }, (_, i) => ({
        inputData: { text: `example ${i}` },
        outputData: { label: "pos" },
      }))
    }
    return program
  }

  test("never drops when there are no examples", () => {
    const program = programWithExamples(0)
    const dropped = dropExamples(program, 4, createRNG(0), createRNG(0))
    expect(dropped).toBe(0)
  })

  test("forces at least one drop at or over the cap, bounded by example count", () => {
    for (let seed = 0; seed < 20; seed += 1) {
      const program = programWithExamples(4)
      const dropped = dropExamples(program, 4, createRNG(seed), createRNG(seed))
      expect(dropped).toBeGreaterThanOrEqual(1)
      expect(dropped).toBeLessThanOrEqual(4)
      const remaining = (program.steps[0]?.examples ?? []).length
      // draws are with replacement, so realized drops can be below `dropped`
      expect(remaining).toBeGreaterThanOrEqual(4 - dropped)
      expect(remaining).toBeLessThan(4)
    }
  })

  test("applies the same index set to every step", () => {
    const stepOf = (id: "a" | "b") =>
      declareStep({
        description: id,
        id,
        inputSchema: z.object({}),
        model: deadModel,
        outputSchema: z.object({}),
      })
    const a = stepOf("a")
    const b = stepOf("b")
    for (const step of [a, b]) {
      step.examples = Array.from({ length: 6 }, (_, i) => ({
        inputData: { i },
        outputData: {},
      }))
    }
    const program = createProgram({
      forward: (call, inputData: Fields) => call("a", inputData),
      steps: [a, b],
    })
    dropExamples(program, 4, createRNG(3), createRNG(3))
    expect(a.examples.map((e) => e.inputData.i)).toEqual(
      b.examples.map((e) => e.inputData.i)
    )
  })
})

describe("appendAnExample", () => {
  const trace = [
    {
      inputData: { text: "x".repeat(50) },
      outputData: { label: "pos" },
      stepId: "classify",
    },
    {
      inputData: { text: "second call" },
      outputData: { label: "neg" },
      stepId: "classify",
    },
  ]

  test("skips when the best score is at or below p10", () => {
    const program = makeProgram("classify")
    const bucket = bucketOf([0.1, 0.1])
    ;(bucket.rollouts[0] as Rollout).trace = trace
    const applied = appendAnExample(bucket, program, {
      maxFewShotInputLength: 100,
      p10: 0.1,
    })
    expect(applied).toBe(false)
    expect(program.steps[0]?.examples).toHaveLength(0)
  })

  test("keeps only the last example per step and truncates long inputData", () => {
    const program = makeProgram("classify")
    const bucket = bucketOf([0.9, 0.1])
    ;(bucket.rollouts[0] as Rollout).trace = structuredClone(trace)
    const applied = appendAnExample(bucket, program, {
      maxFewShotInputLength: 10,
      p10: 0.1,
    })
    expect(applied).toBe(true)
    const examples = program.steps[0]?.examples ?? []
    expect(examples).toHaveLength(1)
    // last trace step for the step wins
    expect(examples[0]?.inputData.text).toBe(
      "second cal\n\t\t... <TRUNCATED FOR BREVITY>"
    )
    expect(examples[0]?.outputData).toEqual({ label: "neg" })
  })
})

describe("appendARule", () => {
  const example: Example = {
    inputData: { text: "hello" },
    outputData: { label: "pos" },
  }

  test("skips when good is at or below p10", async () => {
    const program = makeProgram("classify")
    const applied = await appendARule(bucketOf([0.1, 0]), program, {
      p10: 0.1,
      p90: 0.9,
      promptModel: deadModel,
    })
    expect(applied).toBe(false)
  })

  test("skips when bad is at or above p90", async () => {
    const program = makeProgram("classify")
    const applied = await appendARule(bucketOf([1, 0.95]), program, {
      p10: 0.1,
      p90: 0.9,
      promptModel: deadModel,
    })
    expect(applied).toBe(false)
  })

  test("appends returned advice to the matching step's description", async () => {
    const program = makeProgram("classify")
    const seen: string[] = []
    const promptModel = mockModel((promptText) => {
      seen.push(promptText)
      return JSON.stringify({
        discussion: "the worse run inverted the label",
        moduleAdvice: {
          classify: "RULE: answer with the sentiment, never its inverse.",
          unknown_module: "ignored",
        },
      })
    })
    const bucket = bucketOf([1, 0])
    ;(bucket.rollouts[0] as Rollout).example = example
    const applied = await appendARule(bucket, program, {
      p10: 0.1,
      p90: 0.9,
      promptModel,
    })
    expect(applied).toBe(true)
    expect(program.steps[0]?.description).toEndWith(
      "\n\nRULE: answer with the sentiment, never its inverse."
    )
    // verbatim OfferFeedback text reaches the prompt model
    expect(seen[0]).toContain("The module won't see its own history.")
    expect(seen[0]).toContain("[[ ## worse_program_trajectory ## ]]")
    expect(seen[0]).toContain("Module classify")
  })
})

describe("simba end-to-end", () => {
  test("returns a workflow that beats the baseline and checkpoints prompts", async () => {
    // The mock LM answers from the text prefix only when boosted by an example
    // or an appended RULE; otherwise it fails any input containing "hard".
    const programModel = mockModel((promptText) => {
      const liveInput = promptText.split("Input:\n").at(-1) ?? ""
      const boosted =
        promptText.includes("Example:") || promptText.includes("RULE:")
      const wantsPos = liveInput.includes("pos")
      const hard = liveInput.includes("hard")
      const label = (boosted || !hard ? wantsPos : !wantsPos) ? "pos" : "neg"
      return JSON.stringify({ label })
    })
    const promptModel = mockModel(() =>
      JSON.stringify({
        discussion: "the classifier mishandles hard inputs",
        moduleAdvice: { classify: "RULE: trust the text's stated sentiment." },
      })
    )

    const step = classifyStep("Classify the sentiment.", programModel)
    const workflow = createWorkflow({
      id: "sentiment",
      inputSchema: z.object({ text: z.string() }),
      outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
    })
      .then(step)
      .commit()
    const trainingSet: Example[] = [
      { inputData: { text: "pos easy one" }, outputData: { label: "pos" } },
      { inputData: { text: "neg easy two" }, outputData: { label: "neg" } },
      { inputData: { text: "pos hard one" }, outputData: { label: "pos" } },
      { inputData: { text: "neg hard two" }, outputData: { label: "neg" } },
    ]
    const scorer = fakeScorer((gold, prediction) =>
      prediction?.label === gold.outputData.label ? 1 : 0
    )

    const saved: Prompts[] = []
    const result = await simba(workflow, {
      batchSize: 4,
      candidates: 2,
      maxFewShotExamples: 2,
      maxSteps: 2,
      promptModel,
      savePrompts: (prompts) => {
        saved.push(structuredClone(prompts))
        return Promise.resolve()
      },
      scorer,
      seed: 0,
      trainingSet,
    })

    expect(result.score).toBe(1)
    // Tuning lands in place: the caller's own step reference carries the
    // updated prompt state.
    expect(
      step.description !== "Classify the sentiment." || step.examples.length > 0
    ).toBe(true)
    // Finalists come back as [snapshot, { score }] pairs, best first, and the
    // top-level score is the winner's.
    expect(result.candidates.length).toBeGreaterThanOrEqual(1)
    for (const [snapshot, { score: candidateScore }] of result.candidates) {
      expect(snapshot.version).toBe(1)
      expect(Object.keys(snapshot.steps)).toEqual(["classify"])
      expect(candidateScore).toBeGreaterThanOrEqual(0)
    }
    expect(result.candidates[0]?.[1].score).toBe(result.score)
    // savePrompts fired on every step winner (2) plus once at completion, and
    // each payload is JSON-safe with the workflow's step ids.
    expect(saved.length).toBe(3)
    for (const prompts of saved) {
      expect(prompts.version).toBe(1)
      expect(Object.keys(prompts.steps)).toEqual(["classify"])
      // oxlint-disable-next-line unicorn/prefer-structured-clone -- the JSON round-trip IS what this asserts
      expect(JSON.parse(JSON.stringify(prompts))).toEqual(prompts)
    }
  })

  test("rollout and metric throws become score 0.0, never propagate", async () => {
    const step = classifyStep(
      "Classify.",
      mockModel(() => {
        throw new Error("LM down")
      })
    )
    const workflow = createWorkflow({
      id: "broken",
      inputSchema: z.object({ text: z.string() }),
      outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
    })
      .then(step)
      .commit()
    const trainingSet: Example[] = [
      { inputData: { text: "pos" }, outputData: { label: "pos" } },
      { inputData: { text: "neg" }, outputData: { label: "neg" } },
    ]
    const result = await simba(workflow, {
      batchSize: 2,
      candidates: 2,
      maxFewShotExamples: 0,
      maxSteps: 1,
      promptModel: deadModel,
      savePrompts: () => Promise.resolve(),
      scorer: fakeScorer(() => {
        throw new Error("scorer down")
      }),
      seed: 0,
      trainingSet,
    })
    expect(result.score).toBe(0)
  })
})
