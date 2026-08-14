import { describe, expect, test } from "bun:test"

import type { LanguageModel } from "ai"
import { MockLanguageModelV4 } from "ai/test"
import { z } from "zod"

import type { Fields } from "@/fields"
import { declarePredictor } from "@/predictor"
import { createProgram } from "@/program"
import type { Program } from "@/program"
import { createRNG, samplePoisson } from "@/random"
import {
  appendADemo,
  appendARule,
  dropDemos,
  makeBuckets,
  percentile,
  simba,
  softmaxSample,
  topKPlusBaseline,
} from "@/simba"
import type { Bucket, Example, Rollout } from "@/simba"

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
  example: { inputs: {}, outputs: {} },
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

const makeProgram = (
  instructions: string,
  model: LanguageModel = deadModel
): Program => {
  const predictor = declarePredictor({
    inputSchema: z.object({ text: z.string() }),
    instructions,
    model,
    name: "classify",
    outputSchema: z.object({ label: z.enum(["pos", "neg"]) }),
  })
  return createProgram({
    forward: (call, input: Fields) => call("classify", input),
    predictors: [predictor],
  })
}

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
  test("groups model-major rollouts by example with stride bsize", () => {
    // 3 models x 2 examples, model-major: [m0e0, m0e1, m1e0, m1e1, m2e0, m2e1]
    const scores = [0.5, 0.1, 0.7, 0.9, 0.6, 0.2]
    const rollouts = scores.map((score, i) =>
      makeRollout(score, { example: { inputs: { i: i % 2 }, outputs: {} } })
    )
    const buckets = makeBuckets(rollouts, 2)
    expect(buckets).toHaveLength(2)
    // example 1 (scores 0.1, 0.9, 0.2) has gap 0.8 > example 0's 0.2, so it's first
    const first = buckets[0] as Bucket
    expect(first.rollouts.map((r) => r.score)).toEqual([0.9, 0.2, 0.1])
    expect(first.maxToMinGap).toBeCloseTo(0.8)
    expect(first.rollouts.every((r) => r.example.inputs.i === 1)).toBe(true)
    const second = buckets[1] as Bucket
    expect(second.rollouts.map((r) => r.score)).toEqual([0.7, 0.6, 0.5])
  })

  test("orders by max score then max-to-avg gap when gaps tie", () => {
    // Both examples have max-to-min gap 0.4; example 0 has higher max.
    const rollouts = [0.9, 0.6, 0.5, 0.2].map((score, i) =>
      makeRollout(score, { example: { inputs: { i: i % 2 }, outputs: {} } })
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

describe("dropDemos", () => {
  const programWithDemos = (count: number): Program => {
    const program = makeProgram("classify")
    for (const predictor of program.predictors) {
      predictor.demos = Array.from({ length: count }, (_, i) => ({
        inputs: { text: `demo ${i}` },
        outputs: { label: "pos" },
      }))
    }
    return program
  }

  test("never drops when there are no demos", () => {
    const program = programWithDemos(0)
    const dropped = dropDemos(program, 4, createRNG(0), createRNG(0))
    expect(dropped).toBe(0)
  })

  test("forces at least one drop at or over the cap, bounded by demo count", () => {
    for (let seed = 0; seed < 20; seed += 1) {
      const program = programWithDemos(4)
      const dropped = dropDemos(program, 4, createRNG(seed), createRNG(seed))
      expect(dropped).toBeGreaterThanOrEqual(1)
      expect(dropped).toBeLessThanOrEqual(4)
      const remaining = (program.predictors[0]?.demos ?? []).length
      // draws are with replacement, so realized drops can be below `dropped`
      expect(remaining).toBeGreaterThanOrEqual(4 - dropped)
      expect(remaining).toBeLessThan(4)
    }
  })

  test("applies the same index set to every predictor", () => {
    const a = declarePredictor({
      inputSchema: z.object({}),
      instructions: "a",
      model: deadModel,
      name: "a",
      outputSchema: z.object({}),
    })
    const b = a.clone()
    b.name = "b"
    for (const predictor of [a, b]) {
      predictor.demos = Array.from({ length: 6 }, (_, i) => ({
        inputs: { i },
        outputs: {},
      }))
    }
    const program = createProgram({
      forward: (call, input: Fields) => call("a", input),
      predictors: [a, b],
    })
    dropDemos(program, 4, createRNG(3), createRNG(3))
    expect(a.demos.map((d) => d.inputs.i)).toEqual(
      b.demos.map((d) => d.inputs.i)
    )
  })
})

describe("appendADemo", () => {
  const trace = [
    {
      inputs: { text: "x".repeat(50) },
      outputs: { label: "pos" },
      predictorName: "classify",
    },
    {
      inputs: { text: "second call" },
      outputs: { label: "neg" },
      predictorName: "classify",
    },
  ]

  test("skips when the best score is at or below p10", () => {
    const program = makeProgram("classify")
    const bucket = bucketOf([0.1, 0.1])
    ;(bucket.rollouts[0] as Rollout).trace = trace
    const applied = appendADemo(bucket, program, {
      demoInputFieldMaxlen: 100,
      p10: 0.1,
    })
    expect(applied).toBe(false)
    expect(program.predictors[0]?.demos).toHaveLength(0)
  })

  test("keeps only the last demo per predictor and truncates long inputs", () => {
    const program = makeProgram("classify")
    const bucket = bucketOf([0.9, 0.1])
    ;(bucket.rollouts[0] as Rollout).trace = structuredClone(trace)
    const applied = appendADemo(bucket, program, {
      demoInputFieldMaxlen: 10,
      p10: 0.1,
    })
    expect(applied).toBe(true)
    const demos = program.predictors[0]?.demos ?? []
    expect(demos).toHaveLength(1)
    expect(demos[0]?.augmented).toBe(true)
    // last trace step for the predictor wins
    expect(demos[0]?.inputs.text).toBe(
      "second cal\n\t\t... <TRUNCATED FOR BREVITY>"
    )
    expect(demos[0]?.outputs).toEqual({ label: "neg" })
  })
})

describe("appendARule", () => {
  const example: Example = {
    inputs: { text: "hello" },
    outputs: { label: "pos" },
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

  test("appends returned advice to the matching predictor's instructions", async () => {
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
    expect(program.predictors[0]?.instructions).toEndWith(
      "\n\nRULE: answer with the sentiment, never its inverse."
    )
    // verbatim OfferFeedback text reaches the prompt model
    expect(seen[0]).toContain("The module won't see its own history.")
    expect(seen[0]).toContain("[[ ## worse_program_trajectory ## ]]")
    expect(seen[0]).toContain("Module classify")
  })
})

describe("simba end-to-end", () => {
  test("returns a program that beats or equals the baseline, with candidates", async () => {
    // The mock LM answers from the text prefix only when boosted by a demo or
    // an appended RULE; otherwise it fails any input containing "hard".
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

    const student = makeProgram("Classify the sentiment.", programModel)
    const trainset: Example[] = [
      { inputs: { text: "pos easy one" }, outputs: { label: "pos" } },
      { inputs: { text: "neg easy two" }, outputs: { label: "neg" } },
      { inputs: { text: "pos hard one" }, outputs: { label: "pos" } },
      { inputs: { text: "neg hard two" }, outputs: { label: "neg" } },
    ]
    const metric = (ex: Example, prediction?: Fields) => ({
      score: prediction?.label === ex.outputs.label ? 1 : 0,
    })

    const scoreOn = async (program: typeof student) => {
      const scores = await Promise.all(
        trainset.map(async (ex) => {
          const prediction = (await program.run(ex.inputs as never)) as Record<
            string,
            unknown
          >
          return metric(ex, prediction)
        })
      )
      return scores.reduce((acc, s) => acc + s.score, 0) / trainset.length
    }

    const baselineScore = await scoreOn(student)
    expect(baselineScore).toBe(0.5)

    const result = await simba(student, trainset, {
      bsize: 4,
      maxDemos: 2,
      maxSteps: 2,
      metric,
      numCandidates: 2,
      promptModel,
      seed: 0,
    })

    const finalScore = await scoreOn(result.program)
    expect(finalScore).toBeGreaterThanOrEqual(baselineScore)
    expect(finalScore).toBe(1)

    expect(result.candidates.length).toBeGreaterThanOrEqual(1)
    const scores = result.candidates.map((c) => c.score)
    expect([...scores].toSorted((a, b) => b - a)).toEqual(scores)
    expect(scores[0]).toBe(1)
    expect(result.trialLogs).toHaveLength(2)
    // the untouched student stays in the finalist pool
    expect(scores).toContain(0.5)
  })

  test("rollout and metric throws become score 0.0, never propagate", async () => {
    const student = makeProgram(
      "Classify.",
      mockModel(() => {
        throw new Error("LM down")
      })
    )
    const trainset: Example[] = [
      { inputs: { text: "pos" }, outputs: { label: "pos" } },
      { inputs: { text: "neg" }, outputs: { label: "neg" } },
    ]
    const result = await simba(student, trainset, {
      bsize: 2,
      maxDemos: 0,
      maxSteps: 1,
      metric: () => {
        throw new Error("metric down")
      },
      numCandidates: 2,
      promptModel: deadModel,
      seed: 0,
    })
    expect(result.candidates.every((c) => c.score === 0)).toBe(true)
  })
})
