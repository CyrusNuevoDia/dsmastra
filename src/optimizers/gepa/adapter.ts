import { generateText } from "ai"
import type { LanguageModel } from "ai"
import type { z } from "zod"

import { at } from "@/collections"
import type { Fields } from "@/fields"
import type { MetricOutput, MetricResult } from "@/metrics"
import type {
  Candidate,
  EvaluationBatch,
  GEPAAdapter,
  GEPATraceStep,
  ReflectiveDataset,
  ReflectiveExample,
  RNG,
  Trajectory,
} from "@/optimizers/gepa/engine"
import type { Example, Program } from "@/program"
import { schemaProperties } from "@/schema"

/** A metric result that names the feedback field GEPA's reflection LM reads. */
export type ScoreWithFeedback = MetricResult & { feedback?: string }

/**
 * GEPA metric contract: called with (gold, prediction, trace) at module level
 * and with (gold, prediction, fullTrace, stepId, stepTrace) for
 * per-step feedback. A result without a `feedback` field gets the default
 * feedback string.
 */
export type GEPAMetric<TInput = Fields, TOutput = Fields> = (
  gold: Example<TInput, TOutput>,
  prediction: TOutput | null,
  trace: GEPATraceStep[] | null,
  stepId?: string,
  stepTrace?: GEPATraceStep[]
) => MetricOutput

export type ReflectionModel =
  | LanguageModel
  | ((prompt: string) => Promise<string>)

const defaultFeedback = (score: number) =>
  `This trajectory got a score of ${score}.`

export const runFeedbackMetric = async <TInput, TOutput>(
  metric: GEPAMetric<TInput, TOutput>,
  gold: Example<TInput, TOutput>,
  prediction: TOutput | null,
  trace: GEPATraceStep[] | null,
  stepId?: string,
  stepTrace?: GEPATraceStep[]
): Promise<Required<Pick<ScoreWithFeedback, "feedback" | "score">>> => {
  const { score, ...metadata } = await metric(
    gold,
    prediction,
    trace,
    stepId,
    stepTrace
  )
  const { feedback } = metadata
  return {
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- `feedback` is an optional field on an open metric result: the metric may omit it or return something else entirely, and the default string is the fallback
    feedback: typeof feedback === "string" ? feedback : defaultFeedback(score),
    score,
  }
}

// --- Reflective-dataset rendering -------------------------------------------

const PARSE_FAILURE_OUTPUT = (raw: string) =>
  `Couldn't parse the output as per the expected output format. The model's raw response was:\n\`\`\`\n${raw}\n\`\`\`\n\n`

const PARSE_FAILURE_FEEDBACK_PREFIX =
  "Your output failed to parse. Follow this structure:\n"

const expectedStructure = (outputSchema: z.ZodType | undefined): string =>
  Object.entries(schemaProperties(outputSchema))
    .map(([name, prop]) => `${name}: ${prop.type ?? "unknown"}`)
    .join("\n")

const MAX_HEADER_DEPTH = 6

/**
 * Byte-for-byte port of the Python renderer: scalars end with a blank line
 * (`value\n\n`), headers with a single newline, empty dicts/lists add a bare
 * newline, and the depth cap applies on recursion.
 */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- a step output is whatever the model produced and the schema accepted, so the renderer genuinely meets an open value
const renderValue = (value: unknown, level: number): string => {
  const header = "#".repeat(level)
  const nextLevel = Math.min(level + 1, MAX_HEADER_DEPTH)
  if (Array.isArray(value)) {
    let s = ""
    for (const [k, item] of value.entries()) {
      s += `${header} Item ${k + 1}\n${renderValue(item, nextLevel)}`
    }
    if (value.length === 0) {
      s += "\n"
    }
    return s
  }
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- objects nest into headers; every other JSON case renders as a scalar below
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value)
    let s = ""
    for (const [key, sub] of entries) {
      s += `${header} ${key}\n${renderValue(sub, nextLevel)}`
    }
    if (entries.length === 0) {
      s += "\n"
    }
    return s
  }
  return `${String(value).trim()}\n\n`
}

/** Markdown rendering of the reflective dataset for `<side_info>`. */
export const renderSideInfo = (examples: ReflectiveExample[]): string =>
  examples
    .map((example, n) => {
      let s = `# Example ${n + 1}\n`
      for (const [key, value] of Object.entries(example)) {
        s += `## ${key}\n${renderValue(value, 3)}`
      }
      return s
    })
    .join("\n\n")

// --- Instruction-proposal prompt (verbatim template) ------------------------

export const buildProposalPrompt = (
  currentInstructions: string,
  sideInfo: string
): string =>
  `I provided an assistant with the following instructions to perform a task for me:
\`\`\`
${currentInstructions}
\`\`\`

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
\`\`\`
${sideInfo}
\`\`\`

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within \`\`\` blocks.`

const LANGUAGE_TAG = /^\S*\n/u
const LEADING_FENCE = /^```\S*\n?/u

/**
 * Byte-for-byte port of Python's output_extractor (which receives the
 * response pre-stripped): text between the first and last fences with a
 * leading language tag stripped; incomplete blocks fall back to stripping a
 * leading fence (+ optional language tag and newline) or a trailing fence,
 * else the whole trimmed response.
 */
export const extractInstructionText = (response: string): string => {
  const lmOut = response.trim()
  const start = lmOut.indexOf("```") + 3
  const end = lmOut.lastIndexOf("```")
  if (start >= end) {
    if (lmOut.startsWith("```")) {
      const match = LEADING_FENCE.exec(lmOut)
      if (match) {
        return lmOut.slice(match[0].length).trim()
      }
      return lmOut
    }
    if (lmOut.endsWith("```")) {
      return lmOut.slice(0, -3).trim()
    }
    return lmOut
  }
  let content = lmOut.slice(start, end)
  const tag = LANGUAGE_TAG.exec(content)
  if (tag) {
    content = content.slice(tag[0].length)
  }
  return content.trim()
}

const stringifyFields = (fields: Fields): Record<string, string> =>
  Object.fromEntries(
    Object.entries(fields).map(([key, value]) => [key, String(value)])
  )

// --- Program adapter ---------------------------------------------------------

export type ProgramAdapterConfig<TInput = Fields, TOutput = Fields> = {
  addFormatFailureAsFeedback: boolean
  adapterRNG: RNG
  failureScore: number
  metric: GEPAMetric<TInput, TOutput>
  program: Program<TInput, TOutput>
  reflectionModel: ReflectionModel
  warnOnScoreMismatch: boolean
}

export type ProgramGEPAAdapter<TInput = Fields, TOutput = Fields> = GEPAAdapter<
  TInput,
  TOutput
> & {
  buildProgram: (candidate: Candidate) => Program<TInput, TOutput>
}

export const createProgramAdapter = <TInput, TOutput>(
  config: ProgramAdapterConfig<TInput, TOutput>
): ProgramGEPAAdapter<TInput, TOutput> => {
  const { adapterRNG, failureScore, metric, program, reflectionModel } = config
  let warnedScoreMismatch = false

  const proposeText =
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- `ReflectionModel` is a published `LanguageModel | (prompt) => Promise<string>` union; a callable is only distinguishable from a model object at runtime
    typeof reflectionModel === "function"
      ? reflectionModel
      : async (prompt: string) => {
          const { text } = await generateText({
            model: reflectionModel,
            prompt,
          })
          return text
        }

  // Descriptions only, exactly like upstream build_program — the clone keeps
  // whatever few-shot examples the student's steps already carry (e.g. from a
  // bootstrapFewShot pre-pass).
  const buildProgram = (candidate: Candidate): Program<TInput, TOutput> => {
    const built = program.clone()
    for (const step of built.steps) {
      const description = candidate[step.id]
      if (description !== undefined) {
        step.description = description
      }
    }
    return built
  }

  // Never throws per example: a failed rollout scores failureScore with a
  // null output. Only a build-time program error may abort.
  const evaluate = async (
    batch: Example<TInput, TOutput>[],
    candidate: Candidate,
    captureTraces: boolean
  ): Promise<EvaluationBatch<TInput, TOutput>> => {
    const built = buildProgram(candidate)
    const trajectories = await Promise.all(
      batch.map(async (example): Promise<Trajectory<TInput, TOutput>> => {
        const trace: GEPATraceStep[] = []
        let prediction: TOutput | null = null
        try {
          prediction = await built.run(example.inputData, {
            trace,
          })
        } catch (error) {
          console.warn(error)
        }
        let score = failureScore
        try {
          ;({ score } = await metric(example, prediction, trace))
        } catch (error) {
          console.warn(error)
        }
        return { example, prediction, score, trace }
      })
    )
    const batchResult: EvaluationBatch<TInput, TOutput> = {
      outputData: trajectories.map((t) => t.prediction),
      scores: trajectories.map((t) => t.score),
    }
    if (captureTraces) {
      batchResult.trajectories = trajectories
    }
    return batchResult
  }

  const recordForStep = async (
    trajectory: Trajectory<TInput, TOutput>,
    traceStep: GEPATraceStep,
    componentName: string
  ): Promise<ReflectiveExample> => {
    if (traceStep.parseFailure !== undefined) {
      const step = program.steps.find((s) => s.id === componentName)
      return {
        Feedback:
          PARSE_FAILURE_FEEDBACK_PREFIX + expectedStructure(step?.outputSchema),
        "Generated Outputs": PARSE_FAILURE_OUTPUT(traceStep.parseFailure),
        Inputs: stringifyFields(traceStep.inputData),
      }
    }
    const { feedback, score } = await runFeedbackMetric(
      metric,
      trajectory.example,
      trajectory.prediction,
      trajectory.trace,
      componentName,
      [traceStep]
    )
    // The step-level score is discarded in favor of the module-level
    // score; only the feedback text ever reaches the reflection LM.
    if (
      score !== trajectory.score &&
      config.warnOnScoreMismatch &&
      !warnedScoreMismatch
    ) {
      warnedScoreMismatch = true
      console.warn(
        "GEPA: step-level metric score differs from module-level score; using the module-level score."
      )
    }
    return {
      Feedback: feedback,
      "Generated Outputs": stringifyFields(traceStep.outputData),
      Inputs: stringifyFields(traceStep.inputData),
    }
  }

  /**
   * Pick the trace step to reflect on: the first parse failure if any remain,
   * a random step (adapter RNG) otherwise — unless the whole prediction
   * failed, which skips the example.
   */
  const chooseStep = (
    trajectory: Trajectory<TInput, TOutput>,
    componentName: string
  ): GEPATraceStep | null => {
    let steps = trajectory.trace.filter((step) => step.stepId === componentName)
    if (!config.addFormatFailureAsFeedback) {
      steps = steps.filter((step) => step.parseFailure === undefined)
    }
    if (steps.length === 0) {
      return null
    }
    const failure = steps.find((step) => step.parseFailure !== undefined)
    if (failure) {
      return failure
    }
    if (trajectory.prediction === null) {
      return null
    }
    return at(steps, Math.floor(adapterRNG() * steps.length), "trace steps")
  }

  const makeReflectiveDataset = async (
    _candidate: Candidate,
    evalBatch: EvaluationBatch<TInput, TOutput>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> => {
    const dataset: ReflectiveDataset = {}
    for (const componentName of componentsToUpdate) {
      const records: ReflectiveExample[] = []
      for (const trajectory of evalBatch.trajectories ?? []) {
        const chosen = chooseStep(trajectory, componentName)
        if (!chosen) {
          continue
        }
        // oxlint-disable-next-line no-await-in-loop -- adapter-RNG step choice must stay in trajectory order
        records.push(await recordForStep(trajectory, chosen, componentName))
      }
      if (records.length > 0) {
        dataset[componentName] = records
      }
    }
    return dataset
  }

  const proposeNewTexts = async (
    candidate: Candidate,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): Promise<Record<string, string>> => {
    const texts: Record<string, string> = {}
    // One reflection-LM call per component, sequential by design.
    for (const componentName of componentsToUpdate) {
      const examples = reflectiveDataset[componentName]
      if (!examples || examples.length === 0) {
        continue
      }
      const prompt = buildProposalPrompt(
        candidate[componentName] ?? "",
        renderSideInfo(examples)
      )
      // oxlint-disable-next-line no-await-in-loop -- sequential LM calls per component
      const response = await proposeText(prompt)
      // Python stores even an empty extraction — an empty-string instruction
      // becomes a real child; only an empty proposal dict skips the round.
      texts[componentName] = extractInstructionText(response)
    }
    return texts
  }

  return {
    buildProgram,
    evaluate,
    makeReflectiveDataset,
    proposeNewTexts,
  }
}
