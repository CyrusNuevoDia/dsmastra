import { generateText, type LanguageModel } from "ai"
import { z } from "zod"
import type {
  Candidate,
  EvaluationBatch,
  GEPAAdapter,
  GEPATraceStep,
  ReflectiveDataset,
  ReflectiveExample,
  RNG,
  Trajectory,
} from "@/gepa/engine"
import type { Program } from "@/program"
import type { Example } from "@/simba"

export type ScoreWithFeedback = { feedback?: string; score: number }

/**
 * GEPA metric contract: called with (gold, prediction, trace) at module level
 * and with (gold, prediction, fullTrace, predName, predTrace) for
 * per-predictor feedback. Bare numbers get the default feedback string.
 */
export type GEPAMetric = (
  gold: Example,
  prediction: Record<string, unknown> | null,
  trace: GEPATraceStep[] | null,
  predName?: string,
  predTrace?: GEPATraceStep[]
) => number | Promise<number | ScoreWithFeedback> | ScoreWithFeedback

export type ReflectionLM = LanguageModel | ((prompt: string) => Promise<string>)

const defaultFeedback = (score: number) =>
  `This trajectory got a score of ${score}.`

export async function runFeedbackMetric(
  metric: GEPAMetric,
  gold: Example,
  prediction: Record<string, unknown> | null,
  trace: GEPATraceStep[] | null,
  predName?: string,
  predTrace?: GEPATraceStep[]
): Promise<Required<ScoreWithFeedback>> {
  const output = await metric(gold, prediction, trace, predName, predTrace)
  if (typeof output === "number") {
    return { feedback: defaultFeedback(output), score: output }
  }
  return {
    feedback: output.feedback ?? defaultFeedback(output.score),
    score: output.score,
  }
}

// --- Reflective-dataset rendering -------------------------------------------

const PARSE_FAILURE_OUTPUT = (raw: string) =>
  `Couldn't parse the output as per the expected output format. The model's raw response was:\n\`\`\`\n${raw}\n\`\`\`\n\n`

const PARSE_FAILURE_FEEDBACK_PREFIX =
  "Your output failed to parse. Follow this structure:\n"

function expectedStructure(outputSchema: z.ZodType | undefined): string {
  if (!outputSchema) {
    return ""
  }
  try {
    const jsonSchema = z.toJSONSchema(outputSchema) as {
      properties?: Record<string, { type?: string }>
    }
    return Object.entries(jsonSchema.properties ?? {})
      .map(([name, prop]) => `${name}: ${prop.type ?? "unknown"}`)
      .join("\n")
  } catch {
    return ""
  }
}

const MAX_HEADER_DEPTH = 6

function renderValue(value: unknown, depth: number): string[] {
  const header = "#".repeat(Math.min(depth, MAX_HEADER_DEPTH))
  if (Array.isArray(value)) {
    return value.flatMap((item, k) => [
      `${header} Item ${k + 1}`,
      ...renderValue(item, depth + 1),
    ])
  }
  if (value !== null && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(
      ([key, sub]) => [`${header} ${key}`, ...renderValue(sub, depth + 1)]
    )
  }
  return [String(value).trim()]
}

/** Markdown rendering of the reflective dataset for `<side_info>`. */
export function renderSideInfo(examples: ReflectiveExample[]): string {
  return examples
    .map((example, n) =>
      [
        `# Example ${n + 1}`,
        ...Object.entries(example).flatMap(([key, value]) => [
          `## ${key}`,
          ...renderValue(value, 3),
        ]),
      ].join("\n")
    )
    .join("\n\n")
}

// --- Instruction-proposal prompt (verbatim template) ------------------------

export function buildProposalPrompt(
  currentInstructions: string,
  sideInfo: string
): string {
  return `I provided an assistant with the following instructions to perform a task for me:
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
}

const LANGUAGE_TAG = /^\S*\n/

/**
 * Fenced-block extraction: text between the first and last fences with a
 * leading language tag stripped; fall back to stripping a lone leading or
 * trailing fence, else the whole trimmed response.
 */
export function extractInstructionText(response: string): string {
  const trimmed = response.trim()
  const first = trimmed.indexOf("```")
  const last = trimmed.lastIndexOf("```")
  if (first !== -1 && last > first) {
    return trimmed
      .slice(first + 3, last)
      .replace(LANGUAGE_TAG, "")
      .trim()
  }
  if (trimmed.startsWith("```")) {
    return trimmed.slice(3).replace(LANGUAGE_TAG, "").trim()
  }
  if (trimmed.endsWith("```")) {
    return trimmed.slice(0, -3).trim()
  }
  return trimmed
}

// --- Program adapter ---------------------------------------------------------

export type ProgramAdapterConfig = {
  addFormatFailureAsFeedback: boolean
  adapterRNG: RNG
  failureScore: number
  metric: GEPAMetric
  program: Program<never, unknown>
  reflectionLM: ReflectionLM
  warnOnScoreMismatch: boolean
}

export type ProgramGEPAAdapter = GEPAAdapter & {
  buildProgram: (candidate: Candidate) => Program<never, unknown>
}

export function createProgramAdapter(
  config: ProgramAdapterConfig
): ProgramGEPAAdapter {
  const { adapterRNG, failureScore, metric, program, reflectionLM } = config
  let warnedScoreMismatch = false

  const proposeText =
    typeof reflectionLM === "function"
      ? reflectionLM
      : async (prompt: string) => {
          const { text } = await generateText({ model: reflectionLM, prompt })
          return text
        }

  const buildProgram = (candidate: Candidate): Program<never, unknown> => {
    const built = program.clone()
    for (const predictor of built.predictors) {
      const instructions = candidate[predictor.name]
      if (instructions !== undefined) {
        predictor.instructions = instructions
      }
    }
    return built
  }

  // Never throws per example: a failed rollout scores failureScore with a
  // null output. Only a build-time program error may abort.
  const evaluate = async (
    batch: Example[],
    candidate: Candidate,
    captureTraces: boolean
  ): Promise<EvaluationBatch> => {
    const built = buildProgram(candidate)
    const trajectories = await Promise.all(
      batch.map(async (example): Promise<Trajectory> => {
        const trace: GEPATraceStep[] = []
        let prediction: Record<string, unknown> | null = null
        try {
          prediction = (await built.run(example.input as never, {
            trace,
          })) as Record<string, unknown>
        } catch (error) {
          console.warn(error)
        }
        let score = failureScore
        try {
          const output = await metric(example, prediction, trace)
          score = typeof output === "number" ? output : output.score
        } catch (error) {
          console.warn(error)
        }
        return { example, prediction, score, trace }
      })
    )
    return {
      outputs: trajectories.map((t) => t.prediction),
      scores: trajectories.map((t) => t.score),
      ...(captureTraces ? { trajectories } : {}),
    }
  }

  const stringifyFields = (
    fields: Record<string, unknown>
  ): Record<string, string> =>
    Object.fromEntries(
      Object.entries(fields).map(([key, value]) => [key, String(value)])
    )

  const recordForStep = async (
    trajectory: Trajectory,
    step: GEPATraceStep,
    componentName: string
  ): Promise<ReflectiveExample> => {
    if (step.parseFailure !== undefined) {
      const predictor = program.predictors.find((p) => p.name === componentName)
      return {
        Feedback:
          PARSE_FAILURE_FEEDBACK_PREFIX +
          expectedStructure(predictor?.outputSchema),
        "Generated Outputs": PARSE_FAILURE_OUTPUT(step.parseFailure),
        Inputs: stringifyFields(step.inputs),
      }
    }
    const { feedback, score } = await runFeedbackMetric(
      metric,
      trajectory.example,
      trajectory.prediction,
      trajectory.trace,
      componentName,
      [step]
    )
    // The predictor-level score is discarded in favor of the module-level
    // score; only the feedback text ever reaches the reflection LM.
    if (
      score !== trajectory.score &&
      config.warnOnScoreMismatch &&
      !warnedScoreMismatch
    ) {
      warnedScoreMismatch = true
      console.warn(
        "GEPA: predictor-level metric score differs from module-level score; using the module-level score."
      )
    }
    return {
      Feedback: feedback,
      "Generated Outputs": stringifyFields(step.outputs),
      Inputs: stringifyFields(step.inputs),
    }
  }

  /**
   * Pick the trace step to reflect on: the first parse failure if any remain,
   * a random step (adapter RNG) otherwise — unless the whole prediction
   * failed, which skips the example.
   */
  const chooseStep = (
    trajectory: Trajectory,
    componentName: string
  ): GEPATraceStep | null => {
    let steps = trajectory.trace.filter(
      (step) => step.predictorName === componentName
    )
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
    return steps[Math.floor(adapterRNG() * steps.length)] as GEPATraceStep
  }

  const makeReflectiveDataset = async (
    _candidate: Candidate,
    evalBatch: EvaluationBatch,
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
        // biome-ignore lint/performance/noAwaitInLoops: adapter-RNG step choice must stay in trajectory order
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
      // biome-ignore lint/performance/noAwaitInLoops: sequential LM calls per component
      const response = await proposeText(prompt)
      const text = extractInstructionText(response)
      if (text.length > 0) {
        texts[componentName] = text
      }
    }
    return texts
  }

  return { buildProgram, evaluate, makeReflectiveDataset, proposeNewTexts }
}
