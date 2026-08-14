import { generateText } from "ai"
import type { LanguageModel } from "ai"

import { at } from "@/collections"
import type { Fields } from "@/fields"
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
import {
  expectedStructure,
  extractInstructionText,
  renderSideInfo,
  stringifyFields,
} from "@/prompting"
import type { MetricOutput, MetricResult } from "@/scorers"
import type { ScoreTarget } from "@/step"

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
  stepTrace?: GEPATraceStep[],
  /** The rollout's Mastra trace linkage, when the engine produced one. */
  target?: ScoreTarget
) => MetricOutput

export type ReflectionModel =
  | LanguageModel
  | ((prompt: string) => Promise<string>)

const defaultFeedback = (score: number) =>
  `This trajectory got a score of ${score}.`

/** The rollout's RunContext view: trace capture in, trace linkage out. */
type RolloutContext = {
  target?: ScoreTarget
  trace: GEPATraceStep[]
}

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
        const ctx: RolloutContext = { trace }
        let prediction: TOutput | null = null
        try {
          prediction = await built.run(example.inputData, ctx)
        } catch (error) {
          console.warn(error)
        }
        let score = failureScore
        try {
          // ctx.target was written by the engine runner during the rollout.
          ;({ score } = await metric(
            example,
            prediction,
            trace,
            undefined,
            undefined,
            ctx.target
          ))
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
