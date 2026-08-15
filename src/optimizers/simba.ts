import type { AnyWorkflow } from "@mastra/core/workflows"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import { generateText, Output } from "ai"
import type { LanguageModel } from "ai"
import { z } from "zod"

import { at, first, last } from "../collections"
import type { Fields } from "../fields"
import {
  exampleSchema,
  fieldsSchema,
  loadPrompts,
  optimizerResultSchema,
  programFromPrompts,
  promptsOf,
  promptsSchema,
  workflowToProgram,
} from "../optimizers/utils"
import type {
  OptimizerCheckpoint,
  Prompts,
  SavePrompts,
} from "../optimizers/utils"
import type { Example, Program } from "../program"
import { inspectModules, serializeField } from "../prompting"
import {
  createRNG,
  restoreRNG,
  samplePoisson,
  shuffle,
  weightedChoice,
} from "../random"
import type { RNG } from "../random"
import { resolveScorer, scorerMetric } from "../scorers"
import type { Metric, ScorerRef } from "../scorers"
import type { RunContext, TraceStep } from "../step"

export type SIMBAConfig = {
  batchSize?: number
  /** Pause hook: called before every mini-batch; returning true suspends the
   * run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint
  candidates?: number
  candidateTemperature?: number
  maxFewShotExamples?: number
  maxFewShotInputLength?: number
  maxSteps?: number
  /** LM used to write rules; defaults to the first step's model. */
  promptModel?: LanguageModel
  samplingTemperature?: number
  savePrompts: SavePrompts
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Its `reason` rides along as reward info
   * for SIMBA's reflection. */
  scorer: ScorerRef
  seed?: number
  teacherSettings?: { model: LanguageModel; temperature?: number }
  trainingSet: readonly Example[]
}

type InternalConfig<TInput = Fields, TOutput = Fields> = {
  batchSize?: number
  candidates?: number
  candidateTemperature?: number
  maxFewShotExamples?: number
  maxFewShotInputLength?: number
  maxSteps?: number
  metric: Metric<TInput, TOutput>
  /** Called with each optimization step's winning program — checkpointing. */
  onImprovement?: (program: Program<TInput, TOutput>) => Promise<void>
  promptModel?: LanguageModel
  samplingTemperature?: number
  seed?: number
  teacherSettings?: { model: LanguageModel; temperature?: number }
}

export type Rollout<TInput = Fields, TOutput = Fields> = {
  example: Example<TInput, TOutput>
  outputMetadata: Fields
  prediction: TOutput | undefined
  score: number
  trace: TraceStep[]
}

export type Bucket<TInput = Fields, TOutput = Fields> = {
  maxScore: number
  maxToAvgGap: number
  maxToMinGap: number
  rollouts: Rollout<TInput, TOutput>[]
}

export type TrialLog = {
  baselineScore: number
  candidateScores: number[]
  step: number
}

export type SIMBAProgramResult<TInput, TOutput> = {
  /** All finalist programs with their full-trainingSet scores, sorted descending. */
  candidates: { program: Program<TInput, TOutput>; score: number }[]
  program: Program<TInput, TOutput>
  score: number
  trialLogs: TrialLog[]
}

const mean = (values: number[]): number =>
  values.reduce((acc, v) => acc + v, 0) / values.length

/** Python's round(): banker's rounding, used for final-selection spacing. */
export const roundHalfEven = (x: number): number => {
  const floor = Math.floor(x)
  const diff = x - floor
  if (diff > 0.5) {
    return floor + 1
  }
  if (diff < 0.5) {
    return floor
  }
  return floor % 2 === 0 ? floor : floor + 1
}

// --- Pool helpers -----------------------------------------------------------

/**
 * Sort program indices by average score descending (stable — ties break toward
 * the lower index), take the first k, force the baseline (0) into the last
 * slot if absent, then dedupe preserving order. May return fewer than k.
 */
export const topKPlusBaseline = (avgScores: number[], k: number): number[] => {
  const sorted = avgScores
    .map((avg, idx) => ({ avg, idx }))
    .toSorted((a, b) => b.avg - a.avg)
    .slice(0, k)
    .map((entry) => entry.idx)
  if (sorted.length > 0 && !sorted.includes(0)) {
    sorted[sorted.length - 1] = 0
  }
  return [...new Set(sorted)]
}

/**
 * Sample an index weighted by exp(avg/temperature); uniform fallback when the
 * weight sum is not positive. With all-zero scores this is uniform.
 */
export const softmaxSample = (
  rng: () => number,
  programIdxs: number[],
  avgScores: number[],
  temperature: number
): number => {
  if (programIdxs.length === 0) {
    throw new Error("No programs available for softmax sampling.")
  }
  return weightedChoice(
    rng,
    programIdxs,
    programIdxs.map((idx) => Math.exp((avgScores[idx] ?? 0) / temperature))
  )
}

/** NumPy-default linear-interpolation percentile. */
export const percentile = (values: number[], p: number): number => {
  const sorted = values.toSorted((a, b) => a - b)
  if (sorted.length === 0) {
    return 0
  }
  const rank = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(rank)
  const hi = Math.ceil(rank)
  const low = at(sorted, lo, "sorted scores")
  const high = at(sorted, hi, "sorted scores")
  return low + (high - low) * (rank - lo)
}

/**
 * Group model-major rollouts into per-example buckets (stride = batchSize),
 * each sorted by score descending, then order buckets by
 * (max−min gap, max score, max−avg gap) lexicographically descending.
 * Rollout records are shallow-copied so strategies never mutate shared state.
 */
export const makeBuckets = <TInput, TOutput>(
  rollouts: Rollout<TInput, TOutput>[],
  batchSize: number
): Bucket<TInput, TOutput>[] => {
  const buckets: Bucket<TInput, TOutput>[] = []
  for (let exampleIdx = 0; exampleIdx < batchSize; exampleIdx += 1) {
    const bucket: Rollout<TInput, TOutput>[] = []
    for (let i = exampleIdx; i < rollouts.length; i += batchSize) {
      bucket.push({ ...at(rollouts, i, "rollouts") })
    }
    bucket.sort((a, b) => b.score - a.score)
    const maxScore = first(bucket, "bucket").score
    const minScore = last(bucket, "bucket").score
    const avgScore = mean(bucket.map((r) => r.score))
    buckets.push({
      maxScore,
      maxToAvgGap: maxScore - avgScore,
      maxToMinGap: maxScore - minScore,
      rollouts: bucket,
    })
  }
  return buckets.toSorted(
    (a, b) =>
      b.maxToMinGap - a.maxToMinGap ||
      b.maxScore - a.maxScore ||
      b.maxToAvgGap - a.maxToAvgGap
  )
}

// --- Example dropping --------------------------------------------------------

/**
 * maxFewShotExamples enforced probabilistically: expected ~1 drop for a full
 * step, at least one forced at/over the cap. Draws are with replacement, so
 * the realized drop count can be lower than the sampled one. The single index
 * set applies to every step of the candidate.
 */
export const dropExamples = (
  candidate: Program<never, unknown>,
  maxFewShotExamples: number,
  rng: () => number,
  poissonRNG: () => number
): number => {
  const cap = maxFewShotExamples > 0 ? maxFewShotExamples : 3
  const examplesCount = Math.max(
    0,
    ...candidate.steps.map((step) => step.examples.length)
  )
  let toDrop = Math.max(
    samplePoisson(poissonRNG, examplesCount / cap),
    examplesCount >= cap ? 1 : 0
  )
  toDrop = Math.min(toDrop, examplesCount)
  const dropIdxs = new Set<number>()
  for (let i = 0; i < toDrop; i += 1) {
    dropIdxs.add(Math.floor(rng() * examplesCount))
  }
  for (const step of candidate.steps) {
    step.examples = step.examples.filter((_, idx) => !dropIdxs.has(idx))
  }
  return toDrop
}

// --- Strategy A: append_a_demo ---------------------------------------------

export const appendAnExample = <TInput, TOutput>(
  bucket: Bucket<TInput, TOutput>,
  candidate: Program<TInput, TOutput>,
  opts: { maxFewShotInputLength: number; p10: number }
): boolean => {
  const good = first(bucket.rollouts, "bucket rollouts")
  if (good.score <= opts.p10) {
    console.log(
      `Skipping appending an example as good score ${good.score} is at or below the 10th percentile.`
    )
    return false
  }

  const idToExample = new Map<string, Example>()
  for (const traceStep of good.trace) {
    const inputData: Fields = { ...traceStep.inputData }
    for (const [key, value] of Object.entries(inputData)) {
      const text = String(value)
      if (
        opts.maxFewShotInputLength &&
        text.length > opts.maxFewShotInputLength
      ) {
        inputData[key] =
          `${text.slice(0, opts.maxFewShotInputLength)}\n\t\t... <TRUNCATED FOR BREVITY>`
      }
    }
    // Keep only the last example per step in the trajectory.
    idToExample.set(traceStep.stepId, {
      inputData,
      outputData: traceStep.outputData,
    })
  }

  let added = 0
  for (const [stepId, example] of idToExample) {
    const step = candidate.steps.find((s) => s.id === stepId)
    if (!step) {
      continue
    }
    step.examples.push(example)
    added += 1
  }
  console.log(`Added ${added} examples (one each) across all steps.`)
  return true
}

// --- Strategy B: append_a_rule ---------------------------------------------

const OFFER_FEEDBACK_INSTRUCTIONS = `You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
similar inputs in the future.

The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

In your advice:
- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
- Ensure each unique module name appears exactly once as a key in the advice dictionary.`

const TRAJECTORY_DESCRIPTION =
  "The trajectory of the program's execution, showing each module's I/O"
const OUTPUTS_DESCRIPTION = "The outputs of the program that we are analyzing"
const REWARD_VALUE_DESCRIPTION =
  "The reward value assigned to the program's outputs"
const REWARD_INFO_DESCRIPTION =
  "Additional information that might be helpful to understanding the assigned reward value."

/** OfferFeedback input fields, in declaration order — order matters for prompt layout. */
const OFFER_FEEDBACK_INPUT_FIELDS: [string, string][] = [
  ["program_code", "The code of the program that we are analyzing"],
  [
    "modules_defn",
    "The definition of each module in the program, including its I/O",
  ],
  ["program_inputs", "The inputs to the program that we are analyzing"],
  [
    "oracle_metadata",
    "Any (hidden) metadata about the training set instance we're analyzing",
  ],
  ["worse_program_trajectory", TRAJECTORY_DESCRIPTION],
  ["worse_program_outputs", OUTPUTS_DESCRIPTION],
  ["worse_reward_value", REWARD_VALUE_DESCRIPTION],
  ["worse_reward_info", REWARD_INFO_DESCRIPTION],
  ["better_program_trajectory", TRAJECTORY_DESCRIPTION],
  ["better_program_outputs", OUTPUTS_DESCRIPTION],
  ["better_reward_value", REWARD_VALUE_DESCRIPTION],
  ["better_reward_info", REWARD_INFO_DESCRIPTION],
  [
    "module_names",
    "The names of the modules in the program, for which we seek advice",
  ],
]

const MODULE_ADVICE_DESCRIPTION =
  // oxlint-disable-next-line no-template-curly-in-string -- verbatim upstream field description
  "For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory."

// moduleAdvice is a closed object keyed by the exact module names: a
// z.record would compile to an open-ended JSON schema, which OpenAI's
// structured outputs reject — and the instructions demand each module name
// appear exactly once anyway.
const offerFeedbackSchema = (moduleNames: string[]) =>
  z.object({
    discussion: z
      .string()
      .describe("Discussing blame of where each module went wrong, if it did"),
    moduleAdvice: z
      .object(Object.fromEntries(moduleNames.map((name) => [name, z.string()])))
      .describe(MODULE_ADVICE_DESCRIPTION),
  })

export type OfferFeedbackResult = {
  discussion: string
  moduleAdvice: Record<string, string>
}

export const offerFeedback = async (
  promptModel: LanguageModel,
  moduleNames: string[],
  fields: Fields
): Promise<OfferFeedbackResult> => {
  const sections = OFFER_FEEDBACK_INPUT_FIELDS.map(
    ([name, description]) =>
      `[[ ## ${name} ## ]]\n${description}\n\n${serializeField(fields[name])}`
  )
  const { output } = await generateText({
    model: promptModel,
    output: Output.object({ schema: offerFeedbackSchema(moduleNames) }),
    prompt: [OFFER_FEEDBACK_INSTRUCTIONS, ...sections].join("\n\n"),
  })
  return output
}

/**
 * One side of the good/bad contrast shown to the reflection LM. Either half may
 * be blanked out when the two rollouts didn't actually differ, which is why the
 * score can read "N/A" and the prediction can be a placeholder record.
 */
type RolloutContrast = {
  outputMetadata: Fields
  prediction: Fields | undefined
  score: number | "N/A"
  trace: TraceStep[]
}

const toTrajectory = (trace: TraceStep[]) =>
  // Serialized keys stay `inputs`/`outputs` for byte-parity with the upstream
  // prompt rendering.
  trace.map((traceStep) => ({
    inputs: traceStep.inputData,
    module_name: traceStep.stepId,
    outputs: traceStep.outputData,
  }))

/** Stand-in shown when a rollout carries no usable contrast. */
const BLANKED_CONTRAST = {
  prediction: { "N/A": "Prediction not available" },
  score: "N/A",
  trace: [],
} as const satisfies Partial<RolloutContrast>

export const appendARule = async <TInput, TOutput extends Fields>(
  bucket: Bucket<TInput, TOutput>,
  candidate: Program<TInput, TOutput>,
  opts: {
    p10: number
    p90: number
    promptModel: LanguageModel
  }
): Promise<boolean> => {
  const good = first(bucket.rollouts, "bucket rollouts")
  const bad = last(bucket.rollouts, "bucket rollouts")

  if (good.score <= opts.p10 || bad.score >= opts.p90) {
    console.log(
      `Skipping rule generation as good score ${good.score} is at or below the 10th percentile ` +
        `*or* bad score ${bad.score} is at or above the 90th percentile.`
    )
    return false
  }

  // Blank the uninformative half when there's no real contrast. Local views
  // only — never mutate the shared rollout records.
  let goodView: RolloutContrast = {
    outputMetadata: good.outputMetadata,
    prediction: good.prediction,
    score: good.score,
    trace: good.trace,
  }
  let badView: RolloutContrast = {
    outputMetadata: bad.outputMetadata,
    prediction: bad.prediction,
    score: bad.score,
    trace: bad.trace,
  }
  if (good.score <= bad.score) {
    if (good.score > opts.p90) {
      badView = { ...badView, ...BLANKED_CONTRAST }
    } else {
      goodView = { ...goodView, ...BLANKED_CONTRAST }
    }
  }

  const { example } = good
  const result = await offerFeedback(
    opts.promptModel,
    candidate.steps.map((step) => step.id),
    {
      better_program_outputs: goodView.prediction ?? {},
      better_program_trajectory: toTrajectory(goodView.trace),
      better_reward_info: goodView.outputMetadata,
      better_reward_value: goodView.score,
      module_names: candidate.steps.map((step) => step.id),
      modules_defn: inspectModules(candidate),
      oracle_metadata: example.outputData,
      program_code: candidate.code,
      program_inputs: example.inputData,
      worse_program_outputs: badView.prediction ?? {},
      worse_program_trajectory: toTrajectory(badView.trace),
      worse_reward_info: badView.outputMetadata,
      worse_reward_value: badView.score,
    }
  )

  for (const step of candidate.steps) {
    const advice = result.moduleAdvice[step.id]
    if (advice !== undefined) {
      console.log(`Advice for ${step.id}: ${advice}`)
      step.description = `${step.description}\n\n${advice}`
    }
  }
  return true
}

// --- Rollouts ---------------------------------------------------------------

const runRollout = async <TInput, TOutput>(
  program: Program<TInput, TOutput>,
  example: Example<TInput, TOutput>,
  metric: Metric<TInput, TOutput>,
  ctx: RunContext
): Promise<Rollout<TInput, TOutput>> => {
  const trace: TraceStep[] = []
  const runCtx: RunContext = { ...ctx, trace }
  let prediction: TOutput | undefined
  try {
    prediction = await program.run(example.inputData, runCtx)
  } catch (error) {
    console.warn(error)
  }

  let score = 0
  let outputMetadata: Fields = {}
  try {
    // runCtx.target was written by the engine runner during the rollout.
    const { score: metricScore, ...metadata } = await metric(
      example,
      prediction,
      runCtx.target
    )
    score = metricScore
    outputMetadata = metadata
  } catch (error) {
    console.warn(error)
  }

  return { example, outputMetadata, prediction, score, trace }
}

// --- Batch phases -----------------------------------------------------------
//
// One SIMBA batch decomposes into three phases — rollout sampling, candidate
// generation (the LM-reflection phase), and candidate scoring — shared by the
// in-memory driver (simbaProgram) and the durable workflow driver
// (createSIMBAWorkflow). RNG streams thread through explicitly so the durable
// driver can checkpoint them between phases.

/** The fixed per-run knobs both drivers thread through the phases. */
type SIMBARuntime<TInput, TOutput> = {
  batchSize: number
  candidates: number
  candidateTemperature: number
  maxFewShotExamples: number
  maxFewShotInputLength: number
  metric: Metric<TInput, TOutput>
  promptModel: LanguageModel
  samplingTemperature: number
  teacherSettings?: { model: LanguageModel; temperature?: number }
}

/** The mini-batch cursor: a shuffled index order and a position within it. */
export type SIMBACursor = {
  dataIdxs: number[]
  instanceIdx: number
}

const nextBatch = <TInput, TOutput>(
  rng: RNG,
  cursor: SIMBACursor,
  trainingSet: readonly Example<TInput, TOutput>[],
  batchSize: number
): Example<TInput, TOutput>[] => {
  if (cursor.instanceIdx + batchSize > trainingSet.length) {
    shuffle(rng, cursor.dataIdxs)
    cursor.instanceIdx = 0
  }
  const batch = cursor.dataIdxs
    .slice(cursor.instanceIdx, cursor.instanceIdx + batchSize)
    .map((i) => at([...trainingSet], i, "trainingSet"))
  cursor.instanceIdx += batchSize
  return batch
}

// Rollout models always derive from the baseline program's LM; rollout_id is
// a cache-buster mapped onto the seed parameter.
const prepareModelsForResampling = <TInput, TOutput>(
  rt: SIMBARuntime<TInput, TOutput>,
  nextRolloutId: number
) => {
  const models: RunContext[] = []
  let id = nextRolloutId
  if (rt.teacherSettings) {
    models.push({
      model: rt.teacherSettings.model,
      seed: id,
      temperature: rt.teacherSettings.temperature,
    })
    id += 1
  }
  while (models.length < rt.candidates) {
    models.push({ seed: id, temperature: 1 })
    id += 1
  }
  return { models, nextRolloutId: id }
}

// Model-major, example-minor: bucket extraction strides by batchSize.
const sampleBatchRollouts = <TInput, TOutput>(
  rt: SIMBARuntime<TInput, TOutput>,
  rng: RNG,
  programs: Program<TInput, TOutput>[],
  avg: number[],
  batch: Example<TInput, TOutput>[],
  models: RunContext[]
): Promise<Rollout<TInput, TOutput>[]> => {
  // The pool is frozen for the whole call, so the caller scored it once.
  const topK = topKPlusBaseline(avg, rt.candidates)
  const runs: (() => Promise<Rollout<TInput, TOutput>>)[] = []
  for (const modelCtx of models) {
    for (const example of batch) {
      const srcIdx = softmaxSample(rng, topK, avg, rt.samplingTemperature)
      // Rollouts never mutate the program, so no clone is needed.
      const rolloutProgram = at(programs, srcIdx, "programs")
      runs.push(() => runRollout(rolloutProgram, example, rt.metric, modelCtx))
    }
  }
  return Promise.all(runs.map((run) => run()))
}

const generateCandidatesFromBuckets = async <TInput, TOutput extends Fields>(
  rt: SIMBARuntime<TInput, TOutput>,
  rng: RNG,
  poissonRNG: RNG,
  programs: Program<TInput, TOutput>[],
  avg: number[],
  buckets: Bucket<TInput, TOutput>[],
  percentiles: { p10: number; p90: number }
): Promise<Program<TInput, TOutput>[]> => {
  // Candidates only join the pool after the step, so the caller scored it once.
  const topK = topKPlusBaseline(avg, rt.candidates)
  const strategyCount = rt.maxFewShotExamples > 0 ? 2 : 1
  const generated: Program<TInput, TOutput>[] = []
  for (const bucket of buckets) {
    const srcIdx = softmaxSample(rng, topK, avg, rt.candidateTemperature)
    const candidate = at(programs, srcIdx, "programs").clone()
    dropExamples(candidate, rt.maxFewShotExamples, rng, poissonRNG)
    const strategyIdx = Math.floor(rng() * strategyCount)
    // With maxFewShotExamples > 0 the strategies are [example, rule]; without,
    // [rule] alone — matching the strategy list upstream.
    const useExampleStrategy = rt.maxFewShotExamples > 0 && strategyIdx === 0
    try {
      // A strategy no-op still keeps the candidate. Sequential on purpose:
      // the RNG stream must consume picks in bucket order.
      if (useExampleStrategy) {
        appendAnExample(bucket, candidate, {
          maxFewShotInputLength: rt.maxFewShotInputLength,
          p10: percentiles.p10,
        })
      } else {
        // oxlint-disable-next-line no-await-in-loop -- see above
        await appendARule(bucket, candidate, {
          p10: percentiles.p10,
          p90: percentiles.p90,
          promptModel: rt.promptModel,
        })
      }
    } catch (error) {
      console.error(`Strategy failed with error: ${error}`)
      continue
    }
    generated.push(candidate)
    if (generated.length >= rt.candidates + 1) {
      break
    }
  }
  return generated
}

const evaluateOn = async <TInput, TOutput>(
  programsToScore: Program<TInput, TOutput>[],
  examples: Example<TInput, TOutput>[],
  metric: Metric<TInput, TOutput>
): Promise<number[][]> => {
  const rollouts = await Promise.all(
    programsToScore.flatMap((program) =>
      examples.map((example) => runRollout(program, example, metric, {}))
    )
  )
  return programsToScore.map((_, idx) =>
    rollouts
      .slice(idx * examples.length, (idx + 1) * examples.length)
      .map((rollout) => rollout.score)
  )
}

/** Final selection: candidates+1 programs evenly spaced across the winner
 * timeline, always including the untouched student and the last winner. */
const finalistIdxs = (winnersCount: number, candidates: number): number[] => {
  const m = winnersCount - 1
  const n = candidates + 1
  const spacing =
    m < 1
      ? Array.from({ length: n }, () => 0)
      : Array.from({ length: n }, (_, i) => roundHalfEven((i * m) / (n - 1)))
  return [...new Set(spacing)]
}

const avgOf = (scoreLists: number[][]): number[] =>
  scoreLists.map((scores) => (scores.length > 0 ? mean(scores) : 0))

// --- Main loop --------------------------------------------------------------

export const simbaProgram = async <
  TInput extends Fields,
  TOutput extends Fields,
>(
  student: Program<TInput, TOutput>,
  trainingSet: Example<TInput, TOutput>[],
  config: InternalConfig<TInput, TOutput>
): Promise<SIMBAProgramResult<TInput, TOutput>> => {
  const {
    batchSize = 32,
    candidates = 6,
    maxSteps = 8,
    metric,
    onImprovement,
    seed = 0,
  } = config

  if (trainingSet.length < batchSize) {
    throw new Error(
      `TrainingSet too small: ${trainingSet.length} < ${batchSize}`
    )
  }

  type AnyProgram = Program<TInput, TOutput>
  const baseline = student.clone()
  const rt: SIMBARuntime<TInput, TOutput> = {
    batchSize,
    candidateTemperature: config.candidateTemperature ?? 0.2,
    candidates,
    maxFewShotExamples: config.maxFewShotExamples ?? 4,
    maxFewShotInputLength: config.maxFewShotInputLength ?? 100_000,
    metric,
    promptModel: config.promptModel ?? first(baseline.steps, "steps").model,
    samplingTemperature: config.samplingTemperature ?? 0.2,
    teacherSettings: config.teacherSettings,
  }

  const rng = createRNG(seed)
  const poissonRNG = createRNG(seed)

  // Index 0 is the baseline; its score list stays empty forever, pinning its
  // average at 0.0 — the exploration floor.
  const programs: AnyProgram[] = [baseline]
  const programScores: number[][] = [[]]

  const winningPrograms: AnyProgram[] = [baseline]
  const trialLogs: TrialLog[] = []

  const cursor: SIMBACursor = {
    dataIdxs: trainingSet.map((_, i) => i),
    instanceIdx: 0,
  }
  shuffle(rng, cursor.dataIdxs)
  let nextRolloutId = 0

  const runStep = async (step: number): Promise<void> => {
    console.log(`Starting batch ${step + 1} of ${maxSteps}.`)
    const batch = nextBatch(rng, cursor, trainingSet, batchSize)

    console.log(
      `Sampling program trajectories on ${batchSize} examples x ${candidates} samples.`
    )
    const avg = avgOf(programScores)
    const resampling = prepareModelsForResampling(rt, nextRolloutId)
    ;({ nextRolloutId } = resampling)
    const rollouts = await sampleBatchRollouts(
      rt,
      rng,
      programs,
      avg,
      batch,
      resampling.models
    )

    const allScores = rollouts.map((r) => r.score)
    const percentiles = {
      p10: percentile(allScores, 10),
      p90: percentile(allScores, 90),
    }
    const baselineScore = mean(allScores)
    console.log(
      `Batch ${step + 1}: Baseline mini-batch score: ${baselineScore}`
    )

    const buckets = makeBuckets(rollouts, batchSize)
    const stepCandidates = await generateCandidatesFromBuckets(
      rt,
      rng,
      poissonRNG,
      programs,
      avgOf(programScores),
      buckets,
      percentiles
    )

    console.log(
      `Batch ${step + 1}: Evaluating ${stepCandidates.length} programs on ${batchSize} examples.`
    )
    const candidateScoreLists = await evaluateOn(stepCandidates, batch, metric)
    const candidateScores = candidateScoreLists.map(mean)
    console.log(
      `Scores after ${step + 1} batches: ${candidateScores}, Best: ${candidateScores.length ? Math.max(...candidateScores) : "N/A"}`
    )

    // Winner = argmax mean score, first max wins ties.
    if (candidateScores.length > 0) {
      const bestIdx = candidateScores.indexOf(Math.max(...candidateScores))
      const winner = at(stepCandidates, bestIdx, "candidates").clone()
      winningPrograms.push(winner)
      await onImprovement?.(winner)
    }

    // Register ALL candidates into the pool.
    for (const [idx, candidate] of stepCandidates.entries()) {
      programs.push(candidate)
      programScores.push(at(candidateScoreLists, idx, "candidate scores"))
    }

    trialLogs.push({ baselineScore, candidateScores, step })
  }

  for (let step = 0; step < maxSteps; step += 1) {
    // oxlint-disable-next-line no-await-in-loop -- optimization steps are inherently sequential
    await runStep(step)
  }

  // Winners were already cloned at push time, so no extra copy is needed.
  const finalists = finalistIdxs(winningPrograms.length, candidates).map((i) =>
    at(winningPrograms, i, "winners")
  )

  console.log(
    `VALIDATION: Evaluating ${finalists.length} programs on the full trainingSet.`
  )
  const finalistScores = await evaluateOn(finalists, trainingSet, metric)
  const finalScores = finalistScores.map(mean)

  const candidateData = finalists
    .map((program, idx) => ({
      program,
      score: at(finalScores, idx, "final scores"),
    }))
    .toSorted((a, b) => b.score - a.score)

  // First max wins ties — favors the less-evolved program.
  const bestIdx = finalScores.indexOf(Math.max(...finalScores))
  const bestScore = Math.max(...finalScores)
  console.log(
    `Final trainingSet scores: ${finalScores}, Best: ${bestScore} (at index ${bestIdx})`
  )

  return {
    candidates: candidateData,
    program: at(finalists, bestIdx, "finalists").clone(),
    score: bestScore,
    trialLogs,
  }
}

// --- Durable workflow driver -------------------------------------------------

const trialLogSchema = z.object({
  baselineScore: z.number(),
  candidateScores: z.array(z.number()),
  step: z.number(),
})

/** The loop's whole world between batches, as JSON. */
const simbaStateSchema = z.object({
  cursor: z.object({
    dataIdxs: z.array(z.number()),
    instanceIdx: z.number(),
  }),
  nextRolloutId: z.number(),
  pool: z.array(promptsSchema),
  poolScores: z.array(z.array(z.number())),
  rng: z.object({ main: z.number(), poisson: z.number() }),
  step: z.number(),
  trialLogs: z.array(trialLogSchema),
  winners: z.array(promptsSchema),
})

type SIMBAState = z.infer<typeof simbaStateSchema>

const traceStepSchema = z.object({
  inputData: fieldsSchema,
  outputData: fieldsSchema,
  stepId: z.string(),
})

const rolloutSchema = z.object({
  example: exampleSchema,
  outputMetadata: fieldsSchema,
  prediction: fieldsSchema.optional(),
  score: z.number(),
  trace: z.array(traceStepSchema),
})

const bucketSchema = z.object({
  maxScore: z.number(),
  maxToAvgGap: z.number(),
  maxToMinGap: z.number(),
  rollouts: z.array(rolloutSchema),
})

const rolledOutSchema = simbaStateSchema.extend({
  baselineScore: z.number(),
  batch: z.array(exampleSchema),
  buckets: z.array(bucketSchema),
  p10: z.number(),
  p90: z.number(),
})

const proposedSchema = simbaStateSchema.extend({
  baselineScore: z.number(),
  batch: z.array(exampleSchema),
  candidatePrompts: z.array(promptsSchema),
})

/**
 * SIMBA (Stochastic Introspective Mini-Batch Ascent) as a Mastra workflow
 * over the target `workflow`: each durable loop iteration is one mini-batch
 * step, split into a `rollout` step (trajectory sampling through the engine),
 * a `propose-candidates` step (the introspective phase — appendARule's
 * offerFeedback LM calls live here), and a `score-candidates` step (winner
 * selection, pool registration, savePrompts checkpointing). The candidate
 * pool, winner timeline, batch cursor, and both RNG streams cross step
 * boundaries as JSON, so a storage-backed run resumes mid-optimization
 * without redoing completed batches.
 */
export const createSIMBAWorkflow = (
  workflow: AnyWorkflow,
  config: SIMBAConfig
) => {
  const { checkpoint, savePrompts, trainingSet } = config
  const batchSize = config.batchSize ?? Math.min(32, trainingSet.length)
  const candidates = config.candidates ?? 6
  const maxSteps = config.maxSteps ?? 8
  const seed = config.seed ?? 0
  if (trainingSet.length < batchSize) {
    throw new Error(
      `TrainingSet too small: ${trainingSet.length} < ${batchSize}`
    )
  }
  const metric = scorerMetric(resolveScorer(workflow, config.scorer))
  const base = () => workflowToProgram(workflow)
  const rt: SIMBARuntime<Fields, Fields> = {
    batchSize,
    candidateTemperature: config.candidateTemperature ?? 0.2,
    candidates,
    maxFewShotExamples: config.maxFewShotExamples ?? 4,
    maxFewShotInputLength: config.maxFewShotInputLength ?? 100_000,
    metric,
    promptModel: config.promptModel ?? first(base().steps, "steps").model,
    samplingTemperature: config.samplingTemperature ?? 0.2,
    teacherSettings: config.teacherSettings,
  }
  const examples = [...trainingSet]

  const init = createStep({
    description: "Seed the pool with the baseline and shuffle the batch order",
    execute: () => {
      const rng = createRNG(seed)
      const poissonRNG = createRNG(seed)
      const baselinePrompts = promptsOf(base())
      const dataIdxs = examples.map((_, i) => i)
      shuffle(rng, dataIdxs)
      return Promise.resolve({
        cursor: { dataIdxs, instanceIdx: 0 },
        nextRolloutId: 0,
        // Index 0 is the baseline; its score list stays empty forever,
        // pinning its average at 0.0 — the exploration floor.
        pool: [baselinePrompts],
        poolScores: [[]],
        rng: { main: rng.state, poisson: poissonRNG.state },
        step: 0,
        trialLogs: [],
        winners: [baselinePrompts],
      } satisfies SIMBAState)
    },
    id: "init",
    inputSchema: z.object({}),
    outputSchema: simbaStateSchema,
  })

  const rollout = createStep({
    description: "Sample program trajectories over the next mini-batch",
    execute: async ({ inputData, resumeData, suspend }) => {
      const state: SIMBAState = inputData
      if (!resumeData && (await checkpoint?.({ iteration: state.step }))) {
        return await suspend({ iteration: state.step })
      }
      const rng = restoreRNG(state.rng.main)
      const cursor = structuredClone(state.cursor)
      const programs = state.pool.map((prompts) =>
        programFromPrompts(base(), prompts)
      )
      console.log(`Starting batch ${state.step + 1} of ${maxSteps}.`)
      const batch = nextBatch(rng, cursor, examples, batchSize)
      console.log(
        `Sampling program trajectories on ${batchSize} examples x ${candidates} samples.`
      )
      const resampling = prepareModelsForResampling(rt, state.nextRolloutId)
      const rollouts = await sampleBatchRollouts(
        rt,
        rng,
        programs,
        avgOf(state.poolScores),
        batch,
        resampling.models
      )
      const allScores = rollouts.map((r) => r.score)
      const baselineScore = mean(allScores)
      console.log(
        `Batch ${state.step + 1}: Baseline mini-batch score: ${baselineScore}`
      )
      return {
        ...state,
        baselineScore,
        batch,
        buckets: makeBuckets(rollouts, batchSize),
        cursor,
        nextRolloutId: resampling.nextRolloutId,
        p10: percentile(allScores, 10),
        p90: percentile(allScores, 90),
        rng: { ...state.rng, main: rng.state },
      }
    },
    id: "rollout",
    inputSchema: simbaStateSchema,
    outputSchema: rolledOutSchema,
    resumeSchema: z.object({}),
    suspendSchema: z.object({ iteration: z.number() }),
  })

  const propose = createStep({
    description:
      "Generate candidates from the buckets (introspective offerFeedback LM calls)",
    execute: async ({ inputData }) => {
      const { buckets, p10, p90, ...state } = inputData
      const rng = restoreRNG(state.rng.main)
      const poissonRNG = restoreRNG(state.rng.poisson)
      const programs = state.pool.map((prompts) =>
        programFromPrompts(base(), prompts)
      )
      const generated = await generateCandidatesFromBuckets(
        rt,
        rng,
        poissonRNG,
        programs,
        avgOf(state.poolScores),
        // SAFETY: the schema's optional `prediction` is the JSON face of
        // Rollout's required `TOutput | undefined` — identical runtime shape,
        // deserialized exactly as the rollout step serialized it.
        buckets as Bucket<Fields, Fields>[],
        { p10, p90 }
      )
      return {
        ...state,
        candidatePrompts: generated.map((candidate) => promptsOf(candidate)),
        rng: { main: rng.state, poisson: poissonRNG.state },
      }
    },
    id: "propose-candidates",
    inputSchema: rolledOutSchema,
    outputSchema: proposedSchema,
  })

  const score = createStep({
    description: "Score the candidates, register them, persist the winner",
    execute: async ({ inputData }) => {
      const { baselineScore, batch, candidatePrompts, ...state } = inputData
      const stepCandidates = candidatePrompts.map((prompts) =>
        programFromPrompts(base(), prompts)
      )
      console.log(
        `Batch ${state.step + 1}: Evaluating ${stepCandidates.length} programs on ${batchSize} examples.`
      )
      const candidateScoreLists = await evaluateOn(
        stepCandidates,
        batch,
        metric
      )
      const candidateScores = candidateScoreLists.map(mean)
      console.log(
        `Scores after ${state.step + 1} batches: ${candidateScores}, Best: ${candidateScores.length ? Math.max(...candidateScores) : "N/A"}`
      )
      const winners = [...state.winners]
      if (candidateScores.length > 0) {
        // Winner = argmax mean score, first max wins ties.
        const bestIdx = candidateScores.indexOf(Math.max(...candidateScores))
        const winner = at(candidatePrompts, bestIdx, "candidates")
        winners.push(winner)
        await savePrompts(winner)
      }
      return {
        ...state,
        // Register ALL candidates into the pool.
        pool: [...state.pool, ...candidatePrompts],
        poolScores: [...state.poolScores, ...candidateScoreLists],
        step: state.step + 1,
        trialLogs: [
          ...state.trialLogs,
          { baselineScore, candidateScores, step: state.step },
        ],
        winners,
      }
    },
    id: "score-candidates",
    inputSchema: proposedSchema,
    outputSchema: simbaStateSchema,
  })

  /* oxlint-disable promise/prefer-await-to-then -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  const iteration = createWorkflow({
    id: "iteration",
    inputSchema: simbaStateSchema,
    outputSchema: simbaStateSchema,
  })
    .then(rollout)
    .then(propose)
    .then(score)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then */

  const finalize = createStep({
    description: "Evaluate the finalists, persist and land the winner",
    execute: async ({ inputData }) => {
      const state: SIMBAState = inputData
      // Winner snapshots were taken at push time, so no extra copy is needed.
      const finalists = finalistIdxs(state.winners.length, candidates).map(
        (i) => at(state.winners, i, "winners")
      )
      console.log(
        `VALIDATION: Evaluating ${finalists.length} programs on the full trainingSet.`
      )
      const finalistScores = await evaluateOn(
        finalists.map((prompts) => programFromPrompts(base(), prompts)),
        examples,
        metric
      )
      const finalScores = finalistScores.map(mean)
      const candidateData = finalists
        .map((prompts, idx) => ({
          prompts,
          score: at(finalScores, idx, "final scores"),
        }))
        .toSorted((a, b) => b.score - a.score)
      // First max wins ties — favors the less-evolved program.
      const bestIdx = finalScores.indexOf(Math.max(...finalScores))
      const bestScore = Math.max(...finalScores)
      console.log(
        `Final trainingSet scores: ${finalScores}, Best: ${bestScore} (at index ${bestIdx})`
      )
      const best = at(finalists, bestIdx, "finalists")
      await savePrompts(best)
      // The winner's prompt state lands in place on the caller's workflow.
      loadPrompts(workflow, best)
      return {
        // Every finalist as a JSON-safe snapshot paired with its
        // full-trainingSet score, best first.
        candidates: candidateData.map(({ prompts, score: finalistScore }) => {
          const pair: [Prompts, { score: number }] = [
            prompts,
            { score: finalistScore },
          ]
          return pair
        }),
        score: bestScore,
      }
    },
    id: "finalize",
    inputSchema: simbaStateSchema,
    outputSchema: optimizerResultSchema,
  })

  /* oxlint-disable promise/prefer-await-to-then, promise/no-return-wrap -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  return createWorkflow({
    id: `${workflow.id}.simba`,
    inputSchema: z.object({}),
    outputSchema: optimizerResultSchema,
  })
    .then(init)
    .dountil(iteration, ({ inputData }) =>
      Promise.resolve(inputData.step >= maxSteps)
    )
    .then(finalize)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then, promise/no-return-wrap */
}
