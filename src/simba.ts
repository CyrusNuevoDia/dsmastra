import { generateObject } from "ai"
import type { LanguageModel } from "ai"
import { z } from "zod"

import { at, first, last } from "@/collections"
import type { Fields } from "@/fields"
import type { JSONData } from "@/json"
import { classifyJSON } from "@/json"
import type { MetricOutput } from "@/metric"
import type { RunContext, TraceStep } from "@/predictor"
import type { Example, Program } from "@/program"
import { createRNG, samplePoisson, shuffle, weightedChoice } from "@/random"
import { schemaProperties } from "@/schema"

export type { Example } from "@/program"

export type { MetricResult } from "@/metric"

export type Metric<TInput = Fields, TOutput = Fields> = (
  example: Example<TInput, TOutput>,
  prediction: TOutput | undefined
) => MetricOutput

export type SIMBAConfig<TInput = Fields, TOutput = Fields> = {
  bsize?: number
  demoInputFieldMaxlen?: number
  maxDemos?: number
  maxSteps?: number
  metric: Metric<TInput, TOutput>
  numCandidates?: number
  /** LM used to write rules; defaults to the first predictor's model. */
  promptModel?: LanguageModel
  seed?: number
  teacherSettings?: { model: LanguageModel; temperature?: number }
  temperatureForCandidates?: number
  temperatureForSampling?: number
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

export type SIMBAResult<TInput, TOutput> = {
  /** All finalist programs with their full-trainset scores, sorted descending. */
  candidates: { program: Program<TInput, TOutput>; score: number }[]
  program: Program<TInput, TOutput>
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
 * Group model-major rollouts into per-example buckets (stride = bsize), each
 * sorted by score descending, then order buckets by
 * (max−min gap, max score, max−avg gap) lexicographically descending.
 * Rollout records are shallow-copied so strategies never mutate shared state.
 */
export const makeBuckets = <TInput, TOutput>(
  rollouts: Rollout<TInput, TOutput>[],
  bsize: number
): Bucket<TInput, TOutput>[] => {
  const buckets: Bucket<TInput, TOutput>[] = []
  for (let exampleIdx = 0; exampleIdx < bsize; exampleIdx += 1) {
    const bucket: Rollout<TInput, TOutput>[] = []
    for (let i = exampleIdx; i < rollouts.length; i += bsize) {
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
  buckets.sort(
    (a, b) =>
      b.maxToMinGap - a.maxToMinGap ||
      b.maxScore - a.maxScore ||
      b.maxToAvgGap - a.maxToAvgGap
  )
  return buckets
}

// --- Demo dropping ----------------------------------------------------------

/**
 * max_demos enforced probabilistically: expected ~1 drop for a full predictor,
 * at least one forced at/over the cap. Draws are with replacement, so the
 * realized drop count can be lower than the sampled one. The single index set
 * applies to every predictor of the candidate.
 */
export const dropDemos = (
  candidate: Program<never, unknown>,
  maxDemos: number,
  rng: () => number,
  poissonRNG: () => number
): number => {
  const cap = maxDemos > 0 ? maxDemos : 3
  const numDemos = Math.max(
    0,
    ...candidate.predictors.map((p) => p.demos.length)
  )
  let toDrop = Math.max(
    samplePoisson(poissonRNG, numDemos / cap),
    numDemos >= cap ? 1 : 0
  )
  toDrop = Math.min(toDrop, numDemos)
  const dropIdxs = new Set<number>()
  for (let i = 0; i < toDrop; i += 1) {
    dropIdxs.add(Math.floor(rng() * numDemos))
  }
  for (const predictor of candidate.predictors) {
    predictor.demos = predictor.demos.filter((_, idx) => !dropIdxs.has(idx))
  }
  return toDrop
}

// --- Strategy A: append_a_demo ---------------------------------------------

export const appendADemo = <TInput, TOutput>(
  bucket: Bucket<TInput, TOutput>,
  candidate: Program<TInput, TOutput>,
  opts: { demoInputFieldMaxlen: number; p10: number }
): boolean => {
  const good = first(bucket.rollouts, "bucket rollouts")
  if (good.score <= opts.p10) {
    console.log(
      `Skipping appending a demo as good score ${good.score} is at or below the 10th percentile.`
    )
    return false
  }

  const nameToDemo = new Map<string, { inputs: Fields; outputs: Fields }>()
  for (const step of good.trace) {
    const inputs: Fields = { ...step.inputs }
    for (const [key, value] of Object.entries(inputs)) {
      const text = String(value)
      if (
        opts.demoInputFieldMaxlen &&
        text.length > opts.demoInputFieldMaxlen
      ) {
        inputs[key] =
          `${text.slice(0, opts.demoInputFieldMaxlen)}\n\t\t... <TRUNCATED FOR BREVITY>`
      }
    }
    // Keep only the last demo per predictor in the trajectory.
    nameToDemo.set(step.predictorName, { inputs, outputs: step.outputs })
  }

  let added = 0
  for (const [name, demo] of nameToDemo) {
    const predictor = candidate.predictors.find((p) => p.name === name)
    if (!predictor) {
      continue
    }
    predictor.demos.push({ augmented: true, ...demo })
    added += 1
  }
  console.log(`Added ${added} demos (one each) across all predictors.`)
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

/** Replace non-serializable values recursively, like dspy's recursive_mask. */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- a predictor output is an open value until `classifyJSON` sorts it on the next line; that call is the boundary parse, and `JSONData` is the named type it returns
export const recursiveMask = (value: unknown): JSONData => {
  const json = classifyJSON(value)
  switch (json.kind) {
    case "array": {
      return json.items.map(recursiveMask)
    }
    case "object": {
      return Object.fromEntries(
        json.entries.map(([k, v]) => [k, recursiveMask(v)])
      )
    }
    case "null": {
      return null
    }
    case "primitive":
    case "string": {
      return json.value
    }
    default: {
      return json.description
    }
  }
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters -- same boundary as `recursiveMask`: the field value is open until `classifyJSON` sorts it
const serializeField = (value: unknown): string => {
  const json = classifyJSON(value)
  return json.kind === "string"
    ? json.value
    : JSON.stringify(recursiveMask(value), null, 2)
}

const fieldDescriptionLines = (schema: z.ZodType): string =>
  Object.entries(schemaProperties(schema))
    .map(
      ([name, prop]) =>
        `${name} (${prop.type ?? "unknown"})${prop.description ? `: ${prop.description}` : ""}`
    )
    .join("\n")

const MODULE_SEPARATOR = "-".repeat(80)

const indentContinuations = (text: string): string =>
  ["", ...text.split("\n")].join("\n\t\t")

export const inspectModules = (program: Program<never, unknown>): string => {
  const blocks = [MODULE_SEPARATOR]
  for (const predictor of program.predictors) {
    blocks.push(
      `Module ${predictor.name}`,
      `\n\tInput Fields:${indentContinuations(fieldDescriptionLines(predictor.inputSchema))}`,
      `\tOutput Fields:${indentContinuations(fieldDescriptionLines(predictor.outputSchema))}`,
      `\tOriginal Instructions: ${indentContinuations(predictor.instructions)}`,
      MODULE_SEPARATOR
    )
  }
  return blocks.map((block) => block.replaceAll(/^\n+|\n+$/gu, "")).join("\n")
}

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
  const { object } = await generateObject({
    model: promptModel,
    prompt: [OFFER_FEEDBACK_INSTRUCTIONS, ...sections].join("\n\n"),
    schema: offerFeedbackSchema(moduleNames),
  })
  return object
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
  trace.map((step) => ({
    inputs: step.inputs,
    module_name: step.predictorName,
    outputs: step.outputs,
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
    candidate.predictors.map((p) => p.name),
    {
      better_program_outputs: goodView.prediction ?? {},
      better_program_trajectory: toTrajectory(goodView.trace),
      better_reward_info: goodView.outputMetadata,
      better_reward_value: goodView.score,
      module_names: candidate.predictors.map((p) => p.name),
      modules_defn: inspectModules(candidate),
      oracle_metadata: example.outputs,
      program_code: candidate.code,
      program_inputs: example.inputs,
      worse_program_outputs: badView.prediction ?? {},
      worse_program_trajectory: toTrajectory(badView.trace),
      worse_reward_info: badView.outputMetadata,
      worse_reward_value: badView.score,
    }
  )

  for (const predictor of candidate.predictors) {
    const advice = result.moduleAdvice[predictor.name]
    if (advice !== undefined) {
      console.log(`Advice for ${predictor.name}: ${advice}`)
      predictor.instructions = `${predictor.instructions}\n\n${advice}`
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
  let prediction: TOutput | undefined
  try {
    prediction = await program.run(example.inputs, {
      ...ctx,
      trace,
    })
  } catch (error) {
    console.warn(error)
  }

  let score = 0
  let outputMetadata: Fields = {}
  try {
    const { score: metricScore, ...metadata } = await metric(
      example,
      prediction
    )
    score = metricScore
    outputMetadata = metadata
  } catch (error) {
    console.warn(error)
  }

  return { example, outputMetadata, prediction, score, trace }
}

// --- Main loop --------------------------------------------------------------

export const simba = async <TInput extends Fields, TOutput extends Fields>(
  student: Program<TInput, TOutput>,
  trainset: Example<TInput, TOutput>[],
  config: SIMBAConfig<TInput, TOutput>
): Promise<SIMBAResult<TInput, TOutput>> => {
  const {
    bsize = 32,
    demoInputFieldMaxlen = 100_000,
    maxDemos = 4,
    maxSteps = 8,
    metric,
    numCandidates = 6,
    seed = 0,
    teacherSettings,
    temperatureForCandidates = 0.2,
    temperatureForSampling = 0.2,
  } = config

  if (trainset.length < bsize) {
    throw new Error(`Trainset too small: ${trainset.length} < ${bsize}`)
  }

  type AnyProgram = Program<TInput, TOutput>
  const baseline = student.clone()
  const promptModel =
    config.promptModel ?? first(baseline.predictors, "predictors").model

  const rng = createRNG(seed)
  const poissonRNG = createRNG(seed)

  // Index 0 is the baseline; its score list stays empty forever, pinning its
  // average at 0.0 — the exploration floor.
  const programs: AnyProgram[] = [baseline]
  const programScores: number[][] = [[]]
  const avgScores = () =>
    programScores.map((scores) => (scores.length > 0 ? mean(scores) : 0))

  const winningPrograms: AnyProgram[] = [baseline]
  const trialLogs: TrialLog[] = []

  let currentP10 = 0
  let currentP90 = 0

  const demoStrategy = (
    bucket: Bucket<TInput, TOutput>,
    candidate: AnyProgram
  ) =>
    Promise.resolve(
      appendADemo(bucket, candidate, { demoInputFieldMaxlen, p10: currentP10 })
    )
  const ruleStrategy = (
    bucket: Bucket<TInput, TOutput>,
    candidate: AnyProgram
  ) =>
    appendARule(bucket, candidate, {
      p10: currentP10,
      p90: currentP90,
      promptModel,
    })
  const strategies =
    maxDemos > 0 ? [demoStrategy, ruleStrategy] : [ruleStrategy]

  const dataIdxs = trainset.map((_, i) => i)
  shuffle(rng, dataIdxs)
  let instanceIdx = 0
  let nextRolloutId = 0

  const nextBatch = (): Example<TInput, TOutput>[] => {
    if (instanceIdx + bsize > trainset.length) {
      shuffle(rng, dataIdxs)
      instanceIdx = 0
    }
    const batch = dataIdxs
      .slice(instanceIdx, instanceIdx + bsize)
      .map((i) => at(trainset, i, "trainset"))
    instanceIdx += bsize
    return batch
  }

  // Rollout models always derive from the baseline program's LM; rollout_id is
  // a cache-buster mapped onto the seed parameter.
  const prepareModelsForResampling = (): RunContext[] => {
    const models: RunContext[] = []
    if (teacherSettings) {
      models.push({
        model: teacherSettings.model,
        seed: nextRolloutId,
        temperature: teacherSettings.temperature,
      })
      nextRolloutId += 1
    }
    while (models.length < numCandidates) {
      models.push({ seed: nextRolloutId, temperature: 1 })
      nextRolloutId += 1
    }
    return models
  }

  // Model-major, example-minor: bucket extraction strides by bsize.
  const sampleRollouts = (
    batch: Example<TInput, TOutput>[]
  ): Promise<Rollout<TInput, TOutput>[]> => {
    // The pool is frozen for the whole call, so score the programs once.
    const avg = avgScores()
    const topK = topKPlusBaseline(avg, numCandidates)
    const runs: (() => Promise<Rollout<TInput, TOutput>>)[] = []
    for (const modelCtx of prepareModelsForResampling()) {
      for (const example of batch) {
        const srcIdx = softmaxSample(rng, topK, avg, temperatureForSampling)
        // Rollouts never mutate the program, so no clone is needed.
        const rolloutProgram = at(programs, srcIdx, "programs")
        runs.push(() => runRollout(rolloutProgram, example, metric, modelCtx))
      }
    }
    return Promise.all(runs.map((run) => run()))
  }

  const generateCandidates = async (
    buckets: Bucket<TInput, TOutput>[]
  ): Promise<AnyProgram[]> => {
    // Candidates only join the pool after the step, so score it once.
    const avg = avgScores()
    const topK = topKPlusBaseline(avg, numCandidates)
    const candidates: AnyProgram[] = []
    for (const bucket of buckets) {
      const srcIdx = softmaxSample(rng, topK, avg, temperatureForCandidates)
      const candidate = at(programs, srcIdx, "programs").clone()
      dropDemos(candidate, maxDemos, rng, poissonRNG)
      const strategy = at(
        strategies,
        Math.floor(rng() * strategies.length),
        "strategies"
      )
      try {
        // A strategy no-op still keeps the candidate. Sequential on purpose:
        // the RNG stream must consume picks in bucket order.
        // oxlint-disable-next-line no-await-in-loop -- see above
        await strategy(bucket, candidate)
      } catch (error) {
        console.error(`Strategy failed with error: ${error}`)
        continue
      }
      candidates.push(candidate)
      if (candidates.length >= numCandidates + 1) {
        break
      }
    }
    return candidates
  }

  const evaluateOn = async (
    programsToScore: AnyProgram[],
    examples: Example<TInput, TOutput>[]
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

  const runStep = async (step: number): Promise<void> => {
    console.log(`Starting batch ${step + 1} of ${maxSteps}.`)
    const batch = nextBatch()

    console.log(
      `Sampling program trajectories on ${bsize} examples x ${numCandidates} samples.`
    )
    const rollouts = await sampleRollouts(batch)

    const allScores = rollouts.map((r) => r.score)
    currentP10 = percentile(allScores, 10)
    currentP90 = percentile(allScores, 90)
    const baselineScore = mean(allScores)
    console.log(
      `Batch ${step + 1}: Baseline mini-batch score: ${baselineScore}`
    )

    const buckets = makeBuckets(rollouts, bsize)
    const candidates = await generateCandidates(buckets)

    console.log(
      `Batch ${step + 1}: Evaluating ${candidates.length} programs on ${bsize} examples.`
    )
    const candidateScoreLists = await evaluateOn(candidates, batch)
    const candidateScores = candidateScoreLists.map(mean)
    console.log(
      `Scores after ${step + 1} batches: ${candidateScores}, Best: ${candidateScores.length ? Math.max(...candidateScores) : "N/A"}`
    )

    // Winner = argmax mean score, first max wins ties.
    if (candidateScores.length > 0) {
      const bestIdx = candidateScores.indexOf(Math.max(...candidateScores))
      winningPrograms.push(at(candidates, bestIdx, "candidates").clone())
    }

    // Register ALL candidates into the pool.
    for (const [idx, candidate] of candidates.entries()) {
      programs.push(candidate)
      programScores.push(at(candidateScoreLists, idx, "candidate scores"))
    }

    trialLogs.push({ baselineScore, candidateScores, step })
  }

  for (let step = 0; step < maxSteps; step += 1) {
    // oxlint-disable-next-line no-await-in-loop -- optimization steps are inherently sequential
    await runStep(step)
  }

  // Final selection: numCandidates+1 programs evenly spaced across the winner
  // timeline, always including the untouched student and the last winner.
  const m = winningPrograms.length - 1
  const n = numCandidates + 1
  const spacing =
    m < 1
      ? Array.from({ length: n }, () => 0)
      : Array.from({ length: n }, (_, i) => roundHalfEven((i * m) / (n - 1)))
  // Winners were already cloned at push time, so no extra copy is needed.
  const finalIdxs = [...new Set(spacing)]
  const finalists = finalIdxs.map((i) => at(winningPrograms, i, "winners"))

  console.log(
    `VALIDATION: Evaluating ${finalists.length} programs on the full trainset.`
  )
  const finalistScores = await evaluateOn(finalists, trainset)
  const finalScores = finalistScores.map(mean)

  const candidateData = finalists
    .map((program, idx) => ({
      program,
      score: at(finalScores, idx, "final scores"),
    }))
    .toSorted((a, b) => b.score - a.score)

  // First max wins ties — favors the less-evolved program.
  const bestIdx = finalScores.indexOf(Math.max(...finalScores))
  console.log(
    `Final trainset scores: ${finalScores}, Best: ${Math.max(...finalScores)} (at index ${bestIdx})`
  )

  return {
    candidates: candidateData,
    program: at(finalists, bestIdx, "finalists").clone(),
    trialLogs,
  }
}
