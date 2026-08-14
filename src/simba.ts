import { generateObject, type LanguageModel } from "ai"
import { z } from "zod"
import {
  type AnyPredictor,
  type RunContext,
  schemaProperties,
  type TraceStep,
} from "@/predictor"
import type { Example, Program } from "@/program"
import { createRNG, samplePoisson, shuffle, weightedChoice } from "@/random"

export type { Example } from "@/program"

export type MetricResult = number | { score: number; [key: string]: unknown }

export type Metric = (
  example: Example,
  prediction: Record<string, unknown> | undefined
) => MetricResult | Promise<MetricResult>

export type SIMBAConfig = {
  bsize?: number
  demoInputFieldMaxlen?: number
  maxDemos?: number
  maxSteps?: number
  metric: Metric
  numCandidates?: number
  /** LM used to write rules; defaults to the first predictor's model. */
  promptModel?: LanguageModel
  seed?: number
  teacherSettings?: { model: LanguageModel; temperature?: number }
  temperatureForCandidates?: number
  temperatureForSampling?: number
}

export type Rollout = {
  example: Example
  outputMetadata: Record<string, unknown>
  prediction: Record<string, unknown> | undefined
  score: number
  trace: TraceStep[]
}

export type Bucket = {
  maxScore: number
  maxToAvgGap: number
  maxToMinGap: number
  rollouts: Rollout[]
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

function mean(values: number[]): number {
  return values.reduce((acc, v) => acc + v, 0) / values.length
}

/** Python's round(): banker's rounding, used for final-selection spacing. */
export function roundHalfEven(x: number): number {
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
export function topKPlusBaseline(avgScores: number[], k: number): number[] {
  const sorted = avgScores
    .map((avg, idx) => ({ avg, idx }))
    .sort((a, b) => b.avg - a.avg)
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
export function softmaxSample(
  rng: () => number,
  programIdxs: number[],
  avgScores: number[],
  temperature: number
): number {
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
export function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  if (sorted.length === 0) {
    return 0
  }
  const rank = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(rank)
  const hi = Math.ceil(rank)
  const low = sorted[lo] as number
  const high = sorted[hi] as number
  return low + (high - low) * (rank - lo)
}

/**
 * Group model-major rollouts into per-example buckets (stride = bsize), each
 * sorted by score descending, then order buckets by
 * (max−min gap, max score, max−avg gap) lexicographically descending.
 * Rollout records are shallow-copied so strategies never mutate shared state.
 */
export function makeBuckets(rollouts: Rollout[], bsize: number): Bucket[] {
  const buckets: Bucket[] = []
  for (let exampleIdx = 0; exampleIdx < bsize; exampleIdx += 1) {
    const bucket: Rollout[] = []
    for (let i = exampleIdx; i < rollouts.length; i += bsize) {
      bucket.push({ ...(rollouts[i] as Rollout) })
    }
    bucket.sort((a, b) => b.score - a.score)
    const maxScore = (bucket[0] as Rollout).score
    const minScore = (bucket.at(-1) as Rollout).score
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
export function dropDemos(
  candidate: Program<never, unknown>,
  maxDemos: number,
  rng: () => number,
  poissonRNG: () => number
): number {
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

export function appendADemo(
  bucket: Bucket,
  candidate: Program<never, unknown>,
  opts: { demoInputFieldMaxlen: number; p10: number }
): boolean {
  const good = bucket.rollouts[0] as Rollout
  if (good.score <= opts.p10) {
    console.log(
      `Skipping appending a demo as good score ${good.score} is at or below the 10th percentile.`
    )
    return false
  }

  const nameToDemo = new Map<
    string,
    { inputs: Record<string, unknown>; outputs: Record<string, unknown> }
  >()
  for (const step of good.trace) {
    const inputs: Record<string, unknown> = { ...step.inputs }
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
  // biome-ignore lint/suspicious/noTemplateCurlyInString: verbatim upstream field description
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
export function recursiveMask(value: unknown): unknown {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value
  }
  if (Array.isArray(value)) {
    return value.map(recursiveMask)
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        k,
        recursiveMask(v),
      ])
    )
  }
  return `<non-serializable: ${typeof value}>`
}

function serializeField(value: unknown): string {
  return typeof value === "string"
    ? value
    : JSON.stringify(recursiveMask(value), null, 2)
}

function fieldDescriptionLines(schema: z.ZodType): string {
  return Object.entries(schemaProperties(schema))
    .map(
      ([name, prop]) =>
        `${name} (${prop.type ?? "unknown"})${prop.description ? `: ${prop.description}` : ""}`
    )
    .join("\n")
}

const MODULE_SEPARATOR = "-".repeat(80)

function indentContinuations(text: string): string {
  return ["", ...text.split("\n")].join("\n\t\t")
}

export function inspectModules(program: Program<never, unknown>): string {
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
  return blocks.map((block) => block.replace(/^\n+|\n+$/g, "")).join("\n")
}

export type OfferFeedbackResult = {
  discussion: string
  moduleAdvice: Record<string, string>
}

export async function offerFeedback(
  promptModel: LanguageModel,
  moduleNames: string[],
  fields: Record<string, unknown>
): Promise<OfferFeedbackResult> {
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

export async function appendARule(
  bucket: Bucket,
  candidate: Program<never, unknown>,
  opts: {
    p10: number
    p90: number
    promptModel: LanguageModel
  }
): Promise<boolean> {
  const good = bucket.rollouts[0] as Rollout
  const bad = bucket.rollouts.at(-1) as Rollout

  if (good.score <= opts.p10 || bad.score >= opts.p90) {
    console.log(
      `Skipping rule generation as good score ${good.score} is at or below the 10th percentile ` +
        `*or* bad score ${bad.score} is at or above the 90th percentile.`
    )
    return false
  }

  // Blank the uninformative half when there's no real contrast. Local views
  // only — never mutate the shared rollout records.
  let goodView = {
    outputMetadata: good.outputMetadata,
    prediction: good.prediction as Record<string, unknown> | undefined,
    score: good.score as number | "N/A",
    trace: good.trace,
  }
  let badView = {
    outputMetadata: bad.outputMetadata,
    prediction: bad.prediction as Record<string, unknown> | undefined,
    score: bad.score as number | "N/A",
    trace: bad.trace,
  }
  if (good.score <= bad.score) {
    const blank = {
      prediction: { "N/A": "Prediction not available" },
      score: "N/A" as const,
      trace: [] as TraceStep[],
    }
    if (good.score > opts.p90) {
      badView = { ...badView, ...blank }
    } else {
      goodView = { ...goodView, ...blank }
    }
  }

  const toTrajectory = (trace: TraceStep[]) =>
    trace.map((step) => ({
      inputs: step.inputs,
      module_name: step.predictorName,
      outputs: step.outputs,
    }))

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

async function runRollout(
  program: Program<never, unknown>,
  example: Example,
  metric: Metric,
  ctx: RunContext
): Promise<Rollout> {
  const trace: TraceStep[] = []
  let prediction: Record<string, unknown> | undefined
  try {
    prediction = (await program.run(example.inputs as never, {
      ...ctx,
      trace,
    })) as Record<string, unknown>
  } catch (error) {
    console.warn(error)
  }

  let score = 0
  let outputMetadata: Record<string, unknown> = {}
  try {
    const output = await metric(example, prediction)
    if (typeof output === "number") {
      score = output
    } else {
      const { score: metricScore, ...rest } = output
      score = metricScore
      outputMetadata = rest
    }
  } catch (error) {
    console.warn(error)
  }

  return { example, outputMetadata, prediction, score, trace }
}

// --- Main loop --------------------------------------------------------------

export async function simba<TInput extends Record<string, unknown>, TOutput>(
  student: Program<TInput, TOutput>,
  trainset: Example[],
  config: SIMBAConfig
): Promise<SIMBAResult<TInput, TOutput>> {
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

  type AnyProgram = Program<never, unknown>
  const baseline = student.clone() as AnyProgram
  const promptModel =
    config.promptModel ?? (baseline.predictors[0] as AnyPredictor).model

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

  const demoStrategy = (bucket: Bucket, candidate: AnyProgram) =>
    Promise.resolve(
      appendADemo(bucket, candidate, { demoInputFieldMaxlen, p10: currentP10 })
    )
  const ruleStrategy = (bucket: Bucket, candidate: AnyProgram) =>
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

  const nextBatch = (): Example[] => {
    if (instanceIdx + bsize > trainset.length) {
      shuffle(rng, dataIdxs)
      instanceIdx = 0
    }
    const batch = dataIdxs
      .slice(instanceIdx, instanceIdx + bsize)
      .map((i) => trainset[i] as Example)
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
  const sampleRollouts = (batch: Example[]): Promise<Rollout[]> => {
    // The pool is frozen for the whole call, so score the programs once.
    const avg = avgScores()
    const topK = topKPlusBaseline(avg, numCandidates)
    const runs: (() => Promise<Rollout>)[] = []
    for (const modelCtx of prepareModelsForResampling()) {
      for (const example of batch) {
        const srcIdx = softmaxSample(rng, topK, avg, temperatureForSampling)
        // Rollouts never mutate the program, so no clone is needed.
        const rolloutProgram = programs[srcIdx] as AnyProgram
        runs.push(() => runRollout(rolloutProgram, example, metric, modelCtx))
      }
    }
    return Promise.all(runs.map((run) => run()))
  }

  const generateCandidates = async (
    buckets: Bucket[]
  ): Promise<AnyProgram[]> => {
    // Candidates only join the pool after the step, so score it once.
    const avg = avgScores()
    const topK = topKPlusBaseline(avg, numCandidates)
    const candidates: AnyProgram[] = []
    for (const bucket of buckets) {
      const srcIdx = softmaxSample(rng, topK, avg, temperatureForCandidates)
      const candidate = (programs[srcIdx] as AnyProgram).clone()
      dropDemos(candidate, maxDemos, rng, poissonRNG)
      const strategy = strategies[
        Math.floor(rng() * strategies.length)
      ] as (typeof strategies)[number]
      try {
        // A strategy no-op still keeps the candidate. Sequential on purpose:
        // the RNG stream must consume picks in bucket order.
        // biome-ignore lint/performance/noAwaitInLoops: see above
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
    examples: Example[]
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
      winningPrograms.push((candidates[bestIdx] as AnyProgram).clone())
    }

    // Register ALL candidates into the pool.
    for (const [idx, candidate] of candidates.entries()) {
      programs.push(candidate)
      programScores.push(candidateScoreLists[idx] as number[])
    }

    trialLogs.push({ baselineScore, candidateScores, step })
  }

  for (let step = 0; step < maxSteps; step += 1) {
    // biome-ignore lint/performance/noAwaitInLoops: optimization steps are inherently sequential
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
  const finalists = finalIdxs.map((i) => winningPrograms[i] as AnyProgram)

  console.log(
    `VALIDATION: Evaluating ${finalists.length} programs on the full trainset.`
  )
  const finalScores = (await evaluateOn(finalists, trainset)).map(mean)

  const candidateData = finalists
    .map((program, idx) => ({
      program: program as Program<TInput, TOutput>,
      score: finalScores[idx] as number,
    }))
    .sort((a, b) => b.score - a.score)

  // First max wins ties — favors the less-evolved program.
  const bestIdx = finalScores.indexOf(Math.max(...finalScores))
  console.log(
    `Final trainset scores: ${finalScores}, Best: ${Math.max(...finalScores)} (at index ${bestIdx})`
  )

  return {
    candidates: candidateData,
    program: (finalists[bestIdx] as AnyProgram).clone() as Program<
      TInput,
      TOutput
    >,
    trialLogs,
  }
}
