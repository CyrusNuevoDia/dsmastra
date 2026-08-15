import type { AnyWorkflow } from "@mastra/core/workflows"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { at, first } from "../../collections"
import type { Fields } from "../../fields"
import { bootstrapFewShotPrompts } from "../../optimizers/bootstrap"
import { createWorkflowAdapter } from "../../optimizers/gepa/adapter"
import type { GEPAMetric, ReflectionModel } from "../../optimizers/gepa/adapter"
import {
  acceptReflection,
  aggregateScore,
  argmax,
  deserializeGEPALoopState,
  deserializeGEPAState,
  initGEPALoopState,
  initGEPAState,
  mergeDue,
  prepareReflection,
  runGEPA,
  runMergeBranch,
  serializeGEPALoopState,
  serializeGEPAState,
} from "../../optimizers/gepa/engine"
import type {
  Candidate,
  EngineOptions,
  GEPAState,
  ReflectionPlan,
} from "../../optimizers/gepa/engine"
import {
  declarativeSteps,
  loadPrompts,
  optimizerResultSchema,
  promptsOf,
  promptsSchema,
} from "../../optimizers/utils"
import type {
  OptimizerCheckpoint,
  Prompts,
  SavePrompts,
} from "../../optimizers/utils"
import { createRNG, restoreRNG } from "../../random"
import type { RNG } from "../../random"
import { resolveScorer, scorerMetric } from "../../scorers"
import type { MetricOutput, ScorerRef } from "../../scorers"
import type { Example, ScoreTarget } from "../../step"

// --- Budget -----------------------------------------------------------------

const AUTO_CANDIDATES = { heavy: 18, light: 6, medium: 12 } as const

export type GEPAAuto = keyof typeof AUTO_CANDIDATES

/**
 * DSPy's auto-budget estimate. `fullEvalSteps` (m) exists only here — the
 * engine has no periodic full-eval scheduling.
 */
export const autoBudget = (
  stepsCount: number,
  candidates: number,
  validationSetSize: number,
  minibatchSize = 35,
  fullEvalSteps = 5
): number => {
  const trialsCount = Math.floor(
    Math.max(2 * (stepsCount * 2) * Math.log2(candidates), 1.5 * candidates)
  )
  if (trialsCount < 0 || validationSetSize < 0 || minibatchSize < 0) {
    throw new Error("autoBudget arguments must be non-negative")
  }
  if (fullEvalSteps < 1) {
    throw new Error("fullEvalSteps must be >= 1")
  }
  let total = validationSetSize + candidates * 5 + trialsCount * minibatchSize
  if (trialsCount === 0) {
    return total
  }
  total +=
    (Math.floor((trialsCount + 1) / fullEvalSteps) +
      1 +
      (trialsCount < fullEvalSteps ? 1 : 0)) *
    validationSetSize
  return total
}

// --- Configuration ----------------------------------------------------------

type EngineTuning<TInput, TOutput> = {
  addFormatFailureAsFeedback?: boolean
  candidateSelectionStrategy?: "currentBest" | "pareto"
  componentSelector?: "all" | "roundRobin"
  failureScore?: number
  maxMergeInvocations?: number
  perfectScore?: number
  reflectionMinibatchSize?: number
  seed?: number
  skipPerfectScore?: boolean
  useMerge?: boolean
  validationSet?: Example<TInput, TOutput>[]
  warnOnScoreMismatch?: boolean
}

export type GEPAPromptsConfig<TInput = Fields, TOutput = Fields> = EngineTuning<
  TInput,
  TOutput
> & {
  /** Exactly one of `auto`, `maxFullEvals`, `maxMetricCalls` must be set. */
  auto?: GEPAAuto
  maxFullEvals?: number
  maxMetricCalls?: number
  metric: GEPAMetric<TInput, TOutput>
  /** Called with each new aggregate-score-best prompts snapshot — checkpointing. */
  onImprovement?: (prompts: Prompts) => Promise<void>
  reflectionModel: ReflectionModel
}

export type GEPAConfig = EngineTuning<Fields, Fields> & {
  /** Exactly one of `auto`, `maxFullEvals`, `maxScorerCalls` must be set. */
  auto?: GEPAAuto
  /** Pause hook: called before every iteration; returning true suspends the
   * run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint
  /** When > 0, a bootstrapFewShot pre-pass installs few-shot examples first. */
  maxFewShotExamples?: number
  /** Labeled backfill cap for the pre-pass; defaults to maxFewShotExamples. */
  maxLabeledExamples?: number
  maxFullEvals?: number
  /** Budget cap counted in scorer runs — DSPy's maxMetricCalls. */
  maxScorerCalls?: number
  /** LM used to propose new descriptions; defaults to the first step's model. */
  reflectionModel?: ReflectionModel
  savePrompts: SavePrompts
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Its `reason` (generateReason step) feeds
   * GEPA's reflection LM as feedback. */
  scorer: ScorerRef
  trainingSet: readonly Example[]
}

const VALIDATION_SET_SIZE_NOTE = 35

const resolveBudget = (
  config: {
    auto?: GEPAAuto
    maxFullEvals?: number
    maxMetricCalls?: number
    validationSet?: readonly unknown[]
  },
  stepsCount: number,
  trainingSet: readonly unknown[],
  validationSetSize: number
): number => {
  const provided = [
    config.auto,
    config.maxFullEvals,
    config.maxMetricCalls,
  ].filter((value) => value !== undefined)
  if (provided.length !== 1) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
    )
  }
  if (config.maxMetricCalls !== undefined) {
    return config.maxMetricCalls
  }
  if (config.maxFullEvals !== undefined) {
    // DSPy quirk: when no validationSet was given the multiplier is
    // len(trainingSet) only, even though validationSet then defaults to it.
    return (
      config.maxFullEvals *
      (trainingSet.length + (config.validationSet?.length ?? 0))
    )
  }
  if (config.auto === undefined) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxMetricCalls must be set"
    )
  }
  return autoBudget(
    Math.max(stepsCount, 1),
    AUTO_CANDIDATES[config.auto],
    validationSetSize
  )
}

/** The description-only Candidate view of a student snapshot. */
const seedCandidateOf = (studentPrompts: Prompts): Candidate =>
  Object.fromEntries(
    Object.entries(studentPrompts.steps).map(([id, step]) => [
      id,
      step.description,
    ])
  )

// --- Result -----------------------------------------------------------------

export type GEPAPromptsResult = {
  /** argmax of validationAggregateScores, lowest index winning ties. */
  bestIdx: number
  candidates: Candidate[]
  discoveryEvalCounts: number[]
  validationSetEvalsCount: number
  parents: (number | null)[][]
  perValidationInstanceBestCandidates: Map<number, Set<number>>
  prompts: Prompts
  seed: number
  totalMetricCalls: number
  validationAggregateScores: number[]
  validationSubscores: number[][]
}

export const buildResult = (
  state: GEPAState,
  validationSetSize: number,
  seed: number,
  buildPrompts: (candidate: Candidate) => Prompts
): GEPAPromptsResult => {
  const validationAggregateScores =
    state.candidateValidationSubscores.map(aggregateScore)
  const bestIdx = argmax(validationAggregateScores)
  return {
    bestIdx,
    candidates: state.programCandidates,
    discoveryEvalCounts: state.metricCallCountsByDiscovery,
    parents: state.parentProgramForCandidate,
    perValidationInstanceBestCandidates:
      state.programAtParetoFrontValidationSet,
    prompts: buildPrompts(at(state.programCandidates, bestIdx, "candidates")),
    seed,
    totalMetricCalls: state.totalEvalsCount,
    validationAggregateScores,
    validationSetEvalsCount: state.validationSetEvalsCount,
    validationSubscores: Array.from(
      { length: state.programCandidates.length },
      (_, idx) =>
        Array.from(
          { length: validationSetSize },
          (_2, validationId) =>
            at(
              state.candidateValidationSubscores,
              idx,
              "candidate validation subscores"
            ).get(validationId) ?? Number.NaN
        )
    ),
  }
}

// --- Prompts-level entry point ----------------------------------------------

export const gepaPrompts = async <TInput, TOutput>(
  workflow: AnyWorkflow,
  prompts: Prompts,
  trainingSet: Example<TInput, TOutput>[],
  config: GEPAPromptsConfig<TInput, TOutput>
): Promise<GEPAPromptsResult> => {
  if (trainingSet.length === 0) {
    throw new Error("GEPA requires a non-empty trainingSet")
  }
  const validationSet = config.validationSet ?? trainingSet
  if (!config.validationSet) {
    console.warn(
      "GEPA: no validationSet provided; using the trainingSet for validation."
    )
  }
  if (validationSet.length > VALIDATION_SET_SIZE_NOTE) {
    console.warn(
      `GEPA: validationSet has ${validationSet.length} examples; every accepted candidate costs a full validationSet eval.`
    )
  }

  const seedCandidate = seedCandidateOf(prompts)
  const maxMetricCalls = resolveBudget(
    config,
    declarativeSteps(workflow).length,
    trainingSet,
    validationSet.length
  )

  const seed = config.seed ?? 0
  const engineRNG = createRNG(seed)
  const adapterRNG = createRNG(seed)

  const adapter = createWorkflowAdapter({
    adapterRNG,
    addFormatFailureAsFeedback: config.addFormatFailureAsFeedback ?? false,
    basePrompts: prompts,
    failureScore: config.failureScore ?? 0,
    metric: config.metric,
    reflectionModel: config.reflectionModel,
    warnOnScoreMismatch: config.warnOnScoreMismatch ?? true,
    workflow,
  })

  const { onImprovement } = config
  const state = await runGEPA(adapter, {
    candidateSelectionStrategy: config.candidateSelectionStrategy ?? "pareto",
    componentSelector: config.componentSelector ?? "roundRobin",
    maxMergeInvocations: config.maxMergeInvocations ?? 5,
    maxMetricCalls,
    onImprovement:
      onImprovement &&
      ((candidate) => onImprovement(adapter.buildPrompts(candidate))),
    perfectScore: config.perfectScore ?? 1,
    reflectionMinibatchSize: config.reflectionMinibatchSize ?? 3,
    rng: engineRNG,
    seedCandidate,
    skipPerfectScore: config.skipPerfectScore ?? true,
    trainingSet,
    useMerge: config.useMerge ?? true,
    validationSet,
  })

  return buildResult(state, validationSet.length, seed, adapter.buildPrompts)
}

// --- Workflow-level entry point ----------------------------------------------

/** The serialized engine state, as it crosses workflow step boundaries. */
const serializedStateSchema = z.object({
  candidateValidationSubscores: z.array(
    z.array(z.tuple([z.number(), z.number()]))
  ),
  i: z.number(),
  metricCallCountsByDiscovery: z.array(z.number()),
  parentProgramForCandidate: z.array(z.array(z.number().nullable())),
  paretoFrontValidationSet: z.array(z.tuple([z.number(), z.number()])),
  programAtParetoFrontValidationSet: z.array(
    z.tuple([z.number(), z.array(z.number())])
  ),
  programCandidates: z.array(z.record(z.string(), z.string())),
  stepIdToUpdateNextForCandidate: z.array(z.number()),
  totalEvalsCount: z.number(),
  validationSetEvalsCount: z.number(),
})

const serializedLoopSchema = z.object({
  bestAgg: z.number(),
  lastIterFoundNewProgram: z.boolean(),
  mergeMemory: z.object({
    producedByPair: z.array(z.string()),
    triedTriplets: z.array(z.string()),
  }),
  mergesDue: z.number(),
  samplerState: z.object({
    epoch: z.number(),
    shuffled: z.array(z.number()),
  }),
  totalMergesTested: z.number(),
})

const reflectiveExampleSchema = z.object({
  Feedback: z.string(),
  "Generated Outputs": z.union([z.record(z.string(), z.string()), z.string()]),
  Inputs: z.record(z.string(), z.string()),
})

const reflectionPlanSchema = z.object({
  components: z.array(z.string()),
  minibatchIds: z.array(z.number()),
  parentIdx: z.number(),
  parentScores: z.array(z.number()),
  reflectiveDataset: z.record(z.string(), z.array(reflectiveExampleSchema)),
})

const iterationSchema = z.object({
  loop: serializedLoopSchema,
  rng: z.object({ adapter: z.number(), engine: z.number() }),
  state: serializedStateSchema,
  studentPrompts: promptsSchema,
})

const reflectedSchema = iterationSchema.extend({
  plan: reflectionPlanSchema.nullable(),
})

const proposedSchema = reflectedSchema.extend({
  newTexts: z.record(z.string(), z.string()).nullable(),
})

type IterationPayload = z.infer<typeof iterationSchema>

/**
 * Genetic-Pareto reflective prompt evolution as a Mastra workflow over the
 * target `workflow`: a pre-pass step optionally bootstraps few-shot examples
 * (its metric calls are not billed to GEPA's budget, matching DSPy), a
 * seed-eval step scores the seed candidate over the validationSet, and a
 * durable dountil loop runs one GEPA iteration per pass — split into a
 * `reflect` step (parent selection, minibatch rollouts, reflective dataset,
 * or the merge branch), a `propose` step that makes the reflection-LM calls,
 * and an `accept` step (child evaluation and Pareto bookkeeping). Every
 * candidate crosses step boundaries as a JSON snapshot and randomness as
 * checkpointed RNG state, so a storage-backed run resumes mid-optimization
 * without redoing completed iterations, and savePrompts checkpoints the best
 * candidate whenever the aggregate score improves.
 */
export const createGEPAWorkflow = (
  workflow: AnyWorkflow,
  config: GEPAConfig
) => {
  const {
    checkpoint,
    maxFewShotExamples = 0,
    maxLabeledExamples,
    maxScorerCalls,
    reflectionModel,
    savePrompts,
    scorer,
    trainingSet,
    ...tuning
  } = config
  const examples = [...trainingSet]
  if (examples.length === 0) {
    throw new Error("GEPA requires a non-empty trainingSet")
  }
  const budgetKnobs = [tuning.auto, tuning.maxFullEvals, maxScorerCalls].filter(
    (value) => value !== undefined
  )
  if (budgetKnobs.length !== 1) {
    throw new Error(
      "Exactly one of auto, maxFullEvals, maxScorerCalls must be set"
    )
  }
  const validationSet = tuning.validationSet ?? examples
  if (!tuning.validationSet) {
    console.warn(
      "GEPA: no validationSet provided; using the trainingSet for validation."
    )
  }
  if (validationSet.length > VALIDATION_SET_SIZE_NOTE) {
    console.warn(
      `GEPA: validationSet has ${validationSet.length} examples; every accepted candidate costs a full validationSet eval.`
    )
  }
  const stepsCount = declarativeSteps(workflow).length
  const maxMetricCalls = resolveBudget(
    { ...tuning, maxMetricCalls: maxScorerCalls },
    stepsCount,
    examples,
    validationSet.length
  )
  const seed = tuning.seed ?? 0
  const metric = scorerMetric(resolveScorer(workflow, scorer))

  // One scorer run per rollout: GEPA scores each rollout, then asks again for
  // the selected step's feedback — the prediction object identifies its
  // rollout, so the first result (score + reason-as-feedback) is reused.
  // Rejections are evicted so a transient scorer failure isn't cached.
  const cache = new WeakMap<object, Promise<Awaited<MetricOutput>>>()
  const cachedMetric = (
    gold: Example,
    prediction: Fields | null,
    target?: ScoreTarget
  ): MetricOutput => {
    if (prediction === null) {
      return metric(gold, undefined, target)
    }
    const hit = cache.get(prediction)
    if (hit) {
      return hit
    }
    const pending = (async () => {
      try {
        return await metric(gold, prediction, target)
      } catch (error) {
        cache.delete(prediction)
        throw error
      }
    })()
    cache.set(prediction, pending)
    return pending
  }
  // The scorer's `reason` rides along as `feedback` — GEPA's reflection reads it.
  const gepaMetric: GEPAMetric = (
    gold,
    prediction,
    _trace,
    _stepId,
    _stepTrace,
    target
  ) => cachedMetric(gold, prediction, target)

  // Derived, non-serializable machinery is rebuilt per step invocation from
  // the serializable payload — never carried across steps in closures — so a
  // resumed run in a fresh process reconstructs the exact same world.
  const buildAdapter = (studentPrompts: Prompts, adapterRNGState: number) => {
    const adapterRNG = restoreRNG(adapterRNGState)
    const adapter = createWorkflowAdapter({
      adapterRNG,
      addFormatFailureAsFeedback: tuning.addFormatFailureAsFeedback ?? false,
      basePrompts: studentPrompts,
      failureScore: tuning.failureScore ?? 0,
      metric: gepaMetric,
      reflectionModel:
        reflectionModel ??
        first(declarativeSteps(workflow), "workflow steps").model,
      warnOnScoreMismatch: tuning.warnOnScoreMismatch ?? true,
      workflow,
    })
    return { adapter, adapterRNG }
  }

  const engineOptionsFor = (
    rng: RNG,
    adapter: ReturnType<typeof buildAdapter>["adapter"],
    seedCandidate: Candidate
  ): EngineOptions<Fields, Fields> => ({
    candidateSelectionStrategy: tuning.candidateSelectionStrategy ?? "pareto",
    componentSelector: tuning.componentSelector ?? "roundRobin",
    maxMergeInvocations: tuning.maxMergeInvocations ?? 5,
    maxMetricCalls,
    onImprovement: async (candidate) => {
      await savePrompts(adapter.buildPrompts(candidate))
    },
    perfectScore: tuning.perfectScore ?? 1,
    reflectionMinibatchSize: tuning.reflectionMinibatchSize ?? 3,
    rng,
    seedCandidate,
    skipPerfectScore: tuning.skipPerfectScore ?? true,
    trainingSet: examples,
    useMerge: tuning.useMerge ?? true,
    validationSet,
  })

  const prepass = createStep({
    description:
      "Optional BootstrapFewShot pre-pass installing few-shot examples",
    execute: async () => {
      let studentPrompts = promptsOf(workflow)
      if (maxFewShotExamples > 0) {
        studentPrompts = await bootstrapFewShotPrompts(
          workflow,
          studentPrompts,
          examples,
          {
            maxFewShotExamples,
            // A TOTAL cap per step, so the labeled backfill shares it instead of
            // DSPy's default 16 — unless the caller raises it explicitly.
            maxLabeledExamples: maxLabeledExamples ?? maxFewShotExamples,
            metric: (gold, prediction) => cachedMetric(gold, prediction),
          }
        )
      }
      return { studentPrompts }
    },
    id: "prepass",
    inputSchema: z.object({}),
    outputSchema: z.object({ studentPrompts: promptsSchema }),
  })

  const seedEval = createStep({
    description: "Score the seed candidate over the validationSet",
    execute: async ({ inputData }) => {
      const { studentPrompts } = inputData
      const { adapter } = buildAdapter(studentPrompts, seed)
      const seedCandidate = seedCandidateOf(studentPrompts)
      const evaluated = await adapter.evaluate(
        validationSet,
        seedCandidate,
        false
      )
      const state = initGEPAState(seedCandidate, evaluated.scores)
      const loop = initGEPALoopState(state)
      return {
        loop: serializeGEPALoopState(loop),
        rng: { adapter: createRNG(seed).state, engine: createRNG(seed).state },
        state: serializeGEPAState(state),
        studentPrompts,
      }
    },
    id: "seed-eval",
    inputSchema: z.object({ studentPrompts: promptsSchema }),
    outputSchema: iterationSchema,
  })

  const reflect = createStep({
    description:
      "One iteration's prologue: merge branch, or minibatch rollouts and the reflective dataset",
    execute: async ({ inputData, resumeData, suspend }) => {
      const payload: IterationPayload = inputData
      const state = deserializeGEPAState(payload.state)
      if (state.totalEvalsCount >= maxMetricCalls) {
        // Budget already spent — the dountil body still runs once.
        return { ...payload, plan: null }
      }
      if (!resumeData && (await checkpoint?.({ iteration: state.i + 1 }))) {
        return await suspend({ iteration: state.i + 1 })
      }
      const loop = deserializeGEPALoopState(payload.loop)
      const engineRNG = restoreRNG(payload.rng.engine)
      const { adapter, adapterRNG } = buildAdapter(
        payload.studentPrompts,
        payload.rng.adapter
      )
      const options = engineOptionsFor(
        engineRNG,
        adapter,
        seedCandidateOf(payload.studentPrompts)
      )
      state.i += 1
      let plan: ReflectionPlan | null = null
      if (mergeDue(loop, options)) {
        const outcome = await runMergeBranch(adapter, state, loop, options)
        // A PRODUCED merge proposal ends the iteration whether accepted or
        // rejected; a fruitless search falls through to reflection.
        if (outcome === "none") {
          plan = await prepareReflection(adapter, state, loop, options)
        }
      } else {
        plan = await prepareReflection(adapter, state, loop, options)
      }
      return {
        loop: serializeGEPALoopState(loop),
        plan,
        rng: { adapter: adapterRNG.state, engine: engineRNG.state },
        state: serializeGEPAState(state),
        studentPrompts: payload.studentPrompts,
      }
    },
    id: "reflect",
    inputSchema: iterationSchema,
    outputSchema: reflectedSchema,
    resumeSchema: z.object({}),
    suspendSchema: z.object({ iteration: z.number() }),
  })

  const propose = createStep({
    description: "Reflection-LM calls proposing new instruction texts",
    execute: async ({ inputData }) => {
      const { plan } = inputData
      if (!plan) {
        return { ...inputData, newTexts: null }
      }
      const { adapter } = buildAdapter(
        inputData.studentPrompts,
        inputData.rng.adapter
      )
      const parent = at(
        inputData.state.programCandidates,
        plan.parentIdx,
        "candidates"
      )
      const newTexts = await adapter.proposeNewTexts(
        parent,
        plan.reflectiveDataset,
        plan.components
      )
      return { ...inputData, newTexts }
    },
    id: "propose",
    inputSchema: reflectedSchema,
    outputSchema: proposedSchema,
  })

  const accept = createStep({
    description:
      "Child evaluation, acceptance, Pareto bookkeeping, checkpointing",
    execute: async ({ inputData }) => {
      const { newTexts, plan, ...payload } = inputData
      if (!(plan && newTexts)) {
        return payload
      }
      const state = deserializeGEPAState(payload.state)
      const loop = deserializeGEPALoopState(payload.loop)
      const engineRNG = restoreRNG(payload.rng.engine)
      const { adapter, adapterRNG } = buildAdapter(
        payload.studentPrompts,
        payload.rng.adapter
      )
      const options = engineOptionsFor(
        engineRNG,
        adapter,
        seedCandidateOf(payload.studentPrompts)
      )
      await acceptReflection(adapter, state, loop, options, plan, newTexts)
      return {
        loop: serializeGEPALoopState(loop),
        rng: { adapter: adapterRNG.state, engine: engineRNG.state },
        state: serializeGEPAState(state),
        studentPrompts: payload.studentPrompts,
      }
    },
    id: "accept",
    inputSchema: proposedSchema,
    outputSchema: iterationSchema,
  })

  /* oxlint-disable promise/prefer-await-to-then -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  const iteration = createWorkflow({
    id: "iteration",
    inputSchema: iterationSchema,
    outputSchema: iterationSchema,
  })
    .then(reflect)
    .then(propose)
    .then(accept)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then */

  const finalize = createStep({
    description: "Select the winner, persist it, and land it on the workflow",
    execute: async ({ inputData }) => {
      const payload: IterationPayload = inputData
      const state = deserializeGEPAState(payload.state)
      const { adapter } = buildAdapter(
        payload.studentPrompts,
        payload.rng.adapter
      )
      const result = buildResult(
        state,
        validationSet.length,
        seed,
        adapter.buildPrompts
      )
      await savePrompts(result.prompts)
      // The winner's prompt state lands in place on the caller's workflow.
      loadPrompts(workflow, result.prompts)
      return {
        // Every candidate GEPA tried, as a JSON-safe snapshot paired with its
        // validation aggregate score. Candidates are description-only; the
        // snapshot carries the student's (possibly pre-pass-installed)
        // examples.
        candidates: result.candidates.map((candidate, idx) => {
          const pair: [Prompts, { score: number }] = [
            adapter.buildPrompts(candidate),
            {
              score: at(
                result.validationAggregateScores,
                idx,
                "aggregate scores"
              ),
            },
          ]
          return pair
        }),
        score: at(
          result.validationAggregateScores,
          result.bestIdx,
          "aggregate scores"
        ),
      }
    },
    id: "finalize",
    inputSchema: iterationSchema,
    outputSchema: optimizerResultSchema,
  })

  /* oxlint-disable promise/prefer-await-to-then, promise/no-return-wrap -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  return createWorkflow({
    id: `${workflow.id}.gepa`,
    inputSchema: z.object({}),
    outputSchema: optimizerResultSchema,
  })
    .then(prepass)
    .then(seedEval)
    .dountil(iteration, ({ inputData }) =>
      Promise.resolve(inputData.state.totalEvalsCount >= maxMetricCalls)
    )
    .then(finalize)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then, promise/no-return-wrap */
}
