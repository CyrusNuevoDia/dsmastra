import type { z } from "zod"

import { bootstrapFewShot } from "@/bootstrap"
import { first } from "@/collections"
import type { Fields } from "@/fields"
import { gepa } from "@/gepa"
import type { MetricResult } from "@/metric"
import type { AnyPredictor } from "@/predictor"
import { createProgram } from "@/program"
import type { Example, Program } from "@/program"
import { simba } from "@/simba"
import type { Metric } from "@/simba"

export {
  type BootstrapConfig,
  bootstrapFewShot,
  labeledFewShot,
} from "@/bootstrap"
export { type GEPAConfig, type GEPAResult, gepa } from "@/gepa"
export { declareStep } from "@/predictor"
export type { Example } from "@/program"

/**
 * A step as seen by the workflow builder. `never` on the input side keeps any
 * concretely-typed step assignable here without widening it to `any`.
 * `predictor` is present on steps built by `declareStep` and is what
 * `optimize` tunes.
 */
export type StepLike<TInput = never, TOutput = unknown> = {
  execute: (params: { inputData: TInput }) => Promise<TOutput>
  id: string
  predictor?: AnyPredictor
}

export type AnyStep = StepLike<never, unknown>

/** Steps keyed by their own literal `id`, so lookups stay precisely typed. */
export type StepMap = Record<string, AnyStep>

type WithStep<TSteps extends StepMap, TStep extends AnyStep> = TSteps &
  Record<TStep["id"], TStep>

export type WorkflowConfig = {
  id: string
  inputSchema: z.ZodType
  outputSchema: z.ZodType
}

export type Workflow<TSteps extends StepMap = Record<never, AnyStep>> = {
  commit: () => CommittedWorkflow<TSteps>
  then: <TStep extends AnyStep>(
    step: TStep
  ) => Workflow<WithStep<TSteps, TStep>>
}

export type CommittedWorkflow<TSteps extends StepMap = StepMap> = {
  steps: TSteps
}

export const createWorkflow = (_config: WorkflowConfig): Workflow => {
  const steps: AnyStep[] = []

  // One mutable builder serves every `then` call. Each call returns `this` while
  // the *type* grows by the step just added, which is why `then` is declared
  // generic on `Workflow` and simply hands the same object back here: the runtime
  // value never changes, only the compile-time view of which steps it holds.
  const builder: Workflow = {
    commit: () => {
      const stepMap: StepMap = {}
      for (const step of steps) {
        stepMap[step.id] = step
      }
      return { steps: stepMap }
    },
    // oxlint-disable-next-line unicorn/no-thenable -- mirrors Mastra's workflow builder API
    then: <TStep extends AnyStep>(step: TStep) => {
      steps.push(step)
      // SAFETY: the builder is the same object before and after; only the
      // compile-time record of which steps it holds grows. `step` was just pushed
      // onto `steps`, so by the time `commit` reads them the map really does
      // contain an entry keyed by `step.id` — exactly what `WithStep` adds.
      return builder as Workflow<WithStep<Record<never, AnyStep>, TStep>>
    },
  }

  return builder
}

export type OptimizerConfig = {
  bsize?: number
  maxDemos: number
  maxSteps: number
  /** Defaults to exact match on every expected field. */
  metric?: Metric
  numCandidates?: number
  seed?: number
}

export type Optimizer = OptimizerConfig & {
  type: "GEPA" | "SIMBA"
}

export const GEPA = (config: OptimizerConfig): Optimizer => ({
  type: "GEPA",
  ...config,
})

export const SIMBA = (config: OptimizerConfig): Optimizer => ({
  type: "SIMBA",
  ...config,
})

/** Every expected field must match the prediction exactly for a score of 1. */
const exactMatchMetric = (
  example: Example,
  prediction: Fields | null | undefined
): MetricResult => ({
  score: Object.entries(example.outputs).every(
    ([key, value]) => prediction?.[key] === value
  )
    ? 1
    : 0,
})

/** Steps run in insertion order, each feeding its output to the next. */
const workflowToProgram = (
  workflow: CommittedWorkflow
): Program<Fields, Fields> => {
  const predictors = Object.values(workflow.steps).map((step) => {
    if (!step.predictor) {
      throw new Error(`Step ${step.id} has no predictor to optimize`)
    }
    return step.predictor
  })
  return createProgram({
    forward: async (call, input: Fields) => {
      let data = input
      for (const predictor of predictors) {
        // oxlint-disable-next-line no-await-in-loop -- sequential pipeline
        data = await call(predictor.name, data)
      }
      return data
    },
    predictors,
  })
}

const programToSteps = (
  program: Program<Fields, Fields>,
  stepIds: string[]
) => {
  const tunedSteps: StepMap = {}
  for (const stepId of stepIds) {
    const tuned = program.predictors.find((p) => p.name === stepId)
    if (!tuned) {
      throw new Error(`Tuned program lost predictor ${stepId}`)
    }
    // Carrying the tuned predictor keeps the step re-optimizable.
    tunedSteps[stepId] = {
      execute: ({ inputData }: { inputData: Fields }) => tuned.call(inputData),
      id: stepId,
      predictor: tuned,
    }
  }
  return tunedSteps
}

/** Optimizing preserves the step map, so a tuned workflow stays as typed as its source. */
export const optimize = async <TSteps extends StepMap>(
  optimizer: Optimizer,
  workflow: CommittedWorkflow<TSteps>,
  options: { trainset: readonly Example[] }
): Promise<CommittedWorkflow<TSteps>> => {
  const program = workflowToProgram(workflow)
  const stepIds = Object.keys(workflow.steps)
  const metric = optimizer.metric ?? exactMatchMetric

  if (optimizer.type === "SIMBA") {
    const result = await simba(program, [...options.trainset], {
      bsize: optimizer.bsize ?? Math.min(32, options.trainset.length),
      maxDemos: optimizer.maxDemos,
      maxSteps: optimizer.maxSteps,
      metric,
      numCandidates: optimizer.numCandidates,
      seed: optimizer.seed,
    })
    // SAFETY: `stepIds` is `Object.keys(workflow.steps)`, and `programToSteps`
    // writes exactly one entry per id, so the rebuilt map has precisely the keys
    // `TSteps` declares — the optimizer replaces each step's predictor, never the
    // set of steps.
    return { steps: programToSteps(result.program, stepIds) as TSteps }
  }

  // Demos come from a BootstrapFewShot pre-pass (DSPy-style teleprompter
  // composition): bootstrap installs demos on the predictors, then GEPA
  // evolves instructions on the demo-carrying program. Bootstrap metric calls
  // are not billed to GEPA's budget, matching DSPy.
  let student = program
  if (optimizer.maxDemos > 0) {
    student = await bootstrapFewShot(program, [...options.trainset], {
      maxBootstrappedDemos: optimizer.maxDemos,
      // The wrapper's maxDemos is a TOTAL cap per predictor, so the labeled
      // backfill shares it instead of DSPy's default 16.
      maxLabeledDemos: optimizer.maxDemos,
      // Same contract on both sides, so the user's metric passes straight through.
      metric: (gold, prediction) => metric(gold, prediction ?? undefined),
    })
  }

  // maxSteps maps to full evals.
  const result = await gepa(student, [...options.trainset], {
    maxFullEvals: optimizer.maxSteps,
    // Any `feedback` the metric reported rides along — GEPA's reflection reads it.
    metric: (gold, prediction) => metric(gold, prediction ?? undefined),
    reflectionLM: first(program.predictors, "program predictors").model,
    seed: optimizer.seed,
  })
  // SAFETY: same invariant as the SIMBA branch above — `stepIds` came from
  // `Object.keys(workflow.steps)` and `programToSteps` writes exactly one entry
  // per id, so the rebuilt map carries precisely the keys `TSteps` declares.
  return { steps: programToSteps(result.program, stepIds) as TSteps }
}
