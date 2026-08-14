import type { z } from "zod"
import { gepa } from "@/gepa"
import type { AnyPredictor } from "@/predictor"
import { createProgram, type Program } from "@/program"
import { type Metric, simba } from "@/simba"

export { type GEPAConfig, type GEPAResult, gepa } from "@/gepa"
export { declareStep } from "@/predictor"

/**
 * A step as seen by the workflow builder. `never` on the input side keeps any
 * concretely-typed step assignable here without widening it to `any`.
 */
export type StepLike<TInput = never, TOutput = unknown> = {
  execute: (params: { inputData: TInput }) => Promise<TOutput>
  id: string
}

export type AnyStep = StepLike<never, unknown>

/** Steps keyed by their own literal `id`, so lookups stay precisely typed. */
export type StepMap = Record<string, AnyStep>

type WithStep<TSteps extends StepMap, TStep extends AnyStep> = TSteps & {
  [K in TStep["id"]]: TStep
}

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

export function createWorkflow(_config: WorkflowConfig): Workflow {
  const steps: AnyStep[] = []

  const builder = {
    commit() {
      const stepMap: StepMap = {}
      for (const step of steps) {
        stepMap[step.id] = step
      }
      return { steps: stepMap }
    },
    // biome-ignore lint/suspicious/noThenProperty: mirrors Mastra's workflow builder API
    then(step: AnyStep) {
      steps.push(step)
      return builder
    },
  }

  return builder as unknown as Workflow
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

export type Example = {
  expected: Record<string, unknown>
  input: Record<string, unknown>
}

/** Every expected field must match the prediction exactly for a score of 1. */
const exactMatchMetric = (
  example: Example,
  prediction: Record<string, unknown> | null | undefined
) =>
  Object.entries(example.expected).every(
    ([key, value]) => prediction?.[key] === value
  )
    ? 1
    : 0

/** Steps run in insertion order, each feeding its output to the next. */
function workflowToProgram(workflow: CommittedWorkflow): {
  predictors: AnyPredictor[]
  program: Program<Record<string, unknown>, Record<string, unknown>>
} {
  const steps = Object.values(workflow.steps) as (AnyStep & {
    predictor?: AnyPredictor
  })[]
  const predictors = steps.map((step) => {
    if (!step.predictor) {
      throw new Error(`Step ${step.id} has no predictor to optimize`)
    }
    return step.predictor
  })
  const program = createProgram({
    forward: async (call, input: Record<string, unknown>) => {
      let data = input
      for (const predictor of predictors) {
        // biome-ignore lint/performance/noAwaitInLoops: sequential pipeline
        data = await call(predictor.name, data)
      }
      return data
    },
    predictors,
  })
  return { predictors, program }
}

function programToSteps(
  program: Program<Record<string, unknown>, Record<string, unknown>>,
  stepIds: string[]
): StepMap {
  const tunedSteps: StepMap = {}
  for (const stepId of stepIds) {
    const tuned = program.predictors.find((p) => p.name === stepId)
    if (!tuned) {
      throw new Error(`Tuned program lost predictor ${stepId}`)
    }
    tunedSteps[stepId] = {
      execute: ({ inputData }: { inputData: Record<string, unknown> }) =>
        tuned.call(inputData as never) as Promise<Record<string, unknown>>,
      id: stepId,
    }
  }
  return tunedSteps
}

/** Optimizing preserves the step map, so a tuned workflow stays as typed as its source. */
export async function optimize<TSteps extends StepMap>(
  optimizer: Optimizer,
  workflow: CommittedWorkflow<TSteps>,
  options: { trainset: readonly Example[] }
): Promise<CommittedWorkflow<TSteps>> {
  const { predictors, program } = workflowToProgram(workflow)
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
    return { steps: programToSteps(result.program, stepIds) as TSteps }
  }

  // GEPA is instruction-only: maxDemos is ignored; maxSteps maps to full evals.
  const result = await gepa(program, [...options.trainset], {
    maxFullEvals: optimizer.maxSteps,
    metric: async (gold, prediction) => {
      const output = await metric(gold, prediction ?? undefined)
      return typeof output === "number" ? output : { score: output.score }
    },
    reflectionLM: (predictors[0] as AnyPredictor).model,
    seed: optimizer.seed,
  })
  return { steps: programToSteps(result.program, stepIds) as TSteps }
}
