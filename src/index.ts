import type { z } from "zod"
import { bootstrapFewShot } from "@/bootstrap"
import { gepa } from "@/gepa"
import type { AnyPredictor } from "@/predictor"
import { createProgram, type Example, type Program } from "@/program"
import { type Metric, simba } from "@/simba"

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

/** Every expected field must match the prediction exactly for a score of 1. */
const exactMatchMetric = (
  example: Example,
  prediction: Record<string, unknown> | null | undefined
) =>
  Object.entries(example.outputs).every(
    ([key, value]) => prediction?.[key] === value
  )
    ? 1
    : 0

/** Steps run in insertion order, each feeding its output to the next. */
function workflowToProgram(
  workflow: CommittedWorkflow
): Program<Record<string, unknown>, Record<string, unknown>> {
  const predictors = Object.values(workflow.steps).map((step) => {
    if (!step.predictor) {
      throw new Error(`Step ${step.id} has no predictor to optimize`)
    }
    return step.predictor
  })
  return createProgram({
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
    // Carrying the tuned predictor keeps the step re-optimizable.
    tunedSteps[stepId] = {
      execute: ({ inputData }: { inputData: Record<string, unknown> }) =>
        tuned.call(inputData as never) as Promise<Record<string, unknown>>,
      id: stepId,
      predictor: tuned,
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
      metric: async (gold, prediction, _trace) => {
        const output = await metric(gold, prediction ?? undefined)
        return typeof output === "number" ? output : output.score
      },
    })
  }

  // maxSteps maps to full evals.
  const result = await gepa(student, [...options.trainset], {
    maxFullEvals: optimizer.maxSteps,
    // Feedback strings pass through — GEPA's reflection consumes them.
    metric: async (gold, prediction) => {
      const output = await metric(gold, prediction ?? undefined)
      if (typeof output === "number") {
        return output
      }
      const { feedback, score } = output
      return {
        score,
        ...(typeof feedback === "string" ? { feedback } : {}),
      }
    },
    reflectionLM: (program.predictors[0] as AnyPredictor).model,
    seed: optimizer.seed,
  })
  return { steps: programToSteps(result.program, stepIds) as TSteps }
}
