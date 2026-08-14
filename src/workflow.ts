import type { z } from "zod"

/**
 * A step as seen by the workflow builder. `never` on the input side keeps any
 * concretely-typed step assignable here without widening it to `any`. Steps
 * built by `declareStep` additionally carry the tunable prompt state
 * (description, examples) the optimizers work on.
 */
export type StepLike<TInput = never, TOutput = unknown> = {
  execute: (params: { inputData: TInput }) => Promise<TOutput>
  id: string
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
