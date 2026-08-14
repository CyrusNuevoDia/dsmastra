import type { Fields } from "@/fields"
import { createProgram } from "@/program"
import type { Program } from "@/program"
import type { AnyTunableStep, Example } from "@/step"
import type { AnyStep, CommittedWorkflow, StepMap } from "@/workflow"

/**
 * The tuned prompt state of a workflow: everything an optimizer changes and
 * nothing it doesn't. JSON-safe, so callers can persist it wherever they like.
 */
export type Prompts = {
  steps: Record<string, { description: string; examples: Example[] }>
  version: 1
}

/**
 * Where tuned prompts go. Required on every optimizer config — the types make
 * forgetting to persist impossible. Called with the current best prompts every
 * time an optimizer improves on them, so a crashed run still leaves its best
 * result behind (last write wins).
 */
// oxlint-disable-next-line anti-slop/no-unknown-returns -- deliberately permissive: callers may return whatever their storage layer returns (a write result, a DB row); the optimizers only await it
export type SavePrompts = (prompts: Prompts) => Promise<unknown>

const isTunableStep = (step: AnyStep): step is AnyTunableStep =>
  "description" in step && "examples" in step && "clone" in step

/** Narrow a workflow's steps to tunable ones, in insertion order. */
export const tunableSteps = (workflow: CommittedWorkflow): AnyTunableStep[] =>
  Object.values(workflow.steps).map((step) => {
    if (!isTunableStep(step)) {
      throw new Error(
        `Step ${step.id} was not built with declareStep, so it cannot be optimized`
      )
    }
    return step
  })

/** Steps run in insertion order, each feeding its output to the next. */
export const workflowToProgram = (
  workflow: CommittedWorkflow
): Program<Fields, Fields> => {
  const steps = tunableSteps(workflow).map((step) => step.clone())
  return createProgram({
    forward: async (call, inputData: Fields) => {
      let data = inputData
      for (const step of steps) {
        // oxlint-disable-next-line no-await-in-loop -- sequential pipeline
        data = await call(step.id, data)
      }
      return data
    },
    steps,
  })
}

/**
 * Rebuild a workflow from a tuned program. The program's steps ARE real
 * tunable steps, so the map is just re-keyed by id; carrying them over keeps
 * the workflow re-optimizable.
 */
export const programToWorkflow = <TSteps extends StepMap>(
  program: Program<Fields, Fields>,
  source: CommittedWorkflow<TSteps>
): CommittedWorkflow<TSteps> => {
  const tunedSteps: StepMap = {}
  for (const stepId of Object.keys(source.steps)) {
    const tuned = program.steps.find((step) => step.id === stepId)
    if (!tuned) {
      throw new Error(`Tuned program lost step ${stepId}`)
    }
    tunedSteps[stepId] = tuned
  }
  // SAFETY: the loop writes exactly one entry per key of `source.steps`, so the
  // rebuilt map has precisely the keys `TSteps` declares — the optimizer
  // replaces each step's prompt state, never the set of steps.
  return { steps: tunedSteps as TSteps }
}

/** Snapshot a program's tuned prompt state as a JSON-safe payload. */
export const promptsOf = (program: Program<Fields, Fields>): Prompts => ({
  steps: Object.fromEntries(
    program.steps.map((step) => [
      step.id,
      {
        description: step.description,
        examples: structuredClone(step.examples),
      },
    ])
  ),
  version: 1,
})

/**
 * Apply saved prompts to a workflow, returning a new workflow with the state
 * installed — the input is never mutated. Throws when the prompts' step ids
 * don't exactly match the workflow's, so a stale snapshot fails loudly instead
 * of half-applying. Parsing and storage are the caller's problem.
 */
export const loadPrompts = <TSteps extends StepMap>(
  workflow: CommittedWorkflow<TSteps>,
  prompts: Prompts
): CommittedWorkflow<TSteps> => {
  const workflowIds = Object.keys(workflow.steps)
  const promptIds = Object.keys(prompts.steps)
  const missing = workflowIds.filter((id) => !(id in prompts.steps))
  const unknown = promptIds.filter((id) => !(id in workflow.steps))
  if (missing.length > 0 || unknown.length > 0) {
    throw new Error(
      `Prompts do not match the workflow's steps${
        missing.length > 0 ? `; missing: ${missing.join(", ")}` : ""
      }${unknown.length > 0 ? `; unknown: ${unknown.join(", ")}` : ""}`
    )
  }
  const tunedSteps: StepMap = {}
  for (const step of tunableSteps(workflow)) {
    const saved = prompts.steps[step.id]
    if (!saved) {
      throw new Error(`Prompts lost step ${step.id}`)
    }
    const tuned = step.clone()
    tuned.description = saved.description
    tuned.examples = structuredClone(saved.examples)
    tunedSteps[step.id] = tuned
  }
  // SAFETY: same invariant as `programToWorkflow` — one entry per key of
  // `workflow.steps`, so the map carries precisely the keys `TSteps` declares.
  return { steps: tunedSteps as TSteps }
}
