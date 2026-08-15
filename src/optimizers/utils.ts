import { extractWorkflowTrajectory } from "@mastra/core/evals"
import { RequestContext } from "@mastra/core/request-context"
import type { AnyWorkflow, SingleStepEntry, Step } from "@mastra/core/workflows"

import type { Fields } from "@/fields"
import type { Program } from "@/program"
import type { Metric } from "@/scorers"
import type { AnyDeclarativeStep, Example, RunContext } from "@/step"
import { RUN_CONTEXT_KEY } from "@/step"

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

const isDeclarativeStep = (step: Step): step is Step & AnyDeclarativeStep =>
  "description" in step && "examples" in step && "clone" in step

const singleEntrySteps = (entry: SingleStepEntry): Step[] =>
  entry.type === "step" ? [entry.step] : []

/**
 * A workflow's declarative steps in graph order, read from Mastra's step graph.
 * The walk descends into parallel, branch, loop, and foreach entries; steps
 * that weren't built with `declareStep` (agents, tools, mappings, plain steps,
 * nested workflows) pass through untouched — the engine runs them, the
 * optimizers just don't tune them. Steps inside a nested workflow are opaque.
 */
export const declarativeSteps = (workflow: AnyWorkflow): AnyDeclarativeStep[] =>
  workflow.stepGraph
    .flatMap((entry) => {
      switch (entry.type) {
        case "parallel":
        case "conditional": {
          return entry.steps.flatMap(singleEntrySteps)
        }
        case "loop":
        case "foreach": {
          return singleEntrySteps(entry.step)
        }
        default: {
          return "step" in entry ? [entry.step] : []
        }
      }
    })
    .filter((step) => isDeclarativeStep(step))

// --- The candidate gate -------------------------------------------------------
//
// Engine rollouts execute the workflow's LIVE steps, so a candidate's prompt
// state must be installed on them before its rollouts run — and two candidates
// must never be in flight at once. The gate serializes across holders (one
// candidate's step array identifies it) while letting one holder's rollouts
// run concurrently.

type CandidateSteps = AnyDeclarativeStep[]

type Gate = {
  count: number
  holder: CandidateSteps | null
  wake: (() => void)[]
}

const gates = new WeakMap<AnyWorkflow, Gate>()

const acquire = async (
  workflow: AnyWorkflow,
  holder: CandidateSteps
): Promise<{ fresh: boolean; gate: Gate }> => {
  let gate = gates.get(workflow)
  if (!gate) {
    gate = { count: 0, holder: null, wake: [] }
    gates.set(workflow, gate)
  }
  while (gate.count > 0 && gate.holder !== holder) {
    const { promise, resolve } = Promise.withResolvers()
    gate.wake.push(resolve)
    // oxlint-disable-next-line no-await-in-loop -- waiting for the gate to clear is the point
    await promise
  }
  const fresh = gate.holder !== holder
  gate.holder = holder
  gate.count += 1
  return { fresh, gate }
}

const releaseGate = (gate: Gate): void => {
  gate.count -= 1
  if (gate.count === 0) {
    // Forget the holder: a candidate mutated between evaluation waves (SIMBA
    // strategies edit step state in place) must reinstall, not be trusted as
    // already current.
    gate.holder = null
    for (const wake of gate.wake.splice(0)) {
      wake()
    }
  }
}

/**
 * Wrap a workflow as a Program whose rollouts run through Mastra's engine —
 * any graph shape works, and every run shows up in Mastra observability. Each
 * clone carries its own prompt state (a candidate); `run` installs that state
 * onto the live steps under the gate, starts an engine run with the
 * RunContext smuggled through the request context, and maps a non-success
 * run to a throw (callers score it as a failure).
 */
export const workflowToProgram = (
  workflow: AnyWorkflow
): Program<Fields, Fields> => {
  const liveSteps = declarativeSteps(workflow)
  if (liveSteps.length === 0) {
    throw new Error(
      `Workflow ${workflow.id} has no declareStep steps to optimize`
    )
  }
  const code = `workflow ${workflow.id}: ${liveSteps
    .map((step) => step.id)
    .join(" -> ")}`

  const make = (steps: AnyDeclarativeStep[]): Program<Fields, Fields> => ({
    clone: () => make(steps.map((step) => step.clone())),
    code,
    run: async (inputData: Fields, ctx?: RunContext): Promise<Fields> => {
      const { fresh, gate } = await acquire(workflow, steps)
      try {
        if (fresh) {
          const byId = new Map(steps.map((step) => [step.id, step]))
          for (const live of liveSteps) {
            const candidate = byId.get(live.id)
            if (!candidate) {
              throw new Error(`Candidate program lost step ${live.id}`)
            }
            live.description = candidate.description
            live.examples = structuredClone(candidate.examples)
          }
        }
        // Live-eval scorers attached to steps are disabled for the rollout:
        // the optimizer scores through its own scorer, and letting both run
        // would double every evaluation (Mastra's runEvals does the same).
        const run = await workflow.createRun({ disableScorers: true })
        const result = await run.start({
          inputData,
          requestContext: new RequestContext([[RUN_CONTEXT_KEY, ctx]]),
        })
        if (result.status !== "success") {
          throw result.status === "failed" && result.error instanceof Error
            ? result.error
            : new Error(
                `Workflow ${workflow.id} run ended with status ${result.status}`
              )
        }
        if (ctx) {
          // Trace linkage for the scorer: scores attach to this rollout's
          // trace in Mastra observability when tracing is configured.
          ctx.target = { spanId: result.spanId, traceId: result.traceId }
          // Every engine-executed step — agents, tools, plain steps, nested
          // workflows as single entries — via Mastra's own extractor, the same
          // Trajectory its runEvals hands to trajectory scorers.
          ctx.trajectory = extractWorkflowTrajectory(
            result.steps,
            result.stepExecutionPath
          )
        }
        // SAFETY: a successful run's `result` was produced by the workflow's
        // final step and validated against its output schema by the engine;
        // `Fields` is the optimizers' untyped view of that same record.
        return result.result as Fields
      } finally {
        releaseGate(gate)
      }
    },
    steps,
  })

  return make(liveSteps.map((step) => step.clone()))
}

/**
 * Write a tuned program's prompt state back onto the workflow's live steps.
 * Mutates in place on purpose: the caller's own step references — and any
 * Mastra instance the workflow is registered with — see the tuned prompts
 * immediately, and the same workflow instance stays re-optimizable.
 */
export const applyProgram = <TWorkflow extends AnyWorkflow>(
  workflow: TWorkflow,
  program: Program<Fields, Fields>
): TWorkflow => {
  for (const step of declarativeSteps(workflow)) {
    const tuned = program.steps.find((candidate) => candidate.id === step.id)
    if (!tuned) {
      throw new Error(`Tuned program lost step ${step.id}`)
    }
    step.description = tuned.description
    step.examples = structuredClone(tuned.examples)
  }
  return workflow
}

/**
 * Mean metric score of a program over a set of examples, rollouts running
 * concurrently through the program (and so through Mastra's engine for
 * workflow-backed programs). A failed rollout or metric throw scores 0, same
 * as the optimizers' own rollout handling. Deliberate divergence from DSPy,
 * which leaves its few-shot compilers unscored (the evaluation in
 * `BootstrapFewShot._bootstrap` is commented out upstream).
 */
export const evaluateProgram = async (
  program: Program<Fields, Fields>,
  examples: readonly Example[],
  metric: Metric
): Promise<number> => {
  const scores = await Promise.all(
    examples.map(async (example) => {
      const ctx: RunContext = {}
      try {
        const prediction = await program.run(example.inputData, ctx)
        const { score } = await metric(example, prediction, ctx.target)
        return score
      } catch (error) {
        console.warn(error)
        return 0
      }
    })
  )
  return (
    scores.reduce((sum, score) => sum + score, 0) / Math.max(scores.length, 1)
  )
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
 * Apply saved prompts to a workflow's live steps, in place, and return the
 * same workflow. Throws when the prompts' step ids don't exactly match the
 * workflow's, so a stale snapshot fails loudly instead of half-applying.
 * Parsing and storage are the caller's problem.
 */
export const loadPrompts = <TWorkflow extends AnyWorkflow>(
  workflow: TWorkflow,
  prompts: Prompts
): TWorkflow => {
  const steps = declarativeSteps(workflow)
  const workflowIds = steps.map((step) => step.id)
  const promptIds = Object.keys(prompts.steps)
  const missing = workflowIds.filter((id) => !(id in prompts.steps))
  const unknown = promptIds.filter((id) => !workflowIds.includes(id))
  if (missing.length > 0 || unknown.length > 0) {
    throw new Error(
      `Prompts do not match the workflow's steps${
        missing.length > 0 ? `; missing: ${missing.join(", ")}` : ""
      }${unknown.length > 0 ? `; unknown: ${unknown.join(", ")}` : ""}`
    )
  }
  for (const step of steps) {
    const saved = prompts.steps[step.id]
    if (!saved) {
      throw new Error(`Prompts lost step ${step.id}`)
    }
    step.description = saved.description
    step.examples = structuredClone(saved.examples)
  }
  return workflow
}
