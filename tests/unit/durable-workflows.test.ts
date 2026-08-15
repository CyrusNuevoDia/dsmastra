import { expect, test } from "bun:test"

import { Mastra } from "@mastra/core"
import { InMemoryStore } from "@mastra/core/storage"
import { createWorkflow } from "@mastra/core/workflows"
import type { AnyWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { createBootstrapFewShotWorkflow } from "../../src/optimizers/bootstrap-few-shot"
import { createGEPAWorkflow } from "../../src/optimizers/gepa"
import { createLabeledFewShotWorkflow } from "../../src/optimizers/labeled-few-shot"
import { createSIMBAWorkflow } from "../../src/optimizers/simba"
import { fakeScorer, fakeStep } from "./helpers"

const fieldsSchema = z.record(z.string(), z.unknown())

const makeTarget = () => {
  const log: { inputData: Record<string, unknown> }[] = []
  const step = fakeStep(
    "solve",
    (inputData) => ({ y: (inputData.x as number) * 2 }),
    log
  )
  const workflow = createWorkflow({
    id: "target-wf",
    inputSchema: fieldsSchema,
    outputSchema: fieldsSchema,
  })
    .then(step)
    .commit()
  return { log, step, workflow }
}

const trainingSet = [
  { inputData: { x: 1 }, outputData: { y: 2 } },
  { inputData: { x: 2 }, outputData: { y: 4 } },
]

const matchScorer = () =>
  fakeScorer((gold, prediction) =>
    prediction?.y === gold.outputData.y ? 1 : 0
  )

const savePrompts = () => Promise.resolve()

const makeOptimizers = (): Record<string, AnyWorkflow> => ({
  bootstrap: createBootstrapFewShotWorkflow(makeTarget().workflow, {
    savePrompts,
    scorer: matchScorer(),
    trainingSet,
  }),
  gepa: createGEPAWorkflow(makeTarget().workflow, {
    maxScorerCalls: 8,
    reflectionModel: () => Promise.resolve("```tuned instruction```"),
    savePrompts,
    scorer: matchScorer(),
    trainingSet,
  }),
  labeled: createLabeledFewShotWorkflow(makeTarget().workflow, {
    savePrompts,
    scorer: matchScorer(),
    trainingSet,
  }),
  simba: createSIMBAWorkflow(makeTarget().workflow, {
    batchSize: 2,
    candidates: 2,
    maxSteps: 1,
    promptModel: "stub" as never,
    savePrompts,
    scorer: matchScorer(),
    trainingSet,
  }),
})

// Loop entries wrap their body one level deeper: { type: "loop", step:
// { type: "step", step } } — unwrap to the actual step or nested workflow.
type GraphEntry = {
  step?: { id?: string; step?: { id: string } }
  type?: string
}

const stepOf = (entry: GraphEntry | undefined): { id: string } | undefined => {
  const outer = entry?.step
  if (!outer) {
    return undefined
  }
  // SAFETY: a non-loop entry's `step` IS the step itself, so when there is no
  // inner `step` the outer object carries the id.
  return outer.step ?? (outer as { id: string })
}

const graphOf = (workflow: AnyWorkflow) =>
  workflow.stepGraph.map((entry) => ({
    id: stepOf(entry)?.id,
    type: entry.type,
  }))

const loopBodyIds = (workflow: AnyWorkflow): string[] => {
  const loop = workflow.stepGraph.find((entry) => entry.type === "loop")
  const body = stepOf(loop) as AnyWorkflow | undefined
  if (!body?.stepGraph) {
    return []
  }
  return body.stepGraph.flatMap((entry) => {
    const inner = stepOf(entry)
    return inner ? [inner.id] : []
  })
}

test("every optimizer decomposes into real workflow steps with loop control flow", () => {
  const optimizers = makeOptimizers()

  expect(graphOf(optimizers.labeled as AnyWorkflow)).toEqual([
    { id: "compile", type: "step" },
    { id: "save", type: "step" },
    { id: "evaluate", type: "step" },
    { id: "apply", type: "step" },
  ])

  expect(graphOf(optimizers.bootstrap as AnyWorkflow)).toEqual([
    { id: "prepare", type: "step" },
    { id: "attempt", type: "loop" },
    { id: "compile", type: "step" },
    { id: "save", type: "step" },
    { id: "evaluate", type: "step" },
    { id: "apply", type: "step" },
  ])

  // The search optimizers' loop bodies are nested workflows whose LM-call
  // phases are dedicated steps — GEPA's reflection proposal and SIMBA's
  // introspective offerFeedback each get their own step (and so their own
  // span in observability).
  expect(graphOf(optimizers.gepa as AnyWorkflow)).toEqual([
    { id: "prepass", type: "step" },
    { id: "seed-eval", type: "step" },
    { id: "iteration", type: "loop" },
    { id: "finalize", type: "step" },
  ])
  expect(loopBodyIds(optimizers.gepa as AnyWorkflow)).toEqual([
    "reflect",
    "propose",
    "accept",
  ])

  expect(graphOf(optimizers.simba as AnyWorkflow)).toEqual([
    { id: "init", type: "step" },
    { id: "iteration", type: "loop" },
    { id: "finalize", type: "step" },
  ])
  expect(loopBodyIds(optimizers.simba as AnyWorkflow)).toEqual([
    "rollout",
    "propose-candidates",
    "score-candidates",
  ])
})

/** Collect every workflow-step-result event's (id, output) during a run. */
const runWatched = async (optimizer: AnyWorkflow) => {
  const run = await optimizer.createRun()
  const results: { id: string; output: unknown }[] = []
  run.watch((event) => {
    // SAFETY: watch events are untyped on this surface; the guard below only
    // reads fields after checking the event type discriminant.
    const typed = event as {
      type: string
      payload?: { id?: string; output?: unknown }
    }
    if (typed.type === "workflow-step-result" && typed.payload?.id) {
      results.push({ id: typed.payload.id, output: typed.payload.output })
    }
  })
  const result = await run.start({ inputData: {} })
  expect(result.status).toBe("success")
  return results
}

test("optimizer LM phases run as their own steps, and every step's output survives a JSON round-trip", async () => {
  const optimizers = makeOptimizers()

  for (const [name, optimizer] of Object.entries(optimizers)) {
    // oxlint-disable-next-line no-await-in-loop -- sequential on purpose: each optimizer run is asserted independently
    const results = await runWatched(optimizer)
    const ids = new Set(results.map((r) => r.id))

    // Serializable step IO: everything that crossed a step boundary — loop
    // iterations included — is pure JSON. (JSON round-trips drop undefined
    // properties; toEqual treats those as absent, which is the point.)
    for (const { id, output } of results) {
      if (output === undefined) {
        continue
      }
      expect(
        // oxlint-disable-next-line unicorn/prefer-structured-clone -- the JSON round-trip IS what this asserts
        JSON.parse(JSON.stringify(output)),
        `${name} step ${id} output must be JSON-safe`
      ).toEqual(output)
    }

    // The LM-call phases are identifiable steps of the run.
    if (name === "gepa") {
      expect(ids.has("iteration.propose")).toBe(true)
    }
    if (name === "simba") {
      expect(ids.has("iteration.propose-candidates")).toBe(true)
    }
    if (name === "bootstrap") {
      expect(ids.has("attempt")).toBe(true)
    }
  }
})

test("a checkpointed run suspends durably and resumes by runId without redoing completed attempts", async () => {
  const { log, workflow } = makeTarget()
  let suspensions = 0
  const optimizer = createBootstrapFewShotWorkflow(workflow, {
    checkpoint: ({ iteration }) => {
      if (iteration === 1 && suspensions === 0) {
        suspensions += 1
        return true
      }
      return false
    },
    savePrompts,
    scorer: matchScorer(),
    trainingSet,
  })
  const mastra = new Mastra({
    logger: false,
    storage: new InMemoryStore(),
    workflows: { optimizer },
  })
  const registered = mastra.getWorkflow("optimizer")

  const run = await registered.createRun()
  const { runId } = run
  const first = await run.start({ inputData: {} })
  expect(first.status).toBe("suspended")
  // Exactly the first teacher attempt ran before the suspension.
  const attemptsBeforeResume = log.length
  expect(attemptsBeforeResume).toBe(1)

  // A fresh handle by runId — what a restarted process would do.
  const rehydrated = await registered.createRun({ runId })
  const resumed = await rehydrated.resume({ resumeData: {} })
  expect(resumed.status).toBe("success")
  if (resumed.status === "success") {
    const { candidates, score } = resumed.result as {
      candidates: unknown[]
      score: number
    }
    expect(score).toBe(1)
    expect(candidates).toHaveLength(1)
  }
  // Both examples bootstrap (2 attempts) and the final evaluation rolls the
  // compiled workflow over the trainingSet (2 runs). The completed first
  // attempt was NOT re-executed on resume: 4 total, not 5.
  expect(log.length).toBe(4)
})
