import type { AnyWorkflow } from "@mastra/core/workflows"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"
import { z } from "zod"

import { at } from "../collections"
import type { BootstrapMetric } from "../optimizers/bootstrap"
import {
  BOOTSTRAP_DEFAULT_MAX_ERRORS,
  harvestTraceExamples,
  installTrainExamples,
  prepareStudentAndTeacher,
  resetCopy,
  runBootstrapAttempt,
} from "../optimizers/bootstrap"
import {
  compiledSchema,
  exampleSchema,
  finishingSteps,
  optimizerResultSchema,
  promptsOf,
  promptsSchema,
} from "../optimizers/utils"
import type { OptimizerCheckpoint, SavePrompts } from "../optimizers/utils"
import { resolveScorer, scorerMetric, trajectoryScorerMetric } from "../scorers"
import type { ScorerRef } from "../scorers"
import type { Example } from "../step"

export type BootstrapFewShotConfig = {
  /** Pause hook: called before every teacher attempt; returning true suspends
   * the run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint
  /** Optional trajectory gate: a Mastra `type: "trajectory"` scorer (or its
   * registration key) that sees each teacher rollout as a Trajectory in
   * `run.output` — one workflow_step entry per engine-executed step (agents,
   * tools, and plain steps included; a nested workflow is a single entry;
   * loop iterations collapse to one entry per step id). Per-step inputs are
   * not recorded — the workflow input sits at
   * `rawWorkflowResult.stepResults.input`, and each later step's input is the
   * prior entry's output. The gate decides demo acceptance
   * in place of the objective scorer. Accepted when its score reaches
   * `threshold`; with no threshold, any score above zero. A gate throw counts
   * toward maxErrors, same as a rollout failure. */
  gate?: { scorer: ScorerRef; threshold?: number }
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number
  maxFewShotExamples?: number
  maxLabeledExamples?: number
  maxRounds?: number
  savePrompts: SavePrompts
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Scores the compiled workflow, and — unless
   * a `gate` is set — also gates teacher-trace acceptance; a scorer whose
   * generateScore returns 1 accepts every trace. */
  scorer: ScorerRef
  /** Accept a teacher trace when the objective score reaches this; default:
   * score > 0. Only meaningful without a `gate` (set `gate.threshold` there). */
  scoreThreshold?: number
  teacher?: AnyWorkflow
  trainingSet: readonly Example[]
  teacherSettings?: { model?: LanguageModel; temperature?: number }
}

/** The loop's whole world between attempts, as JSON. */
const bootstrapStateSchema = z.object({
  bootstrapped: z.array(z.number()),
  errorCount: z.number(),
  exampleIdx: z.number(),
  id2traces: z.record(z.string(), z.array(exampleSchema)),
  iteration: z.number(),
  roundIdx: z.number(),
  teacherPrompts: promptsSchema,
})

type BootstrapState = z.infer<typeof bootstrapStateSchema>

/**
 * BootstrapFewShot (dspy.teleprompt.bootstrap.BootstrapFewShot) as a Mastra
 * workflow over the target `workflow`: a prepare step compiles the teacher's
 * prompt state, a durable dountil loop runs ONE teacher attempt per
 * iteration — capturing the trace of every scorer-passing run as few-shot
 * examples per step — and a compile step backfills the remaining slots with
 * labeled examples before the shared save/evaluate/apply tail. All inter-step
 * state is JSON (teacher prompts, harvested traces, counters), so a
 * storage-backed run resumes mid-bootstrap without redoing completed attempts.
 */
export const createBootstrapFewShotWorkflow = (
  workflow: AnyWorkflow,
  config: BootstrapFewShotConfig
) => {
  const {
    checkpoint,
    gate,
    maxErrors = BOOTSTRAP_DEFAULT_MAX_ERRORS,
    maxFewShotExamples = 4,
    maxLabeledExamples = 16,
    maxRounds = 1,
    scoreThreshold,
    teacher,
    trainingSet,
  } = config
  if (gate && scoreThreshold !== undefined) {
    throw new Error(
      "scoreThreshold gates the objective scorer, which does not gate acceptance when a gate is set — use gate.threshold instead"
    )
  }
  const metric = scorerMetric(resolveScorer(workflow, config.scorer))
  const gateMetric =
    gate && trajectoryScorerMetric(resolveScorer(workflow, gate.scorer))
  const acceptThreshold = gate ? gate.threshold : scoreThreshold
  const attemptMetric: BootstrapMetric = gateMetric
    ? (gold, _prediction, _trace, ctx) => {
        if (!ctx?.trajectory) {
          throw new Error(
            "Gate scorer has no trajectory to score (the teacher rollout did not run through Mastra's engine)"
          )
        }
        return gateMetric(gold, ctx.trajectory)
      }
    : (gold, prediction) => metric(gold, prediction ?? undefined)

  const done = (state: BootstrapState): boolean =>
    state.bootstrapped.length >= maxFewShotExamples ||
    state.exampleIdx >= trainingSet.length

  const prepare = createStep({
    description:
      "Compile the teacher's prompt state (labeled few-shot install)",
    execute: () => {
      // Validates student/teacher structure and installs the labeled
      // examples on the teacher snapshot used by every attempt.
      const prepared = prepareStudentAndTeacher(
        promptsOf(workflow),
        [...trainingSet],
        {
          maxLabeledExamples,
          teacherPrompts: teacher && promptsOf(teacher),
        }
      )
      return Promise.resolve({
        bootstrapped: [],
        errorCount: 0,
        exampleIdx: 0,
        id2traces: Object.fromEntries(
          Object.keys(prepared.student.steps).map((stepId) => [stepId, []])
        ),
        iteration: 0,
        roundIdx: 0,
        teacherPrompts: prepared.teacher,
      } satisfies BootstrapState)
    },
    id: "prepare",
    inputSchema: z.object({}),
    outputSchema: bootstrapStateSchema,
  })

  const attempt = createStep({
    description: "One teacher attempt: rollout, gate, harvest on success",
    execute: async ({ inputData, resumeData, suspend }) => {
      const state: BootstrapState = inputData
      if (done(state)) {
        // The dountil body runs at least once even when there is nothing to
        // bootstrap; make that first iteration a no-op.
        return state
      }
      if (!resumeData && (await checkpoint?.({ iteration: state.iteration }))) {
        return await suspend({ iteration: state.iteration })
      }
      const example = at([...trainingSet], state.exampleIdx, "trainingSet")
      const next: BootstrapState = {
        ...state,
        id2traces: structuredClone(state.id2traces),
        iteration: state.iteration + 1,
      }
      let success = false
      try {
        const result = await runBootstrapAttempt(
          teacher ?? workflow,
          state.teacherPrompts,
          example,
          state.roundIdx,
          {
            metric: attemptMetric,
            ...(acceptThreshold !== undefined && {
              metricThreshold: acceptThreshold,
            }),
            teacherSettings: config.teacherSettings,
          }
        )
        ;({ success } = result)
        if (success) {
          const id2traces = new Map(Object.entries(next.id2traces))
          harvestTraceExamples(id2traces, result.trace)
          next.id2traces = Object.fromEntries(id2traces)
        }
      } catch (error) {
        next.errorCount += 1
        if (next.errorCount >= maxErrors) {
          throw error
        }
        console.error(`Failed to run or evaluate example due to ${error}.`)
      }
      if (success) {
        next.bootstrapped = [...next.bootstrapped, state.exampleIdx]
        next.exampleIdx += 1
        next.roundIdx = 0
      } else if (state.roundIdx + 1 >= maxRounds) {
        next.exampleIdx += 1
        next.roundIdx = 0
      } else {
        next.roundIdx = state.roundIdx + 1
      }
      return next
    },
    id: "attempt",
    inputSchema: bootstrapStateSchema,
    outputSchema: bootstrapStateSchema,
    resumeSchema: z.object({}),
    suspendSchema: z.object({ iteration: z.number() }),
  })

  const compile = createStep({
    description: "Install harvested examples plus labeled backfill",
    execute: ({ inputData }) => {
      const state: BootstrapState = inputData
      const bootstrapped = new Set(state.bootstrapped)
      const prompts = installTrainExamples(
        // A fresh reset copy: base descriptions, no examples.
        resetCopy(promptsOf(workflow)),
        new Map(Object.entries(state.id2traces)),
        trainingSet.filter((_x, idx) => !bootstrapped.has(idx)),
        { maxFewShotExamples, maxLabeledExamples }
      )
      return Promise.resolve({ prompts })
    },
    id: "compile",
    inputSchema: bootstrapStateSchema,
    outputSchema: compiledSchema,
  })

  const { apply, evaluate, save } = finishingSteps(workflow, {
    metric,
    savePrompts: config.savePrompts,
    trainingSet,
  })

  /* oxlint-disable promise/prefer-await-to-then, promise/no-return-wrap -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  return createWorkflow({
    id: `${workflow.id}.bootstrap-few-shot`,
    inputSchema: z.object({}),
    outputSchema: optimizerResultSchema,
  })
    .then(prepare)
    .dountil(attempt, ({ inputData }) => Promise.resolve(done(inputData)))
    .then(compile)
    .then(save)
    .then(evaluate)
    .then(apply)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then, promise/no-return-wrap */
}
