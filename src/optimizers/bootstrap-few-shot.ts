import type { AnyWorkflow } from "@mastra/core/workflows"
import type { LanguageModel } from "ai"

import { bootstrapFewShotProgram } from "@/optimizers/bootstrap"
import {
  applyProgram,
  evaluateProgram,
  promptsOf,
  workflowToProgram,
} from "@/optimizers/utils"
import type { Prompts, SavePrompts } from "@/optimizers/utils"
import type { Example } from "@/program"
import { resolveScorer, scorerMetric, trajectoryScorerMetric } from "@/scorers"
import type { ScorerRef } from "@/scorers"

export type BootstrapFewShotConfig = {
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

/**
 * Run a teacher over the trainingSet, capture the trace of every
 * scorer-passing run as few-shot examples per step, and backfill the remaining
 * slots with labeled examples (dspy.teleprompt.bootstrap.BootstrapFewShot).
 */
export const bootstrapFewShot = async (
  workflow: AnyWorkflow,
  config: BootstrapFewShotConfig
): Promise<{ candidates: [Prompts, { score: number }][]; score: number }> => {
  const {
    gate,
    savePrompts,
    scorer,
    scoreThreshold,
    teacher,
    trainingSet,
    ...options
  } = config
  if (gate && scoreThreshold !== undefined) {
    throw new Error(
      "scoreThreshold gates the objective scorer, which does not gate acceptance when a gate is set — use gate.threshold instead"
    )
  }
  const metric = scorerMetric(resolveScorer(workflow, scorer))
  const gateMetric =
    gate && trajectoryScorerMetric(resolveScorer(workflow, gate.scorer))
  const acceptThreshold = gate ? gate.threshold : scoreThreshold
  const compiled = await bootstrapFewShotProgram(
    workflowToProgram(workflow),
    [...trainingSet],
    {
      ...options,
      metric: gateMetric
        ? (gold, _prediction, _trace, ctx) => {
            if (!ctx?.trajectory) {
              throw new Error(
                "Gate scorer has no trajectory to score (the teacher rollout did not run through Mastra's engine)"
              )
            }
            return gateMetric(gold, ctx.trajectory)
          }
        : (gold, prediction) => metric(gold, prediction ?? undefined),
      ...(acceptThreshold !== undefined && {
        metricThreshold: acceptThreshold,
      }),
      teacher: teacher && workflowToProgram(teacher),
    }
  )
  const prompts = promptsOf(compiled)
  await savePrompts(prompts)
  // Score the compiled program over the trainingSet.
  const score = await evaluateProgram(compiled, trainingSet, metric)
  // The compiled prompt state lands in place on the caller's workflow.
  applyProgram(workflow, compiled)
  return { candidates: [[prompts, { score }]], score }
}
