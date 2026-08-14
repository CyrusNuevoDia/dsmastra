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
import { resolveScorer, scorerMetric } from "@/scorers"
import type { ScorerRef } from "@/scorers"

export type BootstrapFewShotConfig = {
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number
  maxFewShotExamples?: number
  maxLabeledExamples?: number
  maxRounds?: number
  savePrompts: SavePrompts
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Gates teacher-trace acceptance AND scores
   * the compiled workflow; a scorer whose generateScore returns 1 accepts
   * every trace. */
  scorer: ScorerRef
  /** Accept a teacher trace when the score reaches this; default: score > 0. */
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
    savePrompts,
    scorer,
    scoreThreshold,
    teacher,
    trainingSet,
    ...options
  } = config
  const metric = scorerMetric(resolveScorer(workflow, scorer))
  const compiled = await bootstrapFewShotProgram(
    workflowToProgram(workflow),
    [...trainingSet],
    {
      ...options,
      metric: (gold, prediction) => metric(gold, prediction ?? undefined),
      ...(scoreThreshold !== undefined && { metricThreshold: scoreThreshold }),
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
