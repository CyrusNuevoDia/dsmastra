import type { AnyWorkflow } from "@mastra/core/workflows"

import { labeledFewShotProgram } from "@/optimizers/bootstrap"
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

export type LabeledFewShotConfig = {
  maxFewShotExamples?: number
  savePrompts: SavePrompts
  /** Scores the compiled workflow over the trainingSet: a Mastra scorer, or
   * its registration key on the workflow's Mastra instance. */
  scorer: ScorerRef
  trainingSet: readonly Example[]
}

/**
 * Install up to `maxFewShotExamples` labeled trainingSet examples as few-shot
 * examples on every step (dspy.teleprompt.vanilla.LabeledFewShot). Compiling
 * makes no LM calls; scoring runs the compiled workflow once.
 */
export const labeledFewShot = async (
  workflow: AnyWorkflow,
  config: LabeledFewShotConfig
): Promise<{ candidates: [Prompts, { score: number }][]; score: number }> => {
  const compiled = labeledFewShotProgram(
    workflowToProgram(workflow),
    [...config.trainingSet],
    config.maxFewShotExamples ?? 16
  )
  const prompts = promptsOf(compiled)
  await config.savePrompts(prompts)
  // Compiling makes no LM calls, but scoring runs the compiled workflow over
  // the trainingSet once.
  const score = await evaluateProgram(
    compiled,
    config.trainingSet,
    scorerMetric(resolveScorer(workflow, config.scorer))
  )
  // The compiled prompt state lands in place on the caller's workflow.
  applyProgram(workflow, compiled)
  return { candidates: [[prompts, { score }]], score }
}
