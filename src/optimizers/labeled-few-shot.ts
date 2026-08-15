import type { AnyWorkflow } from "@mastra/core/workflows"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import { z } from "zod"

import { labeledFewShotPrompts } from "../optimizers/bootstrap"
import {
  compiledSchema,
  finishingSteps,
  optimizerResultSchema,
  promptsOf,
} from "../optimizers/utils"
import type { SavePrompts } from "../optimizers/utils"
import { resolveScorer, scorerMetric } from "../scorers"
import type { ScorerRef } from "../scorers"
import type { Example } from "../step"

export type LabeledFewShotConfig = {
  maxFewShotExamples?: number
  savePrompts: SavePrompts
  /** Scores the compiled workflow over the trainingSet: a Mastra scorer, or
   * its registration key on the workflow's Mastra instance. */
  scorer: ScorerRef
  trainingSet: readonly Example[]
}

/**
 * LabeledFewShot as a Mastra workflow over the target `workflow`: install up
 * to `maxFewShotExamples` labeled trainingSet examples as few-shot examples
 * on every step (dspy.teleprompt.vanilla.LabeledFewShot). Compiling makes no
 * LM calls; the evaluate step runs the compiled workflow over the trainingSet
 * once, and the apply step lands the compiled prompt state in place on the
 * target workflow. All inter-step state is JSON, so a storage-backed run is
 * durable and observable step by step.
 */
export const createLabeledFewShotWorkflow = (
  workflow: AnyWorkflow,
  config: LabeledFewShotConfig
) => {
  const compile = createStep({
    description: "Install labeled examples as few-shot examples on every step",
    execute: () =>
      Promise.resolve({
        prompts: labeledFewShotPrompts(
          promptsOf(workflow),
          [...config.trainingSet],
          config.maxFewShotExamples ?? 16
        ),
      }),
    id: "compile",
    inputSchema: z.object({}),
    outputSchema: compiledSchema,
  })

  const { apply, evaluate, save } = finishingSteps(workflow, {
    metric: scorerMetric(resolveScorer(workflow, config.scorer)),
    savePrompts: config.savePrompts,
    trainingSet: config.trainingSet,
  })

  /* oxlint-disable promise/prefer-await-to-then -- Mastra's workflow builder chains `.then(step)`: these are graph edges, not promise continuations */
  return createWorkflow({
    id: `${workflow.id}.labeled-few-shot`,
    inputSchema: z.object({}),
    outputSchema: optimizerResultSchema,
  })
    .then(compile)
    .then(save)
    .then(evaluate)
    .then(apply)
    .commit()
  /* oxlint-enable promise/prefer-await-to-then */
}
