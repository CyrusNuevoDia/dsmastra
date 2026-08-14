import type { LanguageModel } from "ai"

import { bootstrapFewShotProgram } from "@/optimizers/bootstrap"
import type { BootstrapMetric } from "@/optimizers/bootstrap"
import {
  programToWorkflow,
  promptsOf,
  workflowToProgram,
} from "@/optimizers/utils"
import type { SavePrompts } from "@/optimizers/utils"
import type { Example } from "@/program"
import type { CommittedWorkflow, StepMap } from "@/workflow"

export type { BootstrapMetric } from "@/optimizers/bootstrap"

export type BootstrapFewShotConfig = {
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number
  maxFewShotExamples?: number
  maxLabeledExamples?: number
  maxRounds?: number
  /** No metric means every teacher trace is accepted. */
  metric?: BootstrapMetric
  metricThreshold?: number
  savePrompts: SavePrompts
  teacher?: CommittedWorkflow
  trainingSet: readonly Example[]
  teacherSettings?: { model?: LanguageModel; temperature?: number }
}

/**
 * Run a teacher over the trainingSet, capture the trace of every metric-passing
 * run as few-shot examples per step, and backfill the remaining slots with
 * labeled examples (dspy.teleprompt.bootstrap.BootstrapFewShot).
 */
export const bootstrapFewShot = async <TSteps extends StepMap>(
  workflow: CommittedWorkflow<TSteps>,
  config: BootstrapFewShotConfig
): Promise<{ workflow: CommittedWorkflow<TSteps> }> => {
  const { savePrompts, teacher, trainingSet, ...options } = config
  const compiled = await bootstrapFewShotProgram(
    workflowToProgram(workflow),
    [...trainingSet],
    {
      ...options,
      teacher: teacher && workflowToProgram(teacher),
    }
  )
  await savePrompts(promptsOf(compiled))
  return { workflow: programToWorkflow(compiled, workflow) }
}
