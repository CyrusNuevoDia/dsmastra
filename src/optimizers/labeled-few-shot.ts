import { labeledFewShotProgram } from "@/optimizers/bootstrap"
import {
  programToWorkflow,
  promptsOf,
  workflowToProgram,
} from "@/optimizers/utils"
import type { SavePrompts } from "@/optimizers/utils"
import type { Example } from "@/program"
import type { CommittedWorkflow, StepMap } from "@/workflow"

export type LabeledFewShotConfig = {
  maxFewShotExamples?: number
  savePrompts: SavePrompts
  trainingSet: readonly Example[]
}

/**
 * Install up to `maxFewShotExamples` labeled trainingSet examples as few-shot
 * examples on every step (dspy.teleprompt.vanilla.LabeledFewShot). No LM
 * calls are made.
 */
export const labeledFewShot = async <TSteps extends StepMap>(
  workflow: CommittedWorkflow<TSteps>,
  config: LabeledFewShotConfig
): Promise<{ workflow: CommittedWorkflow<TSteps> }> => {
  const compiled = labeledFewShotProgram(
    workflowToProgram(workflow),
    [...config.trainingSet],
    config.maxFewShotExamples ?? 16
  )
  await config.savePrompts(promptsOf(compiled))
  return { workflow: programToWorkflow(compiled, workflow) }
}
