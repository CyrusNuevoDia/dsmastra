import type { Fields } from "@/fields"
import type { AnyTunableStep, RunContext } from "@/step"

export type { Example } from "@/step"

export type ProgramForward<TInput, TOutput> = (
  call: (stepId: string, inputData: Fields) => Promise<Fields>,
  inputData: TInput
) => Promise<TOutput>

/**
 * The unit the optimizers work on internally: a set of tunable steps plus a
 * forward function that wires them together. Cloning deep-copies step prompt
 * state (description, examples) while sharing the forward function and models.
 */
export type Program<TInput = Fields, TOutput = Fields> = {
  clone: () => Program<TInput, TOutput>
  code: string
  run: (inputData: TInput, ctx?: RunContext) => Promise<TOutput>
  steps: AnyTunableStep[]
}

export const createProgram = <TInput, TOutput>(config: {
  forward: ProgramForward<TInput, TOutput>
  steps: AnyTunableStep[]
}): Program<TInput, TOutput> => {
  const { forward } = config
  const code = forward.toString()

  const make = (steps: AnyTunableStep[]): Program<TInput, TOutput> => {
    const byId = new Map(steps.map((step) => [step.id, step]))
    return {
      clone: () => make(steps.map((step) => step.clone())),
      code,
      run: (inputData, ctx) => {
        const call = async (stepId: string, stepInputData: Fields) => {
          const step = byId.get(stepId)
          if (!step) {
            throw new Error(`Unknown step: ${stepId}`)
          }
          return await step.execute({ inputData: stepInputData }, ctx)
        }
        return forward(call, inputData)
      },
      steps,
    }
  }

  return make(config.steps)
}
