import type { AnyPredictor, RunContext } from "@/predictor"

export type ProgramForward<TInput, TOutput> = (
  call: (
    predictorName: string,
    inputs: Record<string, unknown>
  ) => Promise<Record<string, unknown>>,
  input: TInput
) => Promise<TOutput>

/**
 * The unit SIMBA optimizes: a set of named predictors plus a forward function
 * that wires them together. Cloning deep-copies predictor state (instructions,
 * demos) while sharing the forward function and models.
 */
export type Program<
  TInput = Record<string, unknown>,
  TOutput = Record<string, unknown>,
> = {
  clone: () => Program<TInput, TOutput>
  code: string
  predictors: AnyPredictor[]
  run: (input: TInput, ctx?: RunContext) => Promise<TOutput>
}

export function createProgram<TInput, TOutput>(config: {
  forward: ProgramForward<TInput, TOutput>
  predictors: AnyPredictor[]
}): Program<TInput, TOutput> {
  const { forward, predictors } = config
  return {
    clone: () =>
      createProgram({
        forward,
        predictors: predictors.map((predictor) => predictor.clone()),
      }),
    code: forward.toString(),
    predictors,
    run: (input, ctx) => {
      const call = async (
        predictorName: string,
        inputs: Record<string, unknown>
      ) => {
        const predictor = predictors.find((p) => p.name === predictorName)
        if (!predictor) {
          throw new Error(`Unknown predictor: ${predictorName}`)
        }
        return (await predictor.call(inputs as never, ctx)) as Record<
          string,
          unknown
        >
      }
      return forward(call, input)
    },
  }
}
