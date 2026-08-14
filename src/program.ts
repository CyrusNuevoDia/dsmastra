import type { Fields } from "@/fields"
import type { AnyPredictor, RunContext } from "@/predictor"

/**
 * A training example: inputs plus the expected output fields.
 *
 * Parameterised by the program it trains so that `program.run(example.inputs)`
 * typechecks on its own. Both sides default to `Fields`, which is what a program
 * built from named predictors actually consumes, so plain `Example` still means
 * what it always did.
 */
export type Example<TInput = Fields, TOutput = Fields> = {
  inputs: TInput
  outputs: TOutput
}

export type ProgramForward<TInput, TOutput> = (
  call: (predictorName: string, inputs: Fields) => Promise<Fields>,
  input: TInput
) => Promise<TOutput>

/**
 * The unit SIMBA optimizes: a set of named predictors plus a forward function
 * that wires them together. Cloning deep-copies predictor state (instructions,
 * demos) while sharing the forward function and models.
 */
export type Program<TInput = Fields, TOutput = Fields> = {
  clone: () => Program<TInput, TOutput>
  code: string
  predictors: AnyPredictor[]
  run: (input: TInput, ctx?: RunContext) => Promise<TOutput>
}

export const createProgram = <TInput, TOutput>(config: {
  forward: ProgramForward<TInput, TOutput>
  predictors: AnyPredictor[]
}): Program<TInput, TOutput> => {
  const { forward } = config
  const code = forward.toString()

  const make = (predictors: AnyPredictor[]): Program<TInput, TOutput> => {
    const byName = new Map(predictors.map((p) => [p.name, p]))
    return {
      clone: () => make(predictors.map((predictor) => predictor.clone())),
      code,
      predictors,
      run: (input, ctx) => {
        const call = async (predictorName: string, inputs: Fields) => {
          const predictor = byName.get(predictorName)
          if (!predictor) {
            throw new Error(`Unknown predictor: ${predictorName}`)
          }
          return await predictor.call(inputs, ctx)
        }
        return forward(call, input)
      },
    }
  }

  return make(config.predictors)
}
