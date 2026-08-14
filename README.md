# dsmastra

DSPy-style prompt optimization for [Mastra](https://mastra.ai) workflows, built on the Vercel AI SDK. You declare workflow steps with instructions and zod schemas, hand the workflow a trainset and a metric, and an optimizer (SIMBA or GEPA, with BootstrapFewShot for demos) rewrites the instructions and picks few-shot examples until the workflow scores better. The tuned workflow comes back with the same step map and the same types as the one you passed in.

The optimizers are faithful TypeScript ports of DSPy's teleprompters — traced from `dspy/` source, with the exact flows documented in [`docs/simba.md`](docs/simba.md) and [`docs/gepa.md`](docs/gepa.md).

## Quickstart

```ts
import { z } from "zod"
import { openai } from "@ai-sdk/openai"
import { createWorkflow, declareStep, optimize, SIMBA } from "dsmastra"

const step = declareStep({
  id: "math",
  inputSchema: z.object({ question: z.string() }),
  outputSchema: z.object({ answer: z.string() }),
  instructions: "Answer the arithmetic question.", // underspecified on purpose
  model: openai("gpt-4.1-mini"),
  temperature: 0,
})

const workflow = createWorkflow({
  id: "wf-math",
  inputSchema: z.object({ question: z.string() }),
  outputSchema: z.object({ answer: z.string() }),
})
  .then(step)
  .commit()

const trainset = [
  { inputs: { question: "2+2" }, outputs: { answer: "four" } },
  { inputs: { question: "3+3" }, outputs: { answer: "six" } },
  // ...
]

const tuned = await optimize(
  SIMBA({ maxSteps: 3, maxDemos: 2, seed: 0 }),
  workflow,
  { trainset }
)

// Same shape and types as the input workflow, now with tuned
// instructions and demos baked into each step's predictor.
await tuned.steps.math.execute({ inputData: { question: "4+5" } })
```

The metric defaults to exact match on every expected output field. Pass your own for partial credit. A metric always returns an object with a `score`, and adding a `feedback` string tells the optimizer _why_ a prediction fell short — SIMBA's rule-writing and GEPA's reflection both read it:

```ts
const metric = (example, prediction) =>
  prediction?.answer === example.outputs.answer
    ? { score: 1 }
    : { score: 0.5, feedback: "Numerically right, but not spelled out." }
```

Any other fields you return ride along as metadata. The same `{ score }` contract covers all three metrics — SIMBA's, GEPA's, and BootstrapFewShot's — so one metric works everywhere. This is a deliberate divergence from DSPy, whose bootstrap metric returns `bool | float`; here a pass/fail metric writes `{ score: 1 }` / `{ score: 0 }`, and with no `metricThreshold` set any score above zero counts as a pass.

## The optimizers

- **SIMBA** (Stochastic Introspective Mini-Batch Ascent) — a mini-batch hill-climber. Each step it samples several stochastic rollouts per example, finds the examples with the biggest score spread between rollouts, and turns the best/worst contrast into either an appended few-shot demo or an appended natural-language rule, keeping whichever candidate wins on that minibatch. Needs a temperature-capable model, since rollout diversity is the signal. Details in [`docs/simba.md`](docs/simba.md).
- **GEPA** (Genetic-Pareto reflective prompt evolution) — evolves a pool of instruction candidates by reflective mutation: sample a parent from the Pareto frontier of per-example winners, show a reflection LM the traces plus metric feedback, and let it rewrite one predictor's instructions. GEPA is strictly instruction-only; when `maxDemos > 0`, `optimize` runs a BootstrapFewShot pre-pass first (DSPy-style teleprompter composition) so GEPA evolves instructions on a demo-carrying program. Details in [`docs/gepa.md`](docs/gepa.md).
- **BootstrapFewShot** — runs the current program as its own teacher over the trainset, keeps traces the metric accepts as bootstrapped demos, and backfills with raw labeled examples. Available directly as `bootstrapFewShot` / `labeledFewShot` for use on programs.

Both take the same config: `maxSteps`, `maxDemos`, optional `metric`, `bsize`, `numCandidates`, `seed`. For GEPA, `maxSteps` maps to full-trainset evaluations of the budget.

## Lower-level API

`optimize` is a thin wrapper over a program abstraction you can use directly when you're not in a Mastra workflow: `declarePredictor` builds a named, schema-typed LLM call (instructions + demos rendered into a `generateObject` prompt), and `createProgram` wires predictors together with a `forward` function. `simba(program, trainset, config)`, `gepa(program, trainset, config)`, and `bootstrapFewShot(program, trainset, config)` operate on programs; see `tests/*.int.test.ts` for worked examples.

## Development

Bun-only (see `.mise.toml` — run `mise trust && mise install` if you're missing tools):

```sh
bun install
just test             # unit tests (tests/unit), run concurrently
just test-int         # integration tests (tests/int); need OPENAI_API_KEY in .env
just check            # format check + lint + typecheck + unit tests
just fmt              # format
```

The `dspy/` directory is a reference clone of DSPy used for tracing the original optimizer implementations; it's not part of the package.
