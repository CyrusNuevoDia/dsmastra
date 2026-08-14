# dsmastra

DSPy-style prompt optimization for [Mastra](https://mastra.ai) workflows, built on the Vercel AI SDK. You declare workflow steps with a description and zod schemas, hand the workflow a training set and a metric, and an optimizer (`simba`, `gepa`, `bootstrapFewShot`, or `labeledFewShot`) rewrites each step's description and picks few-shot examples until the workflow scores better. The tuned workflow comes back with the same step map and the same types as the one you passed in.

The optimizers are faithful TypeScript ports of DSPy's teleprompters — traced from `dspy/` source, with the exact flows documented in [`docs/simba.md`](docs/simba.md) and [`docs/gepa.md`](docs/gepa.md).

## Quickstart

```ts
import { z } from "zod"
import { openai } from "@ai-sdk/openai"
import { createWorkflow, declareStep, simba } from "dsmastra"
import { writeFile } from "node:fs/promises"

const step = declareStep({
  id: "math",
  description: "Answer the arithmetic question.", // underspecified on purpose
  inputSchema: z.object({ question: z.string() }),
  outputSchema: z.object({ answer: z.string() }),
  model: openai("gpt-4.1-mini"),
  temperature: 0, // AI SDK call settings (seed, topP, topK, …) are first-class
})

const workflow = createWorkflow({
  id: "wf-math",
  inputSchema: z.object({ question: z.string() }),
  outputSchema: z.object({ answer: z.string() }),
})
  .then(step)
  .commit()

const trainingSet = [
  { inputData: { question: "2+2" }, outputData: { answer: "four" } },
  { inputData: { question: "3+3" }, outputData: { answer: "six" } },
  // ...
]

const { workflow: tuned, score } = await simba(workflow, {
  trainingSet,
  maxSteps: 3,
  maxFewShotExamples: 2,
  seed: 0,
  savePrompts: (prompts) => writeFile("prompts.json", JSON.stringify(prompts)),
})

// Same shape and types as the input workflow, now with a tuned description
// and few-shot examples on each step.
await tuned.steps.math.execute({ inputData: { question: "4+5" } })
```

## Saving and loading tuned prompts

Every optimizer requires `savePrompts` — you can't forget to persist a run. It's called with the current best prompts each time the optimizer improves on them (crash-safe, last write wins) and once more on completion. The payload is plain JSON: `{ version: 1, steps: { [stepId]: { description, examples } } }`. Storage is yours; rehydrating is one call:

```ts
import { loadPrompts } from "dsmastra"

const restored = loadPrompts(
  workflow,
  JSON.parse(await readFile("prompts.json", "utf8"))
)
```

`loadPrompts` returns a new workflow (the input is never mutated) and throws if the saved step ids don't exactly match the workflow's.

## Metrics

The metric defaults to exact match on every expected output field. Pass your own for partial credit. A metric always returns an object with a `score`, and adding a `feedback` string tells the optimizer _why_ a prediction fell short — SIMBA's rule-writing and GEPA's reflection both read it:

```ts
const metric = (example, prediction) =>
  prediction?.answer === example.outputData.answer
    ? { score: 1 }
    : { score: 0.5, feedback: "Numerically right, but not spelled out." }
```

Any other fields you return ride along as metadata. The same `{ score }` contract covers every optimizer, so one metric works everywhere. This is a deliberate divergence from DSPy, whose bootstrap metric returns `bool | float`; here a pass/fail metric writes `{ score: 1 }` / `{ score: 0 }`, and with no `metricThreshold` set any score above zero counts as a pass.

## The optimizers

All four share one signature: `optimizer(workflow, config)` — the training set rides in the config as `trainingSet`.

- **`simba`** (Stochastic Introspective Mini-Batch Ascent) — a mini-batch hill-climber. Each step it samples several stochastic rollouts per example, finds the examples with the biggest score spread between rollouts, and turns the best/worst contrast into either an appended few-shot example or an appended natural-language rule, keeping whichever candidate wins on that minibatch. Needs a temperature-capable model, since rollout diversity is the signal. Config: `maxSteps`, `batchSize`, `candidates`, `maxFewShotExamples`, `promptModel`, `samplingTemperature`, `candidateTemperature`, `seed`. Details in [`docs/simba.md`](docs/simba.md).
- **`gepa`** (Genetic-Pareto reflective prompt evolution) — evolves a pool of description candidates by reflective mutation: sample a parent from the Pareto frontier of per-example winners, show a reflection LM the traces plus metric feedback, and let it rewrite one step's description. GEPA is strictly description-only; when `maxFewShotExamples > 0` it runs a BootstrapFewShot pre-pass first (DSPy-style teleprompter composition, un-billed against GEPA's budget) so it evolves descriptions on an example-carrying workflow. Budget: exactly one of `auto` (`"light" | "medium" | "heavy"`, DSPy's auto-budget estimate), `maxFullEvals`, or `maxMetricCalls`. Details in [`docs/gepa.md`](docs/gepa.md).
- **`bootstrapFewShot`** — runs the workflow as its own teacher over the training set, keeps traces the metric accepts as few-shot examples, and backfills with raw labeled examples up to `maxLabeledExamples`.
- **`labeledFewShot`** — just installs up to `maxFewShotExamples` labeled training examples on every step; no LM calls.

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
