# dsmastra

Declarative, self-improving steps for [Mastra](https://mastra.ai) workflows. You declare each LLM step as a typed signature — input schema, output schema, a rough instruction — and develop against evals instead of prompt strings: give an optimizer labeled examples and a Mastra scorer, and it rewrites each step's instructions and few-shot examples in place using execution traces and scorer feedback. Labeled data goes in, tuned prompts come out, and the artifact is still a plain Mastra workflow that runs through Mastra's engine. No more prompt hell.

Written in TypeScript on the Vercel AI SDK, following DSPy's optimizer implementations.

## Install

```sh
bun add github:CyrusNuevoDia/dsmastra
# or: npm install github:CyrusNuevoDia/dsmastra
# or: pnpm add github:CyrusNuevoDia/dsmastra
```

The repo ships a built `dist/`, so any package manager works. It expects `@mastra/core`, `ai`, `zod`, and `typescript` as peers — which you already have in a Mastra project — plus your model provider (e.g. `@ai-sdk/openai`).

## Example: tuning a support-ticket pipeline

Declare steps with `declareStep`, compose them with Mastra's `createWorkflow`, then pass the committed workflow to an optimizer with labeled data and a scorer:

```ts
import { openai } from "@ai-sdk/openai"
import { readFile, writeFile } from "node:fs/promises"
import { createScorer } from "@mastra/core/evals"
import { createStep, createWorkflow } from "@mastra/core/workflows"
import { declareStep, gepa, loadPrompts } from "dsmastra"
import { z } from "zod"

// Two declarative steps. A rough instruction is enough — the optimizer
// rewrites descriptions and installs few-shot examples on both.
const extract = declareStep({
  id: "extract",
  description: "Pull the product, issue, and sentiment out of the email.",
  inputSchema: z.object({ email: z.string() }),
  outputSchema: z.object({
    product: z.string(),
    issue: z.string(),
    sentiment: z.enum(["calm", "frustrated", "angry"]),
  }),
  model: openai("gpt-5.6-terra"),
})

const triage = declareStep({
  id: "triage",
  description: "Categorize the ticket.",
  inputSchema: extract.outputSchema,
  outputSchema: z.object({
    category: z.enum(["billing", "bug", "how-to", "feature-request"]),
    priority: z.enum(["low", "normal", "urgent"]),
  }),
  model: openai("gpt-5.6-terra"),
})

// An ordinary Mastra step — deterministic, never touched by the optimizer.
const route = createStep({
  id: "route",
  inputSchema: triage.outputSchema,
  outputSchema: triage.outputSchema.extend({ queue: z.string() }),
  execute: async ({ inputData }) => ({
    ...inputData,
    queue: inputData.priority === "urgent" ? "oncall" : inputData.category,
  }),
})

const workflow = createWorkflow({
  id: "support-triage",
  inputSchema: extract.inputSchema,
  outputSchema: route.outputSchema,
})
  .then(extract)
  .then(triage)
  .then(route)
  .commit()

// Labeled tickets — pulled from your helpdesk, categorized by your team.
// A few dozen examples is a realistic starting point.
const trainingSet = [
  {
    inputData: {
      email:
        "Subject: Charged twice. My card shows two charges for the Pro plan this month.",
    },
    outputData: { category: "billing", priority: "urgent", queue: "oncall" },
  },
  {
    inputData: {
      email: "Subject: CSV export? Is there a way to export the dashboard table?",
    },
    outputData: { category: "how-to", priority: "low", queue: "how-to" },
  },
  // ...more labeled tickets
]

// The scorer is the optimization objective. Its reason becomes the feedback
// the reflection model reads, so say what was wrong, not just the score.
const triageAccuracy = createScorer({
  id: "triage-accuracy",
  description: "Category must match; priority is worth half.",
})
  .generateScore(({ run }) =>
    run.output?.category === run.groundTruth?.category
      ? run.output?.priority === run.groundTruth?.priority
        ? 1
        : 0.5
      : 0
  )
  .generateReason(({ run }) =>
    run.output?.category === run.groundTruth?.category
      ? run.output?.priority === run.groundTruth?.priority
        ? "Correct."
        : `Category correct, but priority should be ${run.groundTruth?.priority}, got ${run.output?.priority}.`
      : `Category should be ${run.groundTruth?.category}, got ${run.output?.category}.`
  )

const { score } = await gepa(workflow, {
  trainingSet,
  maxScorerCalls: 200,
  reflectionModel: openai("gpt-5.6-sol"), // a big model proposes the rewrites
  scorer: triageAccuracy,
  savePrompts: (prompts) =>
    writeFile("prompts.json", JSON.stringify(prompts, null, 2)),
})

console.log(score, extract.description, triage.description) // the score, then the rewritten instructions

// In this script gepa already tuned the steps in place. Later, in a fresh
// process — e.g. at deploy time — rebuild the workflow from the code above,
// then loadPrompts is how it catches up to the saved state.
loadPrompts(workflow, JSON.parse(await readFile("prompts.json", "utf8")))

const run = await workflow.createRun()
const result = await run.start({
  inputData: { email: "Subject: Crash on login. The app dies every time..." },
})
console.log(result)
```

The optimizer tunes the workflow in place — there's nothing to reassign — and returns `{ candidates, score }`: every candidate it tried as a `[prompts, { score }]` pair plus the winner's score. `savePrompts` is required and doubles as checkpointing: GEPA and SIMBA — the search optimizers, below — call it on each new aggregate-score best and once with the final result; the few-shot optimizers call it once. The payload is `{ version: 1, steps: { [stepId]: { description, examples } } }`; keep example inputs and outputs JSON-serializable when the storage format is JSON. `loadPrompts` applies a payload to the workflow's live steps and throws when the saved step IDs don't exactly match, so a stale checkpoint can't be partially applied.

## Optimizers

All four optimizers — `gepa`, `simba`, `bootstrapFewShot`, `labeledFewShot` — take the same shape of input (a committed workflow, a `trainingSet`, a `scorer`, and `savePrompts`) and return the same `{ candidates, score }`.

### GEPA

[GEPA](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective prompt evolution ([paper](https://arxiv.org/abs/2507.19457)) — evaluates the workflow, gives a reflection model the execution traces and scorer feedback, and asks it to propose better instructions. It keeps a Pareto frontier of candidates that perform well on different examples, so a prompt that solves one hard case can remain useful even when another candidate has the better average score.

GEPA requires a non-empty `trainingSet` and exactly one budget:

- `auto: "light" | "medium" | "heavy"` estimates a budget from the workflow and validation-set sizes.
- `maxScorerCalls` sets the scorer-run budget directly; an iteration already underway can finish slightly beyond it.
- `maxFullEvals` expresses the budget in full passes over the supplied datasets.

`reflectionModel` is the LM that proposes new instructions; it defaults to the first step's model, but prompt rewriting benefits from a stronger model than the one being tuned, so supply one. If your scorer's scale doesn't top out at 1, set `perfectScore` to its actual maximum — GEPA skips reflecting on minibatches whose every score reaches it, and with the default of 1 on a 0–5 scorer it would wrongly skip batches that still have headroom.

By default, the training set is also the validation set. Supply `validationSet` when candidate selection should be measured on held-out examples. Set `maxFewShotExamples` above zero only when GEPA should run a few-shot bootstrap before evolving the instructions.

The implementation follows DSPy's GEPA control flow; [`docs/gepa.md`](docs/gepa.md) records the port's selection, reflection, merge, and budget behavior.

### SIMBA

[SIMBA](https://dspy.ai/api/optimizers/SIMBA/) — stochastic introspective minibatch ascent, from [DSPy](https://github.com/stanfordnlp/dspy) — samples multiple rollouts per example, finds examples where their scores diverge, and turns the contrast into a few-shot example or a natural-language rule. It needs a model that supports temperature because rollout diversity supplies its learning signal. Like GEPA's `reflectionModel`, its `promptModel` — the LM that writes the advice rules — defaults to the first step's model and deserves a stronger one. See [`docs/simba.md`](docs/simba.md) for its controls and algorithm.

### Few-shot bootstrapping

`bootstrapFewShot` and `labeledFewShot` (ports of DSPy's [BootstrapFewShot](https://dspy.ai/api/optimizers/BootstrapFewShot/) and LabeledFewShot) install few-shot examples without touching the instructions. Unlike their DSPy counterparts they score the compiled workflow over the training set (one evaluation pass), so they return the same result shape as GEPA and SIMBA.

`labeledFewShot` installs labeled examples directly. `bootstrapFewShot` runs a `teacher` workflow (or the student itself) over the training set — `teacherSettings` can override the teacher's model or temperature either way — installs the per-step traces of passing rollouts as demos, and backfills each step's remaining slots with labeled examples up to `maxLabeledExamples`. A rollout passes when the objective scorer scores it above zero, or at or above `scoreThreshold` when set. An optional `gate: { scorer, threshold? }` — a Mastra `type: "trajectory"` scorer that sees each teacher rollout as a Trajectory in `run.output`, one entry per engine-executed step — decides instead which rollouts qualify; the TSDoc on `gate` describes the Trajectory shape in detail.

## Scorers

A `scorer` is required on every optimizer: a Mastra scorer built with `createScorer`, or the registration key of one when the workflow is registered on a Mastra instance (`scorer: "answerQuality"`). Define a scorer once and reuse it everywhere — live evals, experiments, and optimization. The exported `createExactMatchScorer()` covers exact-match scoring across every expected output field. Each evaluation calls `scorer.run()` with the example's `inputData` as `input`, the rollout result as `output`, and the expected `outputData` as `groundTruth` — the same mapping Mastra's own `runEvals` uses — and links the score to the rollout's trace when tracing is configured. A scorer's `generateReason` output becomes the optimizer's feedback, giving the reflection model a concrete reason to change the instructions. Scores must be finite numbers, higher is better.

## How it works

`declareStep` builds a real Mastra step whose prompt is rendered from three parts: the `description` (the instruction), the current `examples` (few-shot demos), and the live input shaped by the Zod schemas. The description and examples are the step's mutable prompt state — that's all an optimizer is allowed to change; the schemas, model, and call settings are fixed config. Any AI SDK-compatible `LanguageModel` works, and dsmastra reads no provider environment variables itself — authentication belongs to the model provider you supply.

During optimization, every candidate is evaluated by running the workflow through Mastra's own engine (`createRun()` + `start()`), so any native graph works — serial chains, `.parallel()`, `.branch()`, loops, and mixes of declarative and ordinary steps all execute with their real semantics, and optimization runs show up in Mastra observability. Only the `declareStep` steps get tuned; everything else runs untouched. Steps inside a nested workflow are opaque to tuning, and a workflow with no `declareStep` steps is rejected.

`declareStep` forwards a `scorers` option to Mastra for live evaluation in production. During optimization rollouts those attached scorers are disabled (`createRun({ disableScorers: true })`) so the objective scorer is the only one billed per rollout — the same guard Mastra's `runEvals` applies.

## Development

The repository is Bun-only. If the pinned tools are missing, run `mise trust && mise install` first.

```sh
bun install
just test       # unit tests
just test-int   # paid integration tests; requires OPENAI_API_KEY in .env
just check      # formatting, lint, typecheck, and unit tests
just fmt
just build      # bundle + type declarations to dist/
```

To trace the original optimizer implementations, clone DSPy into `dspy/` — it's gitignored and referenced by the docs, not part of the package.
