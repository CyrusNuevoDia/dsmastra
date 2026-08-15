# dsmastra

Declarative, self-improving steps for [Mastra](https://mastra.ai) workflows. You declare each LLM step as a typed signature — input schema, output schema, a rough instruction — and develop against evals instead of prompt strings: give an optimizer labeled examples and a Mastra scorer, and it rewrites each step's instructions and few-shot examples in place using execution traces and scorer feedback. Labeled data goes in, tuned prompts come out, and the artifact is still a plain Mastra workflow that runs through Mastra's engine. No more prompt hell.

Written in TypeScript on the Vercel AI SDK, following DSPy's optimizer implementations.

## Install

```sh
bun add github:CyrusNuevoDia/dsmastra
```

The package ships TypeScript source and currently targets Bun projects. It expects `@mastra/core`, `ai`, and `zod` as peers — which you already have in a Mastra project — plus your model provider (e.g. `@ai-sdk/openai`).

## Example: tuning a support-ticket triage step

Declare steps with `declareStep`, compose them with Mastra's `createWorkflow`, then pass the committed workflow to an optimizer with labeled data and a scorer:

```ts
import { openai } from "@ai-sdk/openai"
import { writeFile } from "node:fs/promises"
import { createScorer } from "@mastra/core/evals"
import { createWorkflow } from "@mastra/core/workflows"
import { declareStep, gepa } from "dsmastra"
import { z } from "zod"

const triage = declareStep({
  id: "triage",
  description: "Categorize the support ticket.", // a rough draft is enough — GEPA rewrites it
  inputSchema: z.object({ subject: z.string(), body: z.string() }),
  outputSchema: z.object({
    category: z.enum(["billing", "bug", "how-to", "feature-request"]),
    priority: z.enum(["low", "normal", "urgent"]),
  }),
  model: openai("gpt-5.6-terra"),
})

const workflow = createWorkflow({
  id: "support-triage",
  inputSchema: triage.inputSchema,
  outputSchema: triage.outputSchema,
})
  .then(triage)
  .commit()

// Labeled tickets — pulled from your helpdesk, categorized by your team.
// A few dozen examples is a realistic starting point.
const trainingSet = [
  {
    inputData: {
      subject: "Charged twice this month",
      body: "My card shows two charges for the Pro plan on the 3rd and the 5th.",
    },
    outputData: { category: "billing", priority: "urgent" },
  },
  {
    inputData: {
      subject: "Export to CSV?",
      body: "Is there a way to export the dashboard table to CSV?",
    },
    outputData: { category: "how-to", priority: "low" },
  },
  // ...more labeled tickets
]

// The scorer is the optimization objective. Its reason becomes the feedback
// GEPA's reflection model reads, so say what was wrong, not just the score.
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

console.log(score, triage.description) // the rewritten instruction

// The workflow is tuned in place — run it through Mastra as usual.
const run = await workflow.createRun()
const result = await run.start({
  inputData: { subject: "App crashes on login", body: "Since this morning..." },
})
console.log(result)
```

The optimizer applies the best prompt state to the workflow's live steps and returns `{ candidates, score }`: every candidate it tried as a `[prompts, { score }]` pair plus the winner's score. The workflow itself is the tuned artifact — there's nothing to reassign.

## Optimizers

All four optimizers take the same shape of input — a committed workflow, a `trainingSet`, a `scorer`, and `savePrompts` — and return the same `{ candidates: [prompts, { score }][], score }`.

### GEPA

[GEPA](https://github.com/gepa-ai/gepa) — Genetic-Pareto reflective prompt evolution ([paper](https://arxiv.org/abs/2507.19457)) — evaluates the workflow, gives a reflection model the execution traces and scorer feedback, and asks it to propose better instructions. It keeps a Pareto frontier of candidates that perform well on different examples, so a prompt that solves one hard case can remain useful even when another candidate has the better average score.

GEPA requires a non-empty `trainingSet` and exactly one budget:

- `auto: "light" | "medium" | "heavy"` estimates a budget from the workflow and validation-set sizes.
- `maxScorerCalls` sets the scorer-run budget directly; an iteration already underway can finish slightly beyond it.
- `maxFullEvals` expresses the budget in full passes over the supplied datasets.

`reflectionModel` is the LM that proposes new instructions; it defaults to the first step's model, but prompt rewriting benefits from a stronger model than the one being tuned, so supply one. If your scorer's scale doesn't top out at 1, set `perfectScore` to its actual maximum so GEPA knows when an example can't improve further.

By default, the training set is also the validation set. Supply `validationSet` when candidate selection should be measured on held-out examples. Set `maxFewShotExamples` above zero only when GEPA should run a few-shot bootstrap before evolving the instructions.

The implementation follows DSPy's GEPA control flow; [`docs/gepa.md`](docs/gepa.md) records the port's selection, reflection, merge, and budget behavior.

### SIMBA

[SIMBA](https://dspy.ai/api/optimizers/SIMBA/) — stochastic introspective minibatch ascent, from [DSPy](https://github.com/stanfordnlp/dspy) — samples multiple rollouts per example, finds examples where their scores diverge, and turns the contrast into a few-shot example or a natural-language rule. It needs a model that supports temperature because rollout diversity supplies its learning signal. Like GEPA's `reflectionModel`, its `promptModel` — the LM that writes the advice rules — defaults to the first step's model and deserves a stronger one. See [`docs/simba.md`](docs/simba.md) for its controls and algorithm.

### Few-shot bootstrapping

`bootstrapFewShot` and `labeledFewShot` (ports of DSPy's [BootstrapFewShot](https://dspy.ai/api/optimizers/BootstrapFewShot/) and LabeledFewShot) install examples from teacher traces or labeled data. Unlike their DSPy counterparts they score the compiled workflow over the training set (one evaluation pass), so they return the same result shape as GEPA and SIMBA. `bootstrapFewShot` runs a `teacher` workflow (or the student itself, optionally with `teacherSettings` overriding its model or temperature) over the training set and installs successful rollouts as demos. An optional `gate: { scorer, threshold? }` — a Mastra `type: "trajectory"` scorer that sees each teacher rollout as a Trajectory in `run.output` — decides which rollouts qualify, while the objective `scorer` still scores the compiled workflow; the `gate` option's docs describe the Trajectory shape.

## Scorers

A `scorer` is required on every optimizer: a Mastra scorer built with `createScorer`, or the registration key of one when the workflow is registered on a Mastra instance (`scorer: "answerQuality"`). Define a scorer once and reuse it everywhere — live evals, experiments, and optimization. The exported `createExactMatchScorer()` covers exact-match scoring across every expected output field. Each evaluation calls `scorer.run()` with the example's `inputData` as `input`, the rollout result as `output`, and the expected `outputData` as `groundTruth` — the same mapping Mastra's own `runEvals` uses — and links the score to the rollout's trace when tracing is configured. A scorer's `generateReason` output becomes GEPA's reflection feedback, giving the model a concrete reason to change the instructions. Scores must be finite numbers, higher is better.

## Saving and loading prompts

Every optimizer requires `savePrompts`. GEPA calls it when a new aggregate-score best is found and once more with the final result, so the latest persisted value is a usable checkpoint:

```ts
import { readFile } from "node:fs/promises"
import { loadPrompts } from "dsmastra"

loadPrompts(workflow, JSON.parse(await readFile("prompts.json", "utf8")))
```

The payload is `{ version: 1, steps: { [stepId]: { description, examples } } }`. `loadPrompts` applies it to the workflow's live steps and throws when the saved step IDs don't exactly match, preventing a stale checkpoint from being partially applied. Keep example inputs and outputs JSON-serializable when the storage format is JSON.

## Supported workflows

Optimizers accept any native Mastra workflow — serial chains, `.parallel()`, `.branch()`, loops, and mixes of declarative and ordinary steps. Rollouts run through Mastra's own engine (`createRun()` + `start()`), so every graph executes with its real semantics and optimization runs show up in Mastra observability. Only the `declareStep` steps get tuned; everything else runs untouched. Steps inside a nested workflow are opaque to tuning. A workflow with no `declareStep` steps is rejected.

`declareStep` forwards a `scorers` option to Mastra for live evaluation in production. During optimization rollouts those attached scorers are disabled (`createRun({ disableScorers: true })`) so the objective scorer is the only one billed per rollout — the same guard Mastra's `runEvals` applies.

Each step uses Zod input and output schemas and any AI SDK-compatible `LanguageModel`. dsmastra reads no provider environment variables itself; authentication belongs to the model provider you supply.

## Development

The repository is Bun-only. If the pinned tools are missing, run `mise trust && mise install` first.

```sh
bun install
just test       # unit tests
just test-int   # paid integration tests; requires OPENAI_API_KEY in .env
just check      # formatting, lint, typecheck, and unit tests
just fmt
```

To trace the original optimizer implementations, clone DSPy into `dspy/` — it's gitignored and referenced by the docs, not part of the package.
