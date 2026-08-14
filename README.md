# dsmastra

GEPA prompt optimization for [Mastra](https://mastra.ai) workflows, written in TypeScript on the Vercel AI SDK. Give a native workflow a training set and a Mastra scorer, and dsmastra uses execution traces plus scorer feedback to improve each step's instructions in place. The result is the same Mastra workflow, ready to run through Mastra's engine.

## Tune a workflow with GEPA

Declare tunable steps with `declareStep`, compose them with Mastra's `createWorkflow`, then pass the committed workflow to `gepa`:

```ts
import { openai } from "@ai-sdk/openai"
import { writeFile } from "node:fs/promises"
import { createScorer } from "@mastra/core/evals"
import { createWorkflow } from "@mastra/core/workflows"
import { declareStep, gepa } from "dsmastra"
import { z } from "zod"

const math = declareStep({
  id: "math",
  description: "Return y = x.", // wrong on purpose
  inputSchema: z.object({ x: z.number() }),
  outputSchema: z.object({ y: z.number() }),
  model: openai("gpt-4.1-mini"),
})

const workflow = createWorkflow({
  id: "double",
  inputSchema: z.object({ x: z.number() }),
  outputSchema: z.object({ y: z.number() }),
})
  .then(math)
  .commit()

const trainingSet = [1, 2, 3, 5, 8].map((x) => ({
  inputData: { x },
  outputData: { y: x * 2 },
}))

const doubling = createScorer({
  id: "doubling",
  description: "Exact match on y, with corrective feedback.",
})
  .generateScore(({ run }) => (run.output?.y === run.groundTruth?.y ? 1 : 0))
  .generateReason(({ run }) =>
    run.output?.y === run.groundTruth?.y
      ? "Correct."
      : `Expected ${run.groundTruth?.y}, received ${run.output?.y}.`
  )

const { score } = await gepa(workflow, {
  trainingSet,
  maxScorerCalls: 60,
  scorer: doubling,
  savePrompts: (prompts) =>
    writeFile("prompts.json", JSON.stringify(prompts, null, 2)),
})

console.log(score, math.description)

// GEPA tunes the workflow in place, so it still runs through Mastra.
const run = await workflow.createRun()
const result = await run.start({ inputData: { x: 21 } })
console.log(result)
```

`gepa` applies the best prompt state to the workflow's live steps and returns `{ candidates, score }`: every candidate it tried as a `[prompts, { score }]` pair plus the winner's score. The workflow itself is the tuned artifact — there's nothing to reassign.

## Why GEPA

GEPA—Genetic-Pareto reflective prompt evolution—evaluates the workflow, gives a reflection model the execution traces and scorer feedback, and asks it to propose better instructions. It keeps a Pareto frontier of candidates that perform well on different examples, so a prompt that solves one hard case can remain useful even when another candidate has the better average score.

A `scorer` is required on every optimizer: a Mastra scorer built with `createScorer`, or the registration key of one when the workflow is registered on a Mastra instance (`scorer: "answerQuality"`). Define a scorer once and reuse it everywhere — live evals, experiments, and optimization. The exported `createExactMatchScorer()` covers exact-match scoring across every expected output field. Each evaluation calls `scorer.run()` with the example's `inputData` as `input`, the rollout result as `output`, and the expected `outputData` as `groundTruth` — the same mapping Mastra's own `runEvals` uses — and links the score to the rollout's trace when tracing is configured. A scorer's `generateReason` output becomes GEPA's reflection feedback, giving the model a concrete reason to change the instructions. Scores must be finite numbers, higher is better; keep the scorer's scale aligned with `perfectScore` and any thresholds.

GEPA requires a non-empty `trainingSet` and exactly one budget:

- `auto: "light" | "medium" | "heavy"` estimates a budget from the workflow and validation-set sizes.
- `maxScorerCalls` sets the scorer-run budget directly; an iteration already underway can finish slightly beyond it.
- `maxFullEvals` expresses the budget in full passes over the supplied datasets.

By default, the training set is also the validation set. Supply `validationSet` when candidate selection should be measured on held-out examples. Set `maxFewShotExamples` above zero only when GEPA should run a few-shot bootstrap before evolving the instructions.

The implementation follows DSPy's GEPA control flow; [`docs/gepa.md`](docs/gepa.md) records the port's selection, reflection, merge, and budget behavior.

## Saving and loading prompts

Every optimizer requires `savePrompts`. GEPA calls it when a new aggregate-score best is found and once more with the final result, so the latest persisted value is a usable checkpoint:

```ts
import { readFile } from "node:fs/promises"
import { loadPrompts } from "dsmastra"

loadPrompts(workflow, JSON.parse(await readFile("prompts.json", "utf8")))
```

The payload is `{ version: 1, steps: { [stepId]: { description, examples } } }`. `loadPrompts` applies it to the workflow's live steps and throws when the saved step IDs don't exactly match, preventing a stale checkpoint from being partially applied. Keep example inputs and outputs JSON-serializable when the storage format is JSON.

## SIMBA

`simba` is also available for stochastic minibatch search. It samples multiple rollouts, finds examples where their scores diverge, and turns the contrast into a few-shot example or a natural-language rule. It needs a model that supports temperature because rollout diversity supplies its learning signal; see [`docs/simba.md`](docs/simba.md) for its controls and algorithm.

The package also exports `bootstrapFewShot` and `labeledFewShot` for installing examples from teacher traces or labeled data. Unlike their DSPy counterparts they score the compiled workflow over the training set (one evaluation pass), so all four optimizers return the same `{ candidates: [prompts, { score }][], score }` shape.

## Supported workflows

Optimizers accept any native Mastra workflow — serial chains, `.parallel()`, `.branch()`, loops, and mixes of tunable and ordinary steps. Rollouts run through Mastra's own engine (`createRun()` + `start()`), so every graph executes with its real semantics and optimization runs show up in Mastra observability. Only the `declareStep` steps get tuned; everything else runs untouched. Steps inside a nested workflow are opaque to tuning. A workflow with no `declareStep` steps is rejected.

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

The `dspy/` directory is a reference clone used to trace the original optimizer implementations; it isn't part of the package.
