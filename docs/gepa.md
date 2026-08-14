# GEPA — Genetic-Pareto Reflective Prompt Evolution

Authoritative flow of DSPy's GEPA optimizer, traced from `dspy/dspy/teleprompt/gepa/` plus the external `gepa` engine package (github.com/gepa-ai/gepa, which `GEPA.compile` delegates to via `gepa.optimize()`), written so we can replicate it in `src/`. GEPA evolves a pool of candidates — each candidate is a `{componentName: instructionText}` map — by reflective mutation: sample a parent from the Pareto frontier of per-validation-instance winners, run it on a small train minibatch with trace capture, feed the traces plus metric feedback to a reflection LM that rewrites one component's instructions, and keep the child only if its minibatch score improves. Accepted children get a full valset eval and can later be merged (crossover) with other frontier survivors.

## Source files

DSPy wrapper (in-repo):

- [`dspy/dspy/teleprompt/gepa/gepa.py`](../dspy/dspy/teleprompt/gepa/gepa.py) — `GEPA` class, `compile`, `auto_budget`, metric wrapping, `DspyGEPAResult`
- [`dspy/dspy/teleprompt/gepa/gepa_utils.py`](../dspy/dspy/teleprompt/gepa/gepa_utils.py) — `DspyAdapter` (`build_program`, `evaluate`, `make_reflective_dataset`, `propose_new_texts`), `ScoreWithFeedback`
- [`dspy/dspy/teleprompt/gepa/instruction_proposal.py`](../dspy/dspy/teleprompt/gepa/instruction_proposal.py) — optional multimodal instruction proposer

Core engine (external `gepa` pip package, github.com/gepa-ai/gepa, paths under `src/gepa/`):

- `api.py` — `optimize()` wiring and defaults
- `core/engine.py` — main loop, acceptance, `_add_evaluated_program`
- `core/state.py` — `GEPAState`, Pareto front updates
- `core/result.py` — `GEPAResult`
- `proposer/reflective_mutation/reflective_mutation.py` — reflective mutation stages
- `proposer/merge.py` — merge proposer
- `strategies/` — `candidate_selector.py` (Pareto sampling), `batch_sampler.py` (epoch-shuffled minibatches), `component_selector.py` (round-robin), `instruction_proposal.py` (default prompt template), `eval_policy.py`
- `gepa_utils.py` — dominance filtering (`remove_dominated_programs`)

A local clone for reference lives at `/private/tmp/claude-501/-Users-knrz-Git-CyrusNuevoDia-dsmastra/94f3855e-6b5a-457f-b26d-0a902cc56294/scratchpad/gepa` (session-scoped; re-clone `https://github.com/gepa-ai/gepa` if missing).

## Configuration (DSPy wrapper)

| param | default | meaning |
|---|---|---|
| `metric` | required | `(gold, pred, trace, pred_name, pred_trace) -> number \| {score, feedback}` — must accept exactly 5 positional args |
| `auto` / `max_full_evals` / `max_metric_calls` | exactly one | budget; `auto ∈ light(n=6) / medium(n=12) / heavy(n=18)` |
| `reflection_minibatch_size` | 3 | train examples per reflection round |
| `candidate_selection_strategy` | `"pareto"` | or `"current_best"` |
| `reflection_lm` | required* | LM that rewrites instructions (*or a custom `instruction_proposer`) |
| `skip_perfect_score` | true | skip reflection when the whole minibatch hits `perfect_score` |
| `add_format_failure_as_feedback` | false | keep parse-failure trace steps in the reflective dataset |
| `component_selector` | `"round_robin"` | which component(s) to rewrite per round (`"all"` available) |
| `use_merge` / `max_merge_invocations` | true / 5 | crossover of frontier survivors (the core engine's own default is merge OFF; DSPy turns it on) |
| `num_threads` | none | eval parallelism |
| `failure_score` / `perfect_score` | 0.0 / 1.0 | |
| `track_stats` / `track_best_outputs` | false / false | `track_best_outputs` requires `track_stats`; attaches `detailed_results` |
| `warn_on_score_mismatch` | true | one-time warning when predictor-level metric score ≠ module score |
| `seed` | 0 | |

### Metric / feedback contract

`ScoreWithFeedback = {score: number, feedback: string}`. The wrapper builds a per-predictor feedback function around the user metric:

```text
o = metric(gold, prediction, fullTrace, predName, [(predictor, inputs, outputs)])
if o has .feedback: feedback ??= "This trajectory got a score of {o.score}."; return o
else:              return {score: o, feedback: "This trajectory got a score of {o}."}
```

If the metric's predictor-level score disagrees with the module-level score, the predictor score is **discarded and overwritten with the module score** (feedback text kept; one-time warning). Only the feedback string ever reaches the reflection LM.

### Budget math (exact)

- `max_full_evals` → `max_metric_calls = max_full_evals * (len(trainset) + len(valset ?? []))` — note: when no valset was given it's `* len(trainset)` only, even though valset then defaults to trainset.
- `auto` → `auto_budget(num_preds, n, valsetSize, M = 35, m = 5)` where `n` is the auto tier's candidate count, `num_preds = max(#components, 1)`, `valsetSize = len(valset ?? trainset)`:

```text
N     = int(max(2 * (num_preds * 2) * log2(n), 1.5 * n))    // planned trials
total = valsetSize                                          // seed full eval
      + n * 5                                               // bootstrap allowance
      + N * M                                               // minibatch evals
if N == 0: return total
total += (floor((N + 1) / m) + 1 + (N < m ? 1 : 0)) * valsetSize
```

Guards: throw if `N < 0 || valsetSize < 0 || M < 0`, and if `m < 1`. `full_eval_steps` (the `m`) exists **only** in this estimate — the engine has no periodic full-eval scheduling; every accepted candidate simply gets one full valset eval. Budget enforcement is just `total_num_evals >= max_metric_calls`, checked at the top of each iteration, so the final iteration can overshoot by a full valset eval plus two minibatch evals.

## Setup (`compile(student, trainset, valset?)`)

1. Assert trainset non-empty. `valset = valset ?? trainset` (with a warning; another note if `len(valset) > 35` since every accepted candidate costs a full valset eval).
2. Seed candidate = `{predName: signature.instructions}` over `student.named_predictors()` — **key order defines component order** for round-robin.
3. Two RNGs, both seeded with `seed`: the engine's (parent selection, batch shuffling, merge sampling) and the adapter's (trace-step choice in the reflective dataset).
4. Full valset eval of the seed → per-instance scores. `total_num_evals = len(valset)`, `num_full_ds_evals = 1` (set directly, bypassing the increment hook).
5. Call the engine with the adapter, `raise_on_exception = true`.
6. After the loop: return `adapter.build_program(result.best_candidate)`; if `track_stats`, attach `detailed_results` (a `DspyGEPAResult`).

### Adapter contract

The engine is generic; the DSPy adapter supplies three operations. A port should keep this seam — it's what makes the optimizer program-agnostic.

- **`build_program(candidate)`**: deepcopy the student; for each named predictor whose name is in the candidate, replace its instructions with the candidate's text.
- **`evaluate(batch, candidate, captureTraces) -> {outputs, scores, trajectories?}`**: build the program, run it on the batch, score with the metric. Must never throw per-example — failed rollouts score `failure_score` with `output = null` (only a build-time program error may abort). With `captureTraces`, each trajectory records `{example, prediction, trace, score}` where `trace` is a list of `(predictor, inputs, outputs)` steps. Batches larger than `reflection_minibatch_size` are tagged as "full evals" for logging; minibatch evals are not.
- **`make_reflective_dataset(candidate, evalBatch, componentsToUpdate)`** and **`propose_new_texts(candidate, reflectiveDataset, componentsToUpdate)`**: see below.

## Main loop

```text
while total_num_evals < max_metric_calls:      // checked at top only
├─ A. merge attempt? (only if use_merge && merges_due > 0 && last iteration accepted a candidate)
│     └─ whether accepted or rejected, the iteration ends here — no reflection this round
├─ B. reflective mutation
│  ├─ 1. parent = sample from Pareto frontier (see below)
│  ├─ 2. minibatch = next reflection_minibatch_size train ids (epoch-shuffled, sequential)
│  ├─ 3. eval parent on minibatch WITH trace capture          (+bsize metric calls)
│  │     skip round if: no trajectories, or all scores ≥ perfect_score (skip_perfect_score)
│  ├─ 4. components = round_robin cursor of parent → exactly one component (cursor advances
│  │     on the parent even if the proposal is later rejected)
│  ├─ 5. build reflective dataset for that component; skip round if empty
│  ├─ 6. reflection LM proposes new instruction text (one LM call per component, sequential)
│  ├─ 7. child = parent with the proposed component text(s) replaced
│  │     (empty proposal → skip round, no child eval; proposed names must already exist)
│  ├─ 8. eval child on the SAME minibatch                     (+bsize metric calls)
│  ├─ 9. accept iff sum(child scores) > sum(parent scores)    (strict, sums not means)
│  └─ 10. if accepted: full valset eval                       (+len(valset) calls)
│        → add to pool, update Pareto frontier,
│          last_iter_found_new_program = true,
│          merges_due++ (while total_merges_tested < max_merge_invocations)
└─ result: best candidate = argmax mean valset score (lowest index wins ties)
```

Rejected children still cost their two minibatch evals; they're never added to the pool. Duplicate children within one iteration are deduplicated by content (`sorted(candidate.items())`) before the full eval. Any exception in an iteration aborts the run (DSPy sets `raise_on_exception`).

When a candidate is added: record its parent indices, snapshot `total_num_evals` **before** billing its full eval as its `discovery_eval_count`, bill the eval, update the frontier, and set its round-robin cursor to `max(parent cursors)` (it inherits the parent's next-component pointer rather than resetting to 0).

### Pareto frontier & parent selection

Per validation instance, the state tracks the best score seen (`pareto_front_valset: {valId: number}`) and the **set** of candidates achieving it (`program_at_pareto_front_valset: {valId: Set<idx>}`). Update rule on a new candidate's full eval, per instance: `score > prev` → replace score, front becomes `{idx}` (and `best_outputs_valset[valId]` resets to its output); `score == prev` → add to the set. **Exact float equality, no epsilon.**

Parent sampling (`"pareto"` strategy):

1. Take the frontier mapping `{valId: Set<candidates>}`.
2. **Dominance filter**: a candidate survives iff it is the *sole* occupant of at least one instance's front. Implementation: sort candidates ascending by aggregate score (mean of their per-instance scores, `-inf` if unevaluated); repeatedly scan in that order and mark a candidate dominated if every frontier key containing it also contains some other non-dominated candidate; restart the scan after each removal until a fixed point. Low scorers are eliminated first, so ties resolve toward higher aggregate score. Every non-empty front must retain ≥ 1 survivor.
3. Sample a survivor with probability proportional to **how many instances it sits on the front of** (`rng.choice` over a list with each candidate repeated `freq` times).

`"current_best"` instead picks argmax mean valset score (lowest index on ties). The engine also ships `EpsilonGreedy(ε=0.1)` (random with prob ε, else argmax) and `TopKPareto(k=5)` (frontier sets restricted to the top-k by aggregate score) selectors — optional for a port.

### Minibatch sampler (epoch-shuffled)

- Per epoch: `shuffled = rng.shuffle(allTrainIds)`, then pad the tail to a multiple of `bsize` by repeatedly appending the currently least-frequently-used id (tracking per-id counts across padding).
- Per iteration: `base = (iteration * bsize) % len(shuffled)`; serve `shuffled[base : base + bsize]`. Reshuffle when the un-offset `iteration * bsize` crosses an epoch boundary (`floor(base / len) > epoch`) or the trainset size changed. Repeat calls within one iteration offset by `callsThisIteration * bsize` without reshuffling.

### Reflective dataset

For the chosen component, for each minibatch trajectory:

1. `traceInstances = ` steps in the trace whose predictor's **signature equals** the component's signature (matched by signature equality, not name). Unless `add_format_failure_as_feedback`, drop parse-failure steps. If empty → skip this example.
2. Pick one step: the **first parse-failure** step if any remain; else if the whole prediction is a parse failure, skip the example; else a **random** one (adapter RNG).
3. Emit a record:
   - `Inputs`: `{field: str(value)}`. A conversation-history input is rendered instead into a single `Context` key as a ` ```json ` fenced block of `  {i}: {message}` lines. (With a custom multimodal proposer, rich types like images stay as objects.)
   - `Generated Outputs`: `{field: str(value)}`; for a parse failure, the single **string** `` "Couldn't parse the output as per the expected output format. The model's raw response was:\n```\n{raw}\n```\n\n" ``.
   - `Feedback`: the feedback function's text for this predictor; for a parse failure, `"Your output failed to parse. Follow this structure:\n"` + the expected chat-format rendering of the signature (one `role: content` line per message).

A component with zero records is omitted; if every requested component is empty the round is skipped (in Python this is a thrown-and-caught `"No valid predictions found for any module."`).

### Instruction-proposal prompt (verbatim)

One reflection-LM call per component. `<curr_param>` = the component's current instruction text; `<side_info>` = the reflective dataset rendered as markdown. The fence lines below are literal parts of the template:

````text
I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
````

`<side_info>` rendering: examples joined by `"\n\n"`, each as

```text
# Example {n}
## {key}                  // Inputs / Generated Outputs / Feedback
### {subkey}              // dicts recurse, header depth +1, capped at ######
### Item {k}              // lists/tuples
{str(value).strip()}
```

**Output extraction**: take the text between the first and last ` ``` ` fences, stripping a leading language tag (`^\S*\n`), then trim. If there's no complete fenced block: strip a leading fence + language tag if the response starts with one, or a trailing fence if it ends with one, else use the whole trimmed response.

### Merge (crossover)

Attempted at the top of an iteration when the previous iteration accepted a candidate and `merges_due > 0`. `total_merges_tested` counts **accepted** merges and caps at `max_merge_invocations`; a rejected merge consumes neither counter (but does burn the iteration and its subsample eval).

1. Pool = dominance-filtered frontier survivors. Sample two distinct candidates `i < j` (neither an ancestor of the other; up to 10 outer × 10 inner retries).
2. Find a common ancestor (transitive closure over parent lineage) that (a) does not outscore either descendant (aggregate scores), (b) has at least one component where exactly one descendant changed it (`anc == one side && sides differ`), (c) wasn't already used for this pair. Pick among eligible ancestors weighted by aggregate score (`rng.choices` with score weights). Record the triplet as tried **immediately** — a rejected triplet is never retried.
3. Build the merged candidate starting from a copy of the ancestor, per component: if the descendants agree, take that text; if only one differs from the ancestor, take the differing one's; if both differ, take the higher-aggregate-score descendant's (coin flip on exact tie). Skip if this exact merged candidate was already produced for this pair.
4. Require ≥ 5 shared evaluated val ids between the parents (`merge_val_overlap_floor`). Build a 5-id eval subsample from the shared ids: partition into `i`-better / `j`-better / tied, take up to 2 (`max(1, ceil(5/3))`) from each bucket at random, top up from unused ids (sampling with replacement only as a last resort), truncate to 5.
5. Eval the merged candidate on the subsample (no trace capture; bills metric calls). Accept iff `sum(merged) >= max(sum(parent_i on subsample), sum(parent_j on subsample))` — **non-strict `>=`**, unlike reflection's strict `>`. On accept: full valset eval, add to pool with `parents = [i, j]`, `merges_due--`, `total_merges_tested++`.

## State & result shapes

`GEPAState` (what a port must track):

```text
program_candidates:        Candidate[]            // index 0 = seed
parent_program_for_candidate: (idx | null)[][]    // [[null]] for seed; [i, j] for merges
prog_candidate_val_subscores: Map<valId, number>[]
num_metric_calls_by_discovery: number[]           // budget snapshot at discovery
pareto_front_valset:       Map<valId, number>     // best score per instance
program_at_pareto_front_valset: Map<valId, Set<idx>>
list_of_named_predictors:  string[]               // component order (seed key order)
named_predictor_id_to_update_next_for_program_candidate: number[]   // round-robin cursor per candidate
i: number                                         // iteration, starts at -1
num_full_ds_evals, total_num_evals: number
best_outputs_valset?: Map<valId, [idx, output][]>
```

`GEPAResult` / `DspyGEPAResult`: `candidates` (instruction maps / rebuilt programs), `parents`, `val_aggregate_scores` (mean per candidate), `val_subscores` (per candidate per instance), `per_val_instance_best_candidates` (the frontier sets), `discovery_eval_counts`, `total_metric_calls`, `num_full_val_evals`, `best_outputs_valset?`, `seed`. `best_idx = argmax(val_aggregate_scores)` with lowest-index-wins ties; `highest_score_achieved_per_val_task[valId]` = the frontier score. (The engine's internal best-program policy also tie-breaks on eval coverage, but with full evals the two rules coincide, and `compile` uses plain argmax.)

## Budget accounting (what counts as a metric call)

- The seed's full valset eval (assigned directly at init).
- The parent minibatch eval and child minibatch eval of every reflective iteration — including for proposals later rejected.
- The full valset eval of each accepted candidate.
- The merge subsample eval — including for rejected merges.
- Skipped rounds (no trajectories / all-perfect / empty dataset / empty proposal) bill only what they actually ran; an empty proposal skips the child eval entirely.

## Port checklist / gotchas

- One shared RNG drives parent selection, batch shuffling, and merge sampling; a second one picks trace steps in the reflective dataset. Distributions matter; exact Python streams don't.
- Frontier updates and acceptance use exact float comparisons — no epsilon. Minibatch acceptance compares **sums**; frontier/best-candidate tracking uses **means** over val instances. Merge acceptance is `>=`; reflection acceptance is strict `>`.
- The round-robin component cursor advances on the parent even when the proposal is rejected, and a new child inherits `max(parent cursors)`.
- A merge iteration never also runs reflection, accepted or not.
- Every accepted candidate costs a full valset eval immediately — valset size is the main cost lever.
- Failed rollouts must return `failure_score` rows, never throw; a thrown eval aborts the whole run.
- Trace steps are matched to components by signature equality, not predictor name — after `with_instructions` the copies still compare equal because instructions are part of the signature being compared on both sides.
- `skip_perfect_score` plus a metric capped at `perfect_score` means an already-perfect minibatch produces no proposal — the round just burns the parent minibatch eval.
