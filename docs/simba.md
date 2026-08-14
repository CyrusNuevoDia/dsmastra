# SIMBA — Stochastic Introspective Mini-Batch Ascent

Authoritative flow of DSPy's SIMBA optimizer, traced from `dspy/dspy/teleprompt/simba.py` and `simba_utils.py`, written so we can replicate it in `src/`. SIMBA is a mini-batch hill-climber: each step it samples many stochastic rollouts of the program on a minibatch, finds the examples with the biggest score spread between rollouts, and turns the best/worst contrast into either an appended few-shot demo or an appended natural-language rule, then keeps whichever candidate wins on that same minibatch.

## Source files

- [`dspy/dspy/teleprompt/simba.py`](../dspy/dspy/teleprompt/simba.py) — `SIMBA.__init__` and the whole `compile` loop (minibatching, buckets, candidate generation, final selection)
- [`dspy/dspy/teleprompt/simba_utils.py`](../dspy/dspy/teleprompt/simba_utils.py) — `prepare_models_for_resampling`, `wrap_program`, `append_a_demo`, `append_a_rule`, the `OfferFeedback` signature, `inspect_modules`, `recursive_mask`
- [`dspy/docs/docs/api/optimizers/SIMBA.md`](../dspy/docs/docs/api/optimizers/SIMBA.md) — upstream API doc (intent, not mechanics)

## Configuration

| param | default | meaning |
| --- | --- | --- |
| `metric` | required | `(example, prediction) -> { score: number }`, plus any extra fields carried through as feedback metadata. Upstream also accepts a bare number; ours always returns the object |
| `bsize` | 32 | minibatch size |
| `num_candidates` | 6 | rollouts per example per step, and candidates produced per step |
| `max_steps` | 8 | optimization steps |
| `max_demos` | 4 | soft cap on few-shot demos per predictor (see demo dropping) |
| `demo_input_field_maxlen` | 100 000 | truncation length for demo input fields |
| `prompt_model` | session LM | LM used to write rules (`append_a_rule`) |
| `temperature_for_sampling` | 0.2 | softmax temp when picking source programs for rollouts |
| `temperature_for_candidates` | 0.2 | softmax temp when picking the source program to mutate |
| `teacher_settings` | none | optional teacher LM used as the first rollout model |

Strategies: `[append_a_demo, append_a_rule]` when `max_demos > 0`, else `[append_a_rule]` only. Chosen **uniformly at random** per candidate (50/50 — there are no weights).

Two RNGs, both seeded with `seed`: a general-purpose one (shuffling, softmax sampling, strategy choice, demo-index picks) and one used only for the Poisson draw in demo dropping. Only the distributions matter for a port, not the exact streams.

## State

- `programs`: growing pool; index 0 is a deepcopy of the student (the baseline). Every candidate ever built gets appended with the next index — the pool is never trimmed.
- `programScores[idx]`: list of per-example minibatch scores for that program. **The baseline's list stays empty forever**, so its average score is pinned at 0.0 — it's never rescored.
- `winningPrograms`: `[student]`, plus one winner appended per step.
- `trialLogs[stepIdx]`: bookkeeping.

Helpers used everywhere:

- `avg(idx)` = mean of `programScores[idx]`, 0.0 if empty.
- `topKPlusBaseline(k)`: sort all program indices by `avg` descending (stable sort — ties break toward lower index), take first `k`; if the baseline (0) isn't among them, **overwrite the last slot with 0**; dedupe preserving order. May return fewer than `k`.
- `softmaxSample(idxs, T)`: weights `exp(avg(i)/T)` (no max-subtraction stability guard — fine for scores in [0,1]); if the weight sum is ≤ 0, fall back to uniform. With all-zero scores this is uniform.

## Per-step flow (`for step in 0..max_steps`)

```text
step
├─ 1. minibatch: next bsize examples from a shuffled index list
│     (if fewer than bsize remain, reshuffle and restart — the tail is dropped)
├─ 2. models = prepareModelsForResampling(programs[0], num_candidates)
│     ├─ optional teacher LM first (kept at its own temperature)
│     └─ copies of the base LM with temperature=1.0 and distinct rollout_ids
│        (rollout_id is a cache-buster so identical calls sample differently)
├─ 3. rollout sampling — model-major, example-minor:
│     for each model, for each example:
│        src = softmaxSample(topKPlusBaseline(num_candidates), T_sampling)
│        run a fresh deepcopy of programs[src] with that model on that example
│     → bsize × num_candidates rollouts, each capturing
│        {prediction, trace, score, example, outputMetadata}
│        (any program or metric exception ⇒ score 0.0)
├─ 4. buckets: for each example, its num_candidates rollouts sorted by score desc
│     ordered by (max−min gap, max score, max−avg gap) lexicographic desc
│     — highest-contrast examples with a high ceiling go first
├─ 5. candidate generation — iterate buckets until num_candidates+1 candidates:
│     ├─ src = softmaxSample(topKPlusBaseline(num_candidates), T_candidates)
│     ├─ candidate = deepcopy(programs[src])
│     ├─ drop demos (Poisson; see below)
│     ├─ strategy = random choice: append_a_demo | append_a_rule
│     └─ on strategy exception: skip bucket; on strategy no-op: candidate kept anyway
├─ 6. evaluate every candidate on the SAME minibatch
├─ 7. winner = argmax mean score (first max wins ties) → winningPrograms
└─ 8. register ALL candidates (with their score lists) into the pool
```

Percentile guards for the strategies come from step 3's full score pool: `p10` and `p90` of all `bsize × num_candidates` scores (linear-interpolation percentiles, NumPy default).

### Demo dropping (before applying a strategy)

`max_demos` is not a hard cap — it's enforced probabilistically here:

```text
cap        = max_demos > 0 ? max_demos : 3
numDemos   = max demos across the candidate's predictors
toDrop     = max(poisson(numDemos / cap), numDemos >= cap ? 1 : 0)
toDrop     = min(toDrop, numDemos)
dropIdxs   = toDrop draws of randInt(numDemos)   // with replacement → may drop fewer
every predictor removes the demos at dropIdxs
```

A full predictor drops ~1 demo per candidate on average, and at/over the cap at least one drop is forced.

### Strategy A — `append_a_demo`

Take the bucket's best rollout. If its score ≤ p10, do nothing. Otherwise walk its trace: for each predictor step, truncate any input field longer than `demo_input_field_maxlen` (append `"\n\t\t... <TRUNCATED FOR BREVITY>"`), build a demo `{...inputs, ...outputs, augmented: true}`, and keep **only the last demo per predictor** in the trajectory. Append that one demo to the matching predictor of the new candidate.

Predictor identity matters: rollout traces reference the rollout copy's predictor objects, so a map from predictor object identity → predictor name (built during step 3) bridges to the candidate's predictors by name. In TS, give each predictor a stable name/id rather than relying on object identity.

### Strategy B — `append_a_rule`

Take the bucket's best (`good`) and worst (`bad`) rollouts. Guards: skip if `good.score <= p10` or `bad.score >= p90`. If there's no real contrast (`good.score <= bad.score`): when good is above p90, blank the bad side (empty trace, score `"N/A"`, placeholder prediction); otherwise blank the good side — so the LM sees only the informative half.

Then call the prompt model once with the `OfferFeedback` signature and merge the result: for each module name present in the returned `module_advice` dict, **append** `"\n\n" + advice` to that predictor's instructions (advice accumulates across steps).

Inputs to `OfferFeedback`, in declaration order (order matters for prompt layout). Every non-string value is serialized to pretty (2-space-indented) JSON; non-serializable values are recursively replaced with `<non-serializable: TypeName>`:

| field | description text |
| --- | --- |
| `program_code` | "The code of the program that we are analyzing" — source code of the program class |
| `modules_defn` | "The definition of each module in the program, including its I/O" — see rendering below |
| `program_inputs` | "The inputs to the program that we are analyzing" — the example's input fields |
| `oracle_metadata` | "Any (hidden) metadata about the training set instance we're analyzing" — the example's labels |
| `worse_program_trajectory` | "The trajectory of the program's execution, showing each module's I/O" — list of `{module_name, inputs, outputs}` |
| `worse_program_outputs` | "The outputs of the program that we are analyzing" |
| `worse_reward_value` | "The reward value assigned to the program's outputs" (float; `"N/A"` when blanked) |
| `worse_reward_info` | "Additional information that might be helpful to understanding the assigned reward value." — the metric's extra metadata |
| `better_program_trajectory` / `better_program_outputs` / `better_reward_value` / `better_reward_info` | same descriptions, for the better rollout |
| `module_names` | "The names of the modules in the program, for which we seek advice" (list of strings) |

Outputs: `discussion: string` — "Discussing blame of where each module went wrong, if it did" — then `module_advice: Record<moduleName, string>`. Only advice keys matching a predictor name are applied (`if name in advice`); extra keys are ignored.

`modules_defn` is rendered per predictor, blocks separated by a line of 80 hyphens:

```text
Module {name}

	Input Fields:
		{field descriptions}
	Output Fields:
		{field descriptions}
	Original Instructions: {dedented instructions, continuation lines indented two tabs}
```

`OfferFeedback` instruction text (verbatim):

```
You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
similar inputs in the future.

The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

In your advice:
- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
- Ensure each unique module name appears exactly once as a key in the advice dictionary.
```

The `module_advice` field description (it shapes the output): _"For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory."_

## Final selection

After the loop, pick `num_candidates + 1` programs **evenly spaced across the winner timeline** (`winningPrograms`), always including index 0 (the untouched student) and the last winner — not the top-K by score. (Python uses banker's rounding for the spacing indices.) Evaluate each on the **full trainset**; return a deepcopy of the argmax (first max wins ties, i.e. ties favor the less-evolved program), with the sorted `{score, program}` list and trial logs attached.

## Port checklist / gotchas

- Keep the model-major/example-minor rollout layout — bucket extraction strides by `bsize` and silently mixes examples if the ordering changes.
- The baseline's pinned 0.0 average means it always keeps softmax weight `exp(0)=1`; with T=0.2 a program averaging 0.6 is ~20× more likely to be picked. Preserve this — it's the exploration floor.
- Every rollout uses a fresh deepcopy of a (possibly different) source program per (model, example) cell; a bucket's rollouts can come from different programs.
- Rollout models always derive from the **baseline** program's LM, never from evolved candidates.
- Candidates whose strategy was a no-op are still evaluated and registered, so the pool holds near-duplicates. Failure score for crashed rollouts is 0.0.
- `topKPlusBaseline` can evict a genuine top-k program to make room for the baseline.
- Both strategies mutate the shared rollout records in place (input truncation in `append_a_demo`, `"N/A"` blanking in `append_a_rule`), and those mutations persist for later buckets that share the same rollout objects. A port should treat this as incidental, not load-bearing — copying per bucket is safer.
- The demo-drop index set is drawn once and applied to **every** predictor of the candidate, and draws are with replacement, so the realized drop count can be lower than the sampled one.
- `append_a_rule` runs the prompt model with an empty trace context so the advice call itself isn't captured as a trajectory.
- The upstream `trial_logs["train_score"]` bookkeeping indexes by candidate position rather than step (and would break if `num_candidates > max_steps - 1`) — don't replicate that; log final scores keyed by candidate.
