import { createStep, createWorkflow } from "@mastra/core/workflows";
import { z } from "zod";
import { createScorer, extractWorkflowTrajectory } from "@mastra/core/evals";
import { RequestContext } from "@mastra/core/request-context";
import { Output, generateText } from "ai";
//#region src/collections.ts
/**
* Checked collection access.
*
* `noUncheckedIndexedAccess` makes every `array[i]` and `map.get(k)` return
* `T | undefined`, which is correct — the index really might be out of range. The
* tempting fix is `array[i] as T`, but that just tells the compiler to stop asking
* and turns an out-of-range read into an `undefined` that surfaces somewhere far
* away. These helpers do the check once and throw at the point of the mistake, so
* the invariant "this index is in range" is enforced rather than asserted.
*/
/** The element at `index`, or a thrown error naming what was being indexed. */
const at = (items, index, what) => {
	const item = items[index];
	if (item === void 0) throw new Error(`${what}: index ${index} is out of range (length ${items.length})`);
	return item;
};
/** The first element, or a thrown error naming what was empty. */
const first = (items, what) => at(items, 0, what);
/** The last element, or a thrown error naming what was empty. */
const last = (items, what) => at(items, items.length - 1, what);
/** The property at `key`, or a thrown error naming the record that lacked it. */
const prop = (record, key, what) => {
	const value = record[key];
	if (value === void 0) throw new Error(`${what}: no entry for key ${key}`);
	return value;
};
/** The popped element, or a thrown error naming the stack that was empty. */
const pop = (items, what) => {
	const item = items.pop();
	if (item === void 0) throw new Error(`${what}: popped an empty stack`);
	return item;
};
/** The value for `key`, or a thrown error naming the map that lacked it. */
const get = (map, key, what) => {
	const value = map.get(key);
	if (value === void 0) throw new Error(`${what}: no entry for key ${String(key)}`);
	return value;
};
//#endregion
//#region src/random.ts
/** Seeded RNG + sampling utilities shared by the optimizers. */
/** mulberry32 — small seeded PRNG; only the distributions matter for the port. */
const createRNG = (seed) => {
	let state = seed >>> 0;
	const rng = () => {
		state = state + 1831565813 | 0;
		let t = state;
		t = Math.imul(t ^ t >>> 15, t | 1);
		t ^= t + Math.imul(t ^ t >>> 7, t | 61);
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
	return Object.defineProperty(rng, "state", {
		get: () => state,
		set: (next) => {
			state = next >>> 0;
		}
	});
};
/** An RNG resumed from a checkpointed `state` (uint32), mid-stream. */
const restoreRNG = (state) => {
	const rng = createRNG(0);
	rng.state = state;
	return rng;
};
/** Knuth's Poisson sampler. */
const samplePoisson = (rng, lambda) => {
	if (lambda <= 0) return 0;
	const limit = Math.exp(-lambda);
	let k = 0;
	let p = 1;
	do {
		k += 1;
		p *= rng();
	} while (p > limit);
	return k - 1;
};
/** In-place Fisher–Yates shuffle. */
const shuffle = (rng, items) => {
	for (let i = items.length - 1; i > 0; i -= 1) {
		const j = Math.floor(rng() * (i + 1));
		const a = at(items, i, "shuffle");
		items[i] = at(items, j, "shuffle");
		items[j] = a;
	}
};
/** Python `random.sample` parity: k items without replacement. */
const sample = (rng, items, k) => {
	if (k > items.length) throw new Error("Sample larger than population");
	const pool = [...items];
	shuffle(rng, pool);
	return pool.slice(0, k);
};
/**
* Pick an item with probability proportional to its weight; uniform fallback
* when the weight sum is not positive or not finite.
*/
const weightedChoice = (rng, items, weights) => {
	const total = weights.reduce((acc, w) => acc + w, 0);
	if (total <= 0 || !Number.isFinite(total)) return at(items, Math.floor(rng() * items.length), "weightedChoice pool");
	let r = rng() * total;
	for (const [idx, weight] of weights.entries()) {
		r -= weight;
		if (r <= 0) return at(items, idx, "weightedChoice pool");
	}
	return last(items, "weightedChoice pool");
};
/**
* Python `random.choices` parity: throws when the weight total is not
* positive, exactly like the reference (which aborts the run under
* raise_on_exception). Use where the Python side passes score weights.
*/
const weightedChoiceStrict = (rng, items, weights) => {
	if (weights.reduce((acc, w) => acc + w, 0) <= 0) throw new Error("Total of weights must be greater than zero");
	return weightedChoice(rng, items, weights);
};
//#endregion
//#region src/utils.ts
/** Structural equality via JSON rendering — for JSON-safe values only (the
* optimizer currency: examples, prompts, fields). Key order matters, which is
* fine here because compared values come from the same construction sites. */
const isEqualJSON = (a, b) => JSON.stringify(a) === JSON.stringify(b);
//#endregion
//#region src/optimizers/bootstrap.ts
/** dspy.settings.max_errors default. */
const DEFAULT_MAX_ERRORS = 10;
const cloneExample = (example) => ({
	inputData: structuredClone(example.inputData),
	outputData: structuredClone(example.outputData)
});
const examplesEqual = (a, b) => isEqualJSON(a.inputData, b.inputData) && isEqualJSON(a.outputData, b.outputData);
/** FNV-1a over the JSON rendering — stands in for dspy's Hasher.hash. */
const contentHash = (value) => {
	const text = JSON.stringify(value);
	let hash = 2166136261;
	for (let i = 0; i < text.length; i += 1) {
		hash ^= text.charCodeAt(i);
		hash = Math.imul(hash, 16777619);
	}
	return hash >>> 0;
};
/** Reset copy: fresh clone with step examples cleared (dspy reset_copy). */
const resetCopy = (program) => {
	const copy = program.clone();
	for (const step of copy.steps) step.examples = [];
	return copy;
};
/**
* dspy.teleprompt.vanilla.LabeledFewShot: install k labeled examples as
* few-shot examples on a reset copy of the student. Each step draws its own
* sample from the same seed-0 RNG stream; nothing else is shuffled.
*/
const labeledFewShotProgram = (student, trainingSet, k = 16) => {
	const compiled = resetCopy(student);
	if (trainingSet.length === 0) return compiled;
	const rng = createRNG(0);
	for (const step of compiled.steps) step.examples = sample(rng, trainingSet, Math.min(k, trainingSet.length)).map(cloneExample);
	return compiled;
};
/**
* _prepare_student_and_teacher: reset copy for the student, deep copy for
* the teacher — then LabeledFewShot over a reset teacher copy when labeled
* examples are requested (our programs carry no _compiled flag; a provided
* teacher is treated as uncompiled, see the doc's deviation list). Validates
* _prepare_predictor_mappings: same structure, matched by position + id.
*/
const prepareStudentAndTeacher = (studentProgram, trainingSet, options) => {
	const { maxLabeledExamples = 16 } = options;
	const student = resetCopy(studentProgram);
	let teacher = (options.teacher ?? studentProgram).clone();
	if (maxLabeledExamples > 0) teacher = labeledFewShotProgram(teacher, trainingSet, maxLabeledExamples);
	if (student.steps.length !== teacher.steps.length) throw new Error("Student and teacher must have the same number of steps.");
	for (const [idx, studentStep] of student.steps.entries()) {
		const teacherStep = teacher.steps[idx];
		if (studentStep.id !== teacherStep?.id) throw new Error("Student and teacher must have the same program structure.");
		if (studentStep === teacherStep) throw new Error("Student and teacher must be different objects.");
	}
	return {
		student,
		teacher
	};
};
/**
* One teacher attempt over one example: hide any installed example equal to
* the one being bootstrapped, roll the teacher out with trace capture, and
* decide acceptance through the metric. Rounds past the first take a fresh
* rollout at temperature=1.0 to bypass caches — the round index maps onto the
* seed parameter, exactly like SIMBA's prepareModelsForResampling. Throws
* when the rollout or the metric throws; the caller owns error counting.
*/
const runBootstrapAttempt = async (teacher, example, roundIdx, options) => {
	const { metric, metricThreshold, teacherSettings } = options;
	const trace = [];
	const ctx = {
		model: teacherSettings?.model,
		temperature: teacherSettings?.temperature,
		trace
	};
	if (roundIdx > 0) {
		ctx.seed = roundIdx;
		ctx.temperature = 1;
	}
	const exampleCache = teacher.steps.map((step) => step.examples);
	for (const step of teacher.steps) step.examples = step.examples.filter((installed) => !examplesEqual(installed, example));
	let prediction;
	try {
		prediction = await teacher.run(example.inputData, ctx);
	} finally {
		for (const [idx, step] of teacher.steps.entries()) step.examples = at(exampleCache, idx, "teacher example cache");
	}
	if (!metric) return {
		success: true,
		trace
	};
	const { score } = await metric(example, prediction, trace, ctx);
	return {
		success: metricThreshold === void 0 ? score > 0 : score >= metricThreshold,
		trace
	};
};
/**
* Fold one accepted attempt's trace into the per-step example pools. Multiple
* traces for one step in one example: keep ONE, sampled 50/50 from the first
* N-1 or the last, seeded by example content.
*/
const harvestTraceExamples = (id2traces, trace) => {
	const examplesById = /* @__PURE__ */ new Map();
	for (const traceStep of trace) {
		if (!id2traces.has(traceStep.stepId)) continue;
		const harvested = {
			inputData: traceStep.inputData,
			outputData: traceStep.outputData
		};
		const list = examplesById.get(traceStep.stepId) ?? [];
		list.push(harvested);
		examplesById.set(traceStep.stepId, list);
	}
	for (const [stepId, harvested] of examplesById) {
		let kept = harvested;
		if (harvested.length > 1) {
			const rng = createRNG(contentHash(harvested));
			kept = [rng() < .5 ? at(harvested, Math.floor(rng() * (harvested.length - 1)), "trace examples") : last(harvested, "trace examples")];
		}
		id2traces.get(stepId)?.push(...kept);
	}
};
/**
* _train: bootstrapped examples first, labeled backfill after. The
* un-bootstrapped pool is seed-0 shuffled, and the Python quirk that
* rawExamples is REASSIGNED to each step's sample is preserved, so later
* steps draw from the shrinking pool. Mutates and returns `student`.
*/
const installTrainExamples = (student, id2traces, unBootstrapped, options) => {
	const validation = [...unBootstrapped];
	shuffle(createRNG(0), validation);
	const rng = createRNG(0);
	let rawExamples = validation;
	for (const step of student.steps) {
		const harvested = (id2traces.get(step.id) ?? []).slice(0, options.maxFewShotExamples);
		const sampleSize = Math.max(0, Math.min(options.maxLabeledExamples - harvested.length, rawExamples.length));
		rawExamples = sample(rng, rawExamples, sampleSize);
		step.examples = [...harvested, ...rawExamples.map(cloneExample)];
	}
	return student;
};
/**
* BootstrapFewShot.compile: run a teacher over the trainingSet, capture the trace
* of every metric-passing run as bootstrapped examples per step, and fill the
* remaining slots with raw labeled examples. The durable workflow driver in
* bootstrap-few-shot.ts runs the same helpers one attempt per loop iteration.
*/
const bootstrapFewShotProgram = async (studentProgram, trainingSet, options = {}) => {
	const { maxErrors = DEFAULT_MAX_ERRORS, maxFewShotExamples = 4, maxLabeledExamples = 16, maxRounds = 1 } = options;
	const { student, teacher } = prepareStudentAndTeacher(studentProgram, trainingSet, options);
	const id2traces = new Map(student.steps.map((step) => [step.id, []]));
	let errorCount = 0;
	const bootstrapOneExample = async (example, roundIdx) => {
		let success = false;
		try {
			const attempt = await runBootstrapAttempt(teacher, example, roundIdx, options);
			({success} = attempt);
			if (success) harvestTraceExamples(id2traces, attempt.trace);
		} catch (error) {
			errorCount += 1;
			if (errorCount >= maxErrors) throw error;
			console.error(`Failed to run or evaluate example due to ${error}.`);
		}
		return success;
	};
	const bootstrapped = /* @__PURE__ */ new Set();
	for (const [exampleIdx, example] of trainingSet.entries()) {
		if (bootstrapped.size >= maxFewShotExamples) break;
		for (let roundIdx = 0; roundIdx < maxRounds; roundIdx += 1) if (await bootstrapOneExample(example, roundIdx)) {
			bootstrapped.add(exampleIdx);
			break;
		}
	}
	return installTrainExamples(student, id2traces, trainingSet.filter((_x, idx) => !bootstrapped.has(idx)), {
		maxFewShotExamples,
		maxLabeledExamples
	});
};
//#endregion
//#region src/schema.ts
/**
* The slice of a JSON-schema property we render. `z.toJSONSchema` hands back the
* full JSON-schema union (an object, or the `true`/`false` shorthand), so this
* parses the two fields we use instead of narrowing the representation by hand.
*/
const schemaPropertySchema = z.object({
	description: z.string().optional(),
	type: z.string().optional()
});
/** Top-level properties of a zod object schema; `{}` when the schema can't convert. */
const schemaProperties = (schema) => {
	if (!schema) return {};
	try {
		const { properties } = z.toJSONSchema(schema);
		if (!properties) return {};
		const result = {};
		for (const [name, property] of Object.entries(properties)) {
			const parsed = schemaPropertySchema.safeParse(property);
			if (parsed.success) result[name] = parsed.data;
		}
		return result;
	} catch {
		return {};
	}
};
//#endregion
//#region src/prompting.ts
/**
* Prompt-template formatting shared across the steps and optimizers: rendering
* values, schemas, and few-shot examples into prompt text, and extracting
* instruction text back out of LM responses. Optimizer-specific prompt
* templates stay with their optimizers.
*/
/** The prompt a step sends: description, few-shot examples, then the live input. */
const renderPrompt = (description, examples, inputData) => {
	const parts = [description];
	for (const example of examples) parts.push(`Example:\nInput:\n${JSON.stringify(example.inputData)}\nOutput:\n${JSON.stringify(example.outputData)}`);
	parts.push(`Input:\n${JSON.stringify(inputData)}`);
	return parts.join("\n\n");
};
/** Replace non-serializable values recursively, like dspy's recursive_mask. */
const recursiveMask = (value) => {
	if (value === null) return null;
	if (Array.isArray(value)) return value.map(recursiveMask);
	switch (typeof value) {
		case "boolean":
		case "number":
		case "string": return value;
		case "object": return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, recursiveMask(v)]));
		default: return `<non-serializable: ${typeof value}>`;
	}
};
/** A field value as prompt text: strings raw, everything else masked JSON. */
const serializeField = (value) => typeof value === "string" ? value : JSON.stringify(recursiveMask(value), null, 2);
/** Every field rendered with String(), for records shown verbatim in prompts. */
const stringifyFields = (fields) => Object.fromEntries(Object.entries(fields).map(([key, value]) => [key, String(value)]));
/** One `name (type): description` line per schema property. */
const fieldDescriptionLines = (schema) => Object.entries(schemaProperties(schema)).map(([name, prop]) => `${name} (${prop.type ?? "unknown"})${prop.description ? `: ${prop.description}` : ""}`).join("\n");
/** One `name: type` line per schema property — the compact structure sketch. */
const expectedStructure = (schema) => Object.entries(schemaProperties(schema)).map(([name, prop]) => `${name}: ${prop.type ?? "unknown"}`).join("\n");
const indentContinuations = (text) => ["", ...text.split("\n")].join("\n		");
const MODULE_SEPARATOR = "-".repeat(80);
/** Every module's I/O fields and current instructions, DSPy inspect-style. */
const inspectModules = (program) => {
	const blocks = [MODULE_SEPARATOR];
	for (const step of program.steps) blocks.push(`Module ${step.id}`, `\n\tInput Fields:${indentContinuations(fieldDescriptionLines(step.inputSchema))}`, `\tOutput Fields:${indentContinuations(fieldDescriptionLines(step.outputSchema))}`, `\tOriginal Instructions: ${indentContinuations(step.description)}`, MODULE_SEPARATOR);
	return blocks.map((block) => block.replaceAll(/^\n+|\n+$/gu, "")).join("\n");
};
const MAX_HEADER_DEPTH = 6;
/**
* Byte-for-byte port of the Python renderer: scalars end with a blank line
* (`value\n\n`), headers with a single newline, empty dicts/lists add a bare
* newline, and the depth cap applies on recursion.
*/
const renderValue = (value, level) => {
	const header = "#".repeat(level);
	const nextLevel = Math.min(level + 1, MAX_HEADER_DEPTH);
	if (Array.isArray(value)) {
		let s = "";
		for (const [k, item] of value.entries()) s += `${header} Item ${k + 1}\n${renderValue(item, nextLevel)}`;
		if (value.length === 0) s += "\n";
		return s;
	}
	if (typeof value === "object" && value !== null) {
		const entries = Object.entries(value);
		let s = "";
		for (const [key, sub] of entries) s += `${header} ${key}\n${renderValue(sub, nextLevel)}`;
		if (entries.length === 0) s += "\n";
		return s;
	}
	return `${String(value).trim()}\n\n`;
};
/** Markdown rendering of keyed example records, one `# Example n` per entry. */
const renderSideInfo = (examples) => examples.map((example, n) => {
	let s = `# Example ${n + 1}\n`;
	for (const [key, value] of Object.entries(example)) s += `## ${key}\n${renderValue(value, 3)}`;
	return s;
}).join("\n\n");
const LANGUAGE_TAG = /^\S*\n/u;
const LEADING_FENCE = /^```\S*\n?/u;
/**
* Byte-for-byte port of Python's output_extractor (which receives the
* response pre-stripped): text between the first and last fences with a
* leading language tag stripped; incomplete blocks fall back to stripping a
* leading fence (+ optional language tag and newline) or a trailing fence,
* else the whole trimmed response.
*/
const extractInstructionText = (response) => {
	const lmOut = response.trim();
	const start = lmOut.indexOf("```") + 3;
	const end = lmOut.lastIndexOf("```");
	if (start >= end) {
		if (lmOut.startsWith("```")) {
			const match = LEADING_FENCE.exec(lmOut);
			if (match) return lmOut.slice(match[0].length).trim();
			return lmOut;
		}
		if (lmOut.endsWith("```")) return lmOut.slice(0, -3).trim();
		return lmOut;
	}
	let content = lmOut.slice(start, end);
	const tag = LANGUAGE_TAG.exec(content);
	if (tag) content = content.slice(tag[0].length);
	return content.trim();
};
//#endregion
//#region src/step.ts
/**
* The request-context key rollouts use to hand a RunContext to steps executed
* by Mastra's engine.
*/
const RUN_CONTEXT_KEY = "dsmastra";
const declareStep = (config) => {
	const { description, examples, id, inputSchema, model, outputSchema, scorers, ...settings } = config;
	let declarative;
	const execute = async ({ inputData }, ctx) => {
		const generated = await generateText({
			...declarative.settings,
			model: ctx?.model ?? declarative.model,
			output: Output.object({ schema: declarative.outputSchema }),
			prompt: renderPrompt(declarative.description, declarative.examples, inputData),
			seed: ctx?.seed ?? declarative.settings.seed,
			temperature: ctx?.temperature ?? declarative.settings.temperature
		});
		const outputData = declarative.outputSchema.parse(generated.output);
		ctx?.trace?.push({
			inputData,
			outputData,
			stepId: declarative.id
		});
		return outputData;
	};
	const runStep = (params, ctx) => {
		if (params.requestContext) {
			const engineCtx = params.requestContext.get(RUN_CONTEXT_KEY);
			return execute({ inputData: inputSchema.parse(params.inputData) }, engineCtx);
		}
		return execute(params, ctx);
	};
	const step = createStep({
		description,
		execute: async ({ inputData, requestContext }) => await runStep({
			inputData,
			requestContext
		}),
		id,
		inputSchema,
		outputSchema,
		scorers
	});
	declarative = Object.assign(step, {
		clone: () => declareStep({
			...declarative.settings,
			description: declarative.description,
			examples: declarative.examples,
			id,
			inputSchema,
			model: declarative.model,
			outputSchema,
			scorers: declarative.scorers
		}),
		description,
		examples: structuredClone(examples ?? []),
		execute: runStep,
		inputSchema,
		model,
		outputSchema,
		scorers,
		settings
	});
	return declarative;
};
//#endregion
//#region src/optimizers/utils.ts
const isDeclarativeStep = (step) => "description" in step && "examples" in step && "clone" in step;
const singleEntrySteps = (entry) => entry.type === "step" ? [entry.step] : [];
/**
* A workflow's declarative steps in graph order, read from Mastra's step graph.
* The walk descends into parallel, branch, loop, and foreach entries; steps
* that weren't built with `declareStep` (agents, tools, mappings, plain steps,
* nested workflows) pass through untouched — the engine runs them, the
* optimizers just don't tune them. Steps inside a nested workflow are opaque.
*/
const declarativeSteps = (workflow) => workflow.stepGraph.flatMap((entry) => {
	switch (entry.type) {
		case "parallel":
		case "conditional": return entry.steps.flatMap(singleEntrySteps);
		case "loop":
		case "foreach": return singleEntrySteps(entry.step);
		default: return "step" in entry ? [entry.step] : [];
	}
}).filter((step) => isDeclarativeStep(step));
const gates = /* @__PURE__ */ new WeakMap();
const acquire = async (workflow, holder) => {
	let gate = gates.get(workflow);
	if (!gate) {
		gate = {
			count: 0,
			holder: null,
			wake: []
		};
		gates.set(workflow, gate);
	}
	while (gate.count > 0 && gate.holder !== holder) {
		const { promise, resolve } = Promise.withResolvers();
		gate.wake.push(resolve);
		await promise;
	}
	const fresh = gate.holder !== holder;
	gate.holder = holder;
	gate.count += 1;
	return {
		fresh,
		gate
	};
};
const releaseGate = (gate) => {
	gate.count -= 1;
	if (gate.count === 0) {
		gate.holder = null;
		for (const wake of gate.wake.splice(0)) wake();
	}
};
/**
* Wrap a workflow as a Program whose rollouts run through Mastra's engine —
* any graph shape works, and every run shows up in Mastra observability. Each
* clone carries its own prompt state (a candidate); `run` installs that state
* onto the live steps under the gate, starts an engine run with the
* RunContext smuggled through the request context, and maps a non-success
* run to a throw (callers score it as a failure).
*/
const workflowToProgram = (workflow) => {
	const liveSteps = declarativeSteps(workflow);
	if (liveSteps.length === 0) throw new Error(`Workflow ${workflow.id} has no declareStep steps to optimize`);
	const code = `workflow ${workflow.id}: ${liveSteps.map((step) => step.id).join(" -> ")}`;
	const make = (steps) => ({
		clone: () => make(steps.map((step) => step.clone())),
		code,
		run: async (inputData, ctx) => {
			const { fresh, gate } = await acquire(workflow, steps);
			try {
				if (fresh) {
					const byId = new Map(steps.map((step) => [step.id, step]));
					for (const live of liveSteps) {
						const candidate = byId.get(live.id);
						if (!candidate) throw new Error(`Candidate program lost step ${live.id}`);
						live.description = candidate.description;
						live.examples = structuredClone(candidate.examples);
					}
				}
				const result = await (await workflow.createRun({ disableScorers: true })).start({
					inputData,
					requestContext: new RequestContext([[RUN_CONTEXT_KEY, ctx]])
				});
				if (result.status !== "success") throw result.status === "failed" && result.error instanceof Error ? result.error : /* @__PURE__ */ new Error(`Workflow ${workflow.id} run ended with status ${result.status}`);
				if (ctx) {
					ctx.target = {
						spanId: result.spanId,
						traceId: result.traceId
					};
					ctx.trajectory = extractWorkflowTrajectory(result.steps, result.stepExecutionPath);
				}
				return result.result;
			} finally {
				releaseGate(gate);
			}
		},
		steps
	});
	return make(liveSteps.map((step) => step.clone()));
};
/**
* Write a tuned program's prompt state back onto the workflow's live steps.
* Mutates in place on purpose: the caller's own step references — and any
* Mastra instance the workflow is registered with — see the tuned prompts
* immediately, and the same workflow instance stays re-optimizable.
*/
const applyProgram = (workflow, program) => {
	for (const step of declarativeSteps(workflow)) {
		const tuned = program.steps.find((candidate) => candidate.id === step.id);
		if (!tuned) throw new Error(`Tuned program lost step ${step.id}`);
		step.description = tuned.description;
		step.examples = structuredClone(tuned.examples);
	}
	return workflow;
};
/**
* Mean metric score of a program over a set of examples, rollouts running
* concurrently through the program (and so through Mastra's engine for
* workflow-backed programs). A failed rollout or metric throw scores 0, same
* as the optimizers' own rollout handling. Deliberate divergence from DSPy,
* which leaves its few-shot compilers unscored (the evaluation in
* `BootstrapFewShot._bootstrap` is commented out upstream).
*/
const evaluateProgram = async (program, examples, metric) => {
	const scores = await Promise.all(examples.map(async (example) => {
		const ctx = {};
		try {
			const { score } = await metric(example, await program.run(example.inputData, ctx), ctx.target);
			return score;
		} catch (error) {
			console.warn(error);
			return 0;
		}
	}));
	return scores.reduce((sum, score) => sum + score, 0) / Math.max(scores.length, 1);
};
const fieldsSchema = z.record(z.string(), z.unknown());
const exampleSchema = z.object({
	inputData: fieldsSchema,
	outputData: fieldsSchema
});
const promptsSchema = z.object({
	steps: z.record(z.string(), z.object({
		description: z.string(),
		examples: z.array(exampleSchema)
	})),
	version: z.literal(1)
});
/** Every optimizer workflow ends in this shape: candidate snapshots with their
* scores (best first for the search optimizers), plus the winner's score. */
const optimizerResultSchema = z.object({
	candidates: z.array(z.tuple([promptsSchema, z.object({ score: z.number() })])),
	score: z.number()
});
/**
* Rebuild a runnable candidate from its JSON snapshot: clone the base program
* and install the snapshot's descriptions and examples. Inverse of promptsOf,
* up to the base program's fixed config (models, schemas, settings).
*/
const programFromPrompts = (base, prompts) => {
	const built = base.clone();
	for (const step of built.steps) {
		const saved = prompts.steps[step.id];
		if (!saved) throw new Error(`Prompts lost step ${step.id}`);
		step.description = saved.description;
		step.examples = structuredClone(saved.examples);
	}
	return built;
};
/** Snapshot a program's tuned prompt state as a JSON-safe payload. */
const promptsOf = (program) => ({
	steps: Object.fromEntries(program.steps.map((step) => [step.id, {
		description: step.description,
		examples: structuredClone(step.examples)
	}])),
	version: 1
});
/**
* Apply saved prompts to a workflow's live steps, in place, and return the
* same workflow. Throws when the prompts' step ids don't exactly match the
* workflow's, so a stale snapshot fails loudly instead of half-applying.
* Parsing and storage are the caller's problem.
*/
const loadPrompts = (workflow, prompts) => {
	const steps = declarativeSteps(workflow);
	const workflowIds = steps.map((step) => step.id);
	const promptIds = Object.keys(prompts.steps);
	const missing = workflowIds.filter((id) => !(id in prompts.steps));
	const unknown = promptIds.filter((id) => !workflowIds.includes(id));
	if (missing.length > 0 || unknown.length > 0) throw new Error(`Prompts do not match the workflow's steps${missing.length > 0 ? `; missing: ${missing.join(", ")}` : ""}${unknown.length > 0 ? `; unknown: ${unknown.join(", ")}` : ""}`);
	for (const step of steps) {
		const saved = prompts.steps[step.id];
		if (!saved) throw new Error(`Prompts lost step ${step.id}`);
		step.description = saved.description;
		step.examples = structuredClone(saved.examples);
	}
	return workflow;
};
const compiledSchema = z.object({ prompts: promptsSchema });
const scoredSchema = z.object({
	prompts: promptsSchema,
	score: z.number()
});
/**
* The tail every compile-style optimizer workflow shares, as three steps over
* a `{ prompts }` payload: persist through savePrompts, score the compiled
* workflow over the trainingSet, and land the prompt state in place on the
* target workflow — returning the optimizer result shape.
*/
const finishingSteps = (workflow, options) => {
	const save = createStep({
		description: "Persist the compiled prompts through savePrompts",
		execute: async ({ inputData }) => {
			await options.savePrompts(inputData.prompts);
			return inputData;
		},
		id: "save",
		inputSchema: compiledSchema,
		outputSchema: compiledSchema
	});
	const evaluate = createStep({
		description: "Score the compiled workflow over the trainingSet",
		execute: async ({ inputData }) => {
			const compiled = programFromPrompts(workflowToProgram(workflow), inputData.prompts);
			const score = await evaluateProgram(compiled, options.trainingSet, options.metric);
			return {
				...inputData,
				score
			};
		},
		id: "evaluate",
		inputSchema: compiledSchema,
		outputSchema: scoredSchema
	});
	return {
		apply: createStep({
			description: "Land the compiled prompt state on the target workflow",
			execute: ({ inputData }) => {
				const { prompts } = inputData;
				loadPrompts(workflow, prompts);
				const winner = [prompts, { score: inputData.score }];
				return Promise.resolve({
					candidates: [winner],
					score: inputData.score
				});
			},
			id: "apply",
			inputSchema: scoredSchema,
			outputSchema: optimizerResultSchema
		}),
		evaluate,
		save
	};
};
//#endregion
//#region src/scorers.ts
/**
* Resolve a ScorerRef against the workflow. A string is strictly a Mastra
* registration key — resolved through the workflow's Mastra instance, never a
* fuzzy name or step lookup — so an unregistered workflow fails loudly.
*/
const resolveScorer = (workflow, ref) => {
	if (typeof ref !== "string") return ref;
	const { mastra } = workflow;
	if (!mastra) throw new Error(`Cannot resolve scorer "${ref}": workflow ${workflow.id} is not registered on a Mastra instance`);
	return mastra.getScorer(ref);
};
/**
* Narrow a scorer run's result to the internal metric shape, rejecting
* non-finite scores; the scorer's `reason` rides along as `feedback`.
*/
const metricResultOf = (scorer, result) => {
	const score = result.score;
	if (!Number.isFinite(score)) throw new TypeError(`Scorer ${scorer.id} returned a non-finite score`);
	return typeof result.reason === "string" ? {
		feedback: result.reason,
		score
	} : { score };
};
const scorerMetric = (scorer) => async (example, prediction, target) => {
	if (prediction === void 0) throw new Error(`Scorer ${scorer.id} has no prediction to score (the rollout failed)`);
	const runInput = {
		groundTruth: example.outputData,
		input: example.inputData,
		output: prediction,
		scoreSource: "experiment"
	};
	if (target?.traceId) {
		runInput.targetScope = "span";
		runInput.targetSpanId = target.spanId;
		runInput.targetTraceId = target.traceId;
	}
	return metricResultOf(scorer, await scorer.run(runInput));
};
/**
* A Mastra trajectory scorer as a trajectory metric. Runs the scorer the way
* Mastra's runEvals runs `type: "trajectory"` scorers — the Trajectory as
* `output`, the example's inputData as `input`, expected outputData as
* `groundTruth`. The trajectory is cloned first: its rawWorkflowResult aliases
* the very step outputs that become few-shot demos after acceptance, so a
* scorer mutating its run input must not reach them. No trace linkage: these
* runs happen mid-bootstrap, before any score target exists.
*/
const trajectoryScorerMetric = (scorer) => async (example, trajectory) => metricResultOf(scorer, await scorer.run({
	groundTruth: example.outputData,
	input: example.inputData,
	output: structuredClone(trajectory),
	scoreSource: "experiment",
	targetScope: "trajectory"
}));
/**
* A scorer where every expected output field must strictly equal the
* prediction's for a score of 1, else 0. A factory rather than a shared
* instance: registered scorers carry a mutable Mastra backpointer, so sharing
* one across Mastra instances would cross-wire their observability.
*/
const createExactMatchScorer = () => createScorer({
	description: "1 when every expected output field strictly equals the prediction's, else 0.",
	id: "exact-match"
}).generateScore(({ run }) => Object.entries(run.groundTruth ?? {}).every(([key, value]) => run.output?.[key] === value) ? 1 : 0);
//#endregion
//#region src/optimizers/bootstrap-few-shot.ts
/** The loop's whole world between attempts, as JSON. */
const bootstrapStateSchema = z.object({
	bootstrapped: z.array(z.number()),
	errorCount: z.number(),
	exampleIdx: z.number(),
	id2traces: z.record(z.string(), z.array(exampleSchema)),
	iteration: z.number(),
	roundIdx: z.number(),
	teacherPrompts: promptsSchema
});
/**
* BootstrapFewShot (dspy.teleprompt.bootstrap.BootstrapFewShot) as a Mastra
* workflow over the target `workflow`: a prepare step compiles the teacher's
* prompt state, a durable dountil loop runs ONE teacher attempt per
* iteration — capturing the trace of every scorer-passing run as few-shot
* examples per step — and a compile step backfills the remaining slots with
* labeled examples before the shared save/evaluate/apply tail. All inter-step
* state is JSON (teacher prompts, harvested traces, counters), so a
* storage-backed run resumes mid-bootstrap without redoing completed attempts.
*/
const createBootstrapFewShotWorkflow = (workflow, config) => {
	const { checkpoint, gate, maxErrors = 10, maxFewShotExamples = 4, maxLabeledExamples = 16, maxRounds = 1, scoreThreshold, teacher, trainingSet } = config;
	if (gate && scoreThreshold !== void 0) throw new Error("scoreThreshold gates the objective scorer, which does not gate acceptance when a gate is set — use gate.threshold instead");
	const metric = scorerMetric(resolveScorer(workflow, config.scorer));
	const gateMetric = gate && trajectoryScorerMetric(resolveScorer(workflow, gate.scorer));
	const acceptThreshold = gate ? gate.threshold : scoreThreshold;
	const attemptMetric = gateMetric ? (gold, _prediction, _trace, ctx) => {
		if (!ctx?.trajectory) throw new Error("Gate scorer has no trajectory to score (the teacher rollout did not run through Mastra's engine)");
		return gateMetric(gold, ctx.trajectory);
	} : (gold, prediction) => metric(gold, prediction ?? void 0);
	const done = (state) => state.bootstrapped.length >= maxFewShotExamples || state.exampleIdx >= trainingSet.length;
	const prepare = createStep({
		description: "Compile the teacher's prompt state (labeled few-shot install)",
		execute: () => {
			const prepared = prepareStudentAndTeacher(workflowToProgram(workflow), [...trainingSet], {
				maxLabeledExamples,
				teacher: teacher && workflowToProgram(teacher)
			});
			return Promise.resolve({
				bootstrapped: [],
				errorCount: 0,
				exampleIdx: 0,
				id2traces: Object.fromEntries(prepared.student.steps.map((step) => [step.id, []])),
				iteration: 0,
				roundIdx: 0,
				teacherPrompts: promptsOf(prepared.teacher)
			});
		},
		id: "prepare",
		inputSchema: z.object({}),
		outputSchema: bootstrapStateSchema
	});
	const attempt = createStep({
		description: "One teacher attempt: rollout, gate, harvest on success",
		execute: async ({ inputData, resumeData, suspend }) => {
			const state = inputData;
			if (done(state)) return state;
			if (!resumeData && await checkpoint?.({ iteration: state.iteration })) return await suspend({ iteration: state.iteration });
			const teacherBase = teacher ? workflowToProgram(teacher) : workflowToProgram(workflow);
			const builtTeacher = programFromPrompts(teacherBase, state.teacherPrompts);
			const example = at([...trainingSet], state.exampleIdx, "trainingSet");
			const next = {
				...state,
				id2traces: structuredClone(state.id2traces),
				iteration: state.iteration + 1
			};
			let success = false;
			try {
				const result = await runBootstrapAttempt(builtTeacher, example, state.roundIdx, {
					metric: attemptMetric,
					...acceptThreshold !== void 0 && { metricThreshold: acceptThreshold },
					teacherSettings: config.teacherSettings
				});
				({success} = result);
				if (success) {
					const id2traces = new Map(Object.entries(next.id2traces));
					harvestTraceExamples(id2traces, result.trace);
					next.id2traces = Object.fromEntries(id2traces);
				}
			} catch (error) {
				next.errorCount += 1;
				if (next.errorCount >= maxErrors) throw error;
				console.error(`Failed to run or evaluate example due to ${error}.`);
			}
			if (success) {
				next.bootstrapped = [...next.bootstrapped, state.exampleIdx];
				next.exampleIdx += 1;
				next.roundIdx = 0;
			} else if (state.roundIdx + 1 >= maxRounds) {
				next.exampleIdx += 1;
				next.roundIdx = 0;
			} else next.roundIdx = state.roundIdx + 1;
			return next;
		},
		id: "attempt",
		inputSchema: bootstrapStateSchema,
		outputSchema: bootstrapStateSchema,
		resumeSchema: z.object({}),
		suspendSchema: z.object({ iteration: z.number() })
	});
	const compile = createStep({
		description: "Install harvested examples plus labeled backfill",
		execute: ({ inputData }) => {
			const state = inputData;
			const bootstrapped = new Set(state.bootstrapped);
			const student = installTrainExamples(prepareStudentAndTeacher(workflowToProgram(workflow), [...trainingSet], {
				maxLabeledExamples: 0,
				teacher: teacher && workflowToProgram(teacher)
			}).student, new Map(Object.entries(state.id2traces)), trainingSet.filter((_x, idx) => !bootstrapped.has(idx)), {
				maxFewShotExamples,
				maxLabeledExamples
			});
			return Promise.resolve({ prompts: promptsOf(student) });
		},
		id: "compile",
		inputSchema: bootstrapStateSchema,
		outputSchema: compiledSchema
	});
	const { apply, evaluate, save } = finishingSteps(workflow, {
		metric,
		savePrompts: config.savePrompts,
		trainingSet
	});
	return createWorkflow({
		id: `${workflow.id}.bootstrap-few-shot`,
		inputSchema: z.object({}),
		outputSchema: optimizerResultSchema
	}).then(prepare).dountil(attempt, ({ inputData }) => Promise.resolve(done(inputData))).then(compile).then(save).then(evaluate).then(apply).commit();
};
//#endregion
//#region src/optimizers/gepa/adapter.ts
const defaultFeedback = (score) => `This trajectory got a score of ${score}.`;
const runFeedbackMetric = async (metric, gold, prediction, trace, stepId, stepTrace) => {
	const { score, ...metadata } = await metric(gold, prediction, trace, stepId, stepTrace);
	const { feedback } = metadata;
	return {
		feedback: typeof feedback === "string" ? feedback : defaultFeedback(score),
		score
	};
};
const PARSE_FAILURE_OUTPUT = (raw) => `Couldn't parse the output as per the expected output format. The model's raw response was:\n\`\`\`\n${raw}\n\`\`\`\n\n`;
const PARSE_FAILURE_FEEDBACK_PREFIX = "Your output failed to parse. Follow this structure:\n";
const buildProposalPrompt = (currentInstructions, sideInfo) => `I provided an assistant with the following instructions to perform a task for me:
\`\`\`
${currentInstructions}
\`\`\`

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
\`\`\`
${sideInfo}
\`\`\`

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within \`\`\` blocks.`;
const createProgramAdapter = (config) => {
	const { adapterRNG, failureScore, metric, program, reflectionModel } = config;
	let warnedScoreMismatch = false;
	const proposeText = typeof reflectionModel === "function" ? reflectionModel : async (prompt) => {
		const { text } = await generateText({
			model: reflectionModel,
			prompt
		});
		return text;
	};
	const buildProgram = (candidate) => {
		const built = program.clone();
		for (const step of built.steps) {
			const description = candidate[step.id];
			if (description !== void 0) step.description = description;
		}
		return built;
	};
	const evaluate = async (batch, candidate, captureTraces) => {
		const built = buildProgram(candidate);
		const trajectories = await Promise.all(batch.map(async (example) => {
			const trace = [];
			const ctx = { trace };
			let prediction = null;
			try {
				prediction = await built.run(example.inputData, ctx);
			} catch (error) {
				console.warn(error);
			}
			let score = failureScore;
			try {
				({score} = await metric(example, prediction, trace, void 0, void 0, ctx.target));
			} catch (error) {
				console.warn(error);
			}
			return {
				example,
				prediction,
				score,
				trace
			};
		}));
		const batchResult = {
			outputData: trajectories.map((t) => t.prediction),
			scores: trajectories.map((t) => t.score)
		};
		if (captureTraces) batchResult.trajectories = trajectories;
		return batchResult;
	};
	const recordForStep = async (trajectory, traceStep, componentName) => {
		if (traceStep.parseFailure !== void 0) {
			const step = program.steps.find((s) => s.id === componentName);
			return {
				Feedback: PARSE_FAILURE_FEEDBACK_PREFIX + expectedStructure(step?.outputSchema),
				"Generated Outputs": PARSE_FAILURE_OUTPUT(traceStep.parseFailure),
				Inputs: stringifyFields(traceStep.inputData)
			};
		}
		const { feedback, score } = await runFeedbackMetric(metric, trajectory.example, trajectory.prediction, trajectory.trace, componentName, [traceStep]);
		if (score !== trajectory.score && config.warnOnScoreMismatch && !warnedScoreMismatch) {
			warnedScoreMismatch = true;
			console.warn("GEPA: step-level metric score differs from module-level score; using the module-level score.");
		}
		return {
			Feedback: feedback,
			"Generated Outputs": stringifyFields(traceStep.outputData),
			Inputs: stringifyFields(traceStep.inputData)
		};
	};
	/**
	* Pick the trace step to reflect on: the first parse failure if any remain,
	* a random step (adapter RNG) otherwise — unless the whole prediction
	* failed, which skips the example.
	*/
	const chooseStep = (trajectory, componentName) => {
		let steps = trajectory.trace.filter((step) => step.stepId === componentName);
		if (!config.addFormatFailureAsFeedback) steps = steps.filter((step) => step.parseFailure === void 0);
		if (steps.length === 0) return null;
		const failure = steps.find((step) => step.parseFailure !== void 0);
		if (failure) return failure;
		if (trajectory.prediction === null) return null;
		return at(steps, Math.floor(adapterRNG() * steps.length), "trace steps");
	};
	const makeReflectiveDataset = async (_candidate, evalBatch, componentsToUpdate) => {
		const dataset = {};
		for (const componentName of componentsToUpdate) {
			const records = [];
			for (const trajectory of evalBatch.trajectories ?? []) {
				const chosen = chooseStep(trajectory, componentName);
				if (!chosen) continue;
				records.push(await recordForStep(trajectory, chosen, componentName));
			}
			if (records.length > 0) dataset[componentName] = records;
		}
		return dataset;
	};
	const proposeNewTexts = async (candidate, reflectiveDataset, componentsToUpdate) => {
		const texts = {};
		for (const componentName of componentsToUpdate) {
			const examples = reflectiveDataset[componentName];
			if (!examples || examples.length === 0) continue;
			const prompt = buildProposalPrompt(candidate[componentName] ?? "", renderSideInfo(examples));
			const response = await proposeText(prompt);
			texts[componentName] = extractInstructionText(response);
		}
		return texts;
	};
	return {
		buildProgram,
		evaluate,
		makeReflectiveDataset,
		proposeNewTexts
	};
};
const toSubscores = (scores) => new Map(scores.map((score, validationId) => [validationId, score]));
/**
* Per-instance frontier update after a candidate's full validationSet eval. Strict
* improvement replaces the set; an exact float tie (no epsilon) adds to it.
*/
const updateParetoFront = (front, frontPrograms, candidateIdx, subscores) => {
	for (const [validationId, score] of subscores) {
		const prev = front.get(validationId);
		if (prev === void 0 || score > prev) {
			front.set(validationId, score);
			frontPrograms.set(validationId, /* @__PURE__ */ new Set([candidateIdx]));
		} else if (score === prev) frontPrograms.get(validationId)?.add(candidateIdx);
	}
};
/**
* Register an accepted candidate: snapshot the budget as its discovery count
* BEFORE billing its full eval, bill it, update the frontier, and inherit
* `max(parent cursors)` as its round-robin cursor.
*/
const addCandidate = (state, candidate, parents, subscores, validationSetSize) => {
	const idx = state.programCandidates.length;
	state.programCandidates.push(candidate);
	state.parentProgramForCandidate.push(parents);
	state.candidateValidationSubscores.push(subscores);
	state.metricCallCountsByDiscovery.push(state.totalEvalsCount);
	state.stepIdToUpdateNextForCandidate.push(Math.max(0, ...parents.filter((p) => p !== null).map((p) => at(state.stepIdToUpdateNextForCandidate, p, "step cursors"))));
	state.totalEvalsCount += validationSetSize;
	state.validationSetEvalsCount += 1;
	updateParetoFront(state.paretoFrontValidationSet, state.programAtParetoFrontValidationSet, idx, subscores);
	return idx;
};
const sum = (values) => values.reduce((acc, v) => acc + v, 0);
/** Mean of a candidate's per-instance scores; -Infinity if unevaluated. */
const aggregateScore = (subscores) => {
	if (subscores.size === 0) return Number.NEGATIVE_INFINITY;
	return sum([...subscores.values()]) / subscores.size;
};
/** argmax with lowest-index-wins ties. */
const argmax = (values) => {
	let best = 0;
	for (let i = 1; i < values.length; i += 1) if (at(values, i, "argmax values") > at(values, best, "argmax values")) best = i;
	return best;
};
/**
* Dominance filter: a candidate survives iff it is the sole occupant of at
* least one instance's front. Scan candidates ascending by aggregate score,
* mark one dominated when every front containing it also holds another
* non-dominated candidate, and restart after each removal — so ties resolve
* toward the higher aggregate scorer.
*/
const removeDominatedPrograms = (frontPrograms, aggScores) => {
	const members = /* @__PURE__ */ new Set();
	for (const set of frontPrograms.values()) for (const idx of set) members.add(idx);
	const ascending = [...members].toSorted((a, b) => (aggScores[a] ?? Number.NEGATIVE_INFINITY) - (aggScores[b] ?? Number.NEGATIVE_INFINITY));
	const dominated = /* @__PURE__ */ new Set();
	let changed = true;
	while (changed) {
		changed = false;
		for (const candidate of ascending) {
			if (dominated.has(candidate)) continue;
			if ([...frontPrograms.values()].filter((set) => set.has(candidate)).every((set) => [...set].some((other) => other !== candidate && !dominated.has(other)))) {
				dominated.add(candidate);
				changed = true;
				break;
			}
		}
	}
	return [...members].filter((idx) => !dominated.has(idx)).toSorted((a, b) => a - b);
};
/**
* Pareto parent sampling: dominance-filter the frontier, then pick a survivor
* with probability proportional to how many instance fronts it occupies.
*/
const selectParetoParent = (frontPrograms, aggScores, rng) => {
	const survivors = removeDominatedPrograms(frontPrograms, aggScores);
	const frequencies = survivors.map((idx) => [...frontPrograms.values()].filter((set) => set.has(idx)).length);
	return weightedChoice(rng, survivors, frequencies);
};
const initEpochSamplerState = () => ({
	epoch: -1,
	shuffled: []
});
/**
* Epoch-shuffled minibatch draw: shuffle the train ids once per epoch, pad
* the tail to a multiple of bsize with the least-frequently-used id, and serve
* sequential windows keyed by iteration number. Mutates `samplerState` in
* place; the state plus the RNG's checkpoint fully determine the stream.
*/
const sampleEpochShuffled = (rng, trainSize, bsize, samplerState, iteration) => {
	const reshuffle = () => {
		const shuffled = Array.from({ length: trainSize }, (_, i) => i);
		shuffle(rng, shuffled);
		const counts = /* @__PURE__ */ new Map();
		for (const id of shuffled) counts.set(id, (counts.get(id) ?? 0) + 1);
		while (shuffled.length % bsize !== 0) {
			let pick = -1;
			let pickCount = Number.POSITIVE_INFINITY;
			for (const [id, count] of counts) if (count <= pickCount) {
				pick = id;
				pickCount = count;
			}
			shuffled.push(pick);
			counts.set(pick, pickCount + 1);
		}
		samplerState.shuffled = shuffled;
	};
	const base = iteration * bsize;
	const currEpoch = samplerState.epoch === -1 ? 0 : Math.floor(base / Math.max(samplerState.shuffled.length, 1));
	if (samplerState.shuffled.length === 0 || currEpoch > samplerState.epoch) {
		samplerState.epoch = currEpoch;
		reshuffle();
	}
	const start = base % samplerState.shuffled.length;
	return samplerState.shuffled.slice(start, start + bsize);
};
/** Transitive closure over parent lineage (excludes the candidate itself). */
const findAncestors = (parents, idx) => {
	const ancestors = /* @__PURE__ */ new Set();
	const stack = (parents[idx] ?? []).filter((p) => p !== null);
	while (stack.length > 0) {
		const current = pop(stack, "ancestor stack");
		if (ancestors.has(current)) continue;
		ancestors.add(current);
		stack.push(...(parents[current] ?? []).filter((p) => p !== null));
	}
	return ancestors;
};
/**
* Per-component merge rule: agreeing descendants win outright; a lone
* divergence from the ancestor wins; a double divergence goes to the
* higher-aggregate descendant, coin-flipped on an exact tie.
*/
const buildMergedCandidate = (ancestor, descendantI, descendantJ, aggI, aggJ, rng) => {
	const merged = { ...ancestor };
	for (const component of Object.keys(ancestor)) {
		const anc = prop(ancestor, component, "ancestor components");
		const di = descendantI[component] ?? anc;
		const dj = descendantJ[component] ?? anc;
		if (di === dj) merged[component] = di;
		else if (anc === di) merged[component] = dj;
		else if (anc === dj) merged[component] = di;
		else if (aggI > aggJ || aggI === aggJ && rng() < .5) merged[component] = di;
		else merged[component] = dj;
	}
	return merged;
};
const tripletKey = (ancestor, i, j) => `${ancestor}|${i}|${j}`;
const MERGE_MAX_ATTEMPTS = 10;
const candidateKey = (candidate) => JSON.stringify(Object.entries(candidate).toSorted(([a], [b]) => a.localeCompare(b)));
/**
* One pair-sampling attempt loop (Python's find_common_ancestor_pair): sample
* two distinct non-ancestral survivors, then pick ONE aggregate-score-weighted
* common ancestor among those that (a) don't outscore either descendant,
* (b) have a component exactly one descendant changed, (c) aren't a tried
* triplet. No eligible ancestor → resample the pair.
*/
const findCommonAncestorPair = (state, aggScores, survivors, rng, memory) => {
	for (let attempt = 0; attempt < MERGE_MAX_ATTEMPTS; attempt += 1) {
		if (survivors.length < 2) return null;
		const first = at(survivors, Math.floor(rng() * survivors.length), "merge survivors");
		const rest = survivors.filter((idx) => idx !== first);
		const second = at(rest, Math.floor(rng() * rest.length), "merge survivors");
		const i = Math.min(first, second);
		const j = Math.max(first, second);
		const ancestorsI = findAncestors(state.parentProgramForCandidate, i);
		const ancestorsJ = findAncestors(state.parentProgramForCandidate, j);
		if (ancestorsI.has(j) || ancestorsJ.has(i)) continue;
		const eligible = [...ancestorsI].filter((idx) => ancestorsJ.has(idx)).filter((anc) => {
			if (memory.triedTriplets.has(tripletKey(anc, i, j))) return false;
			const ancScore = aggScores[anc] ?? Number.NEGATIVE_INFINITY;
			if (ancScore > (aggScores[i] ?? Number.NEGATIVE_INFINITY) || ancScore > (aggScores[j] ?? Number.NEGATIVE_INFINITY)) return false;
			const ancestor = at(state.programCandidates, anc, "candidates");
			const ci = at(state.programCandidates, i, "candidates");
			const cj = at(state.programCandidates, j, "candidates");
			return Object.keys(ancestor).some((component) => {
				const anc_ = ancestor[component];
				const di = ci[component];
				const dj = cj[component];
				return di !== dj && (anc_ === di || anc_ === dj);
			});
		});
		if (eligible.length === 0) continue;
		return [
			i,
			j,
			weightedChoiceStrict(rng, eligible, eligible.map((anc) => aggScores[anc] ?? 0))
		];
	}
	return null;
};
/**
* Ancestor-triplet merge search (Python's
* sample_and_attempt_merge_programs_by_common_predictors): every failure —
* no triplet, duplicate merged candidate, insufficient val overlap —
* resamples a fresh pair rather than giving up. The tried-triplet memo is
* recorded by the CALLER for the returned proposal only.
*/
const proposeMerge = (state, aggScores, rng, memory, valOverlapFloor = 5) => {
	const survivors = removeDominatedPrograms(state.programAtParetoFrontValidationSet, aggScores);
	if (survivors.length < 2 || state.parentProgramForCandidate.length < 3) return null;
	for (let attempt = 0; attempt < MERGE_MAX_ATTEMPTS; attempt += 1) {
		const triplet = findCommonAncestorPair(state, aggScores, survivors, rng, memory);
		if (!triplet) continue;
		const [i, j, ancestor] = triplet;
		if (memory.triedTriplets.has(tripletKey(ancestor, i, j))) continue;
		const merged = buildMergedCandidate(at(state.programCandidates, ancestor, "candidates"), at(state.programCandidates, i, "candidates"), at(state.programCandidates, j, "candidates"), aggScores[i] ?? Number.NEGATIVE_INFINITY, aggScores[j] ?? Number.NEGATIVE_INFINITY, rng);
		const pairKey = `${i}|${j}|${candidateKey(merged)}`;
		if (memory.producedByPair.has(pairKey)) continue;
		const subscoresI = at(state.candidateValidationSubscores, i, "validation subscores");
		const subscoresJ = at(state.candidateValidationSubscores, j, "validation subscores");
		if ([...subscoresI.keys()].filter((validationId) => subscoresJ.has(validationId)).length < valOverlapFloor) continue;
		memory.producedByPair.add(pairKey);
		return {
			ancestor,
			candidate: merged,
			parentI: i,
			parentJ: j
		};
	}
	return null;
};
const MERGE_SUBSAMPLE_SIZE = 5;
const MERGE_PER_BUCKET = Math.max(1, Math.ceil(MERGE_SUBSAMPLE_SIZE / 3));
const sampleUpTo = (rng, items, count) => {
	const pool = [...items];
	shuffle(rng, pool);
	return pool.slice(0, count);
};
/**
* Balanced 5-id subsample over the parents' shared val ids: up to 2 each from
* the i-better / j-better / tied buckets, topped up from unused ids (with
* replacement only as a last resort), truncated to 5.
*/
const buildMergeSubsample = (sharedIds, subscoresI, subscoresJ, rng) => {
	const iBetter = [];
	const jBetter = [];
	const tied = [];
	for (const validationId of sharedIds) {
		const si = get(subscoresI, validationId, "subscores I");
		const sj = get(subscoresJ, validationId, "subscores J");
		if (si > sj) iBetter.push(validationId);
		else if (sj > si) jBetter.push(validationId);
		else tied.push(validationId);
	}
	const chosen = [];
	for (const bucket of [
		iBetter,
		jBetter,
		tied
	]) {
		if (chosen.length >= MERGE_SUBSAMPLE_SIZE) break;
		const available = bucket.filter((validationId) => !chosen.includes(validationId));
		const take = Math.min(available.length, MERGE_PER_BUCKET, MERGE_SUBSAMPLE_SIZE - chosen.length);
		if (take > 0) chosen.push(...sampleUpTo(rng, available, take));
	}
	const remaining = MERGE_SUBSAMPLE_SIZE - chosen.length;
	if (remaining > 0) {
		const unused = sharedIds.filter((validationId) => !chosen.includes(validationId));
		if (unused.length >= remaining) chosen.push(...sampleUpTo(rng, unused, remaining));
		else if (sharedIds.length > 0) for (let k = 0; k < remaining; k += 1) chosen.push(at(sharedIds, Math.floor(rng() * sharedIds.length), "shared ids"));
	}
	return chosen.slice(0, MERGE_SUBSAMPLE_SIZE);
};
/**
* One merge iteration: propose an ancestor-triplet merge, eval it on the
* balanced 5-id subsample (billed even when rejected), and accept iff the
* merged sum is `>=` the better parent's sum on the same ids — non-strict,
* unlike reflection's strict `>`.
*/
const runMergeIteration = async (adapter, state, options, memory) => {
	const { rng, validationSet } = options;
	const aggScores = state.candidateValidationSubscores.map(aggregateScore);
	const proposal = proposeMerge(state, aggScores, rng, memory);
	if (!proposal) return "none";
	memory.triedTriplets.add(tripletKey(proposal.ancestor, proposal.parentI, proposal.parentJ));
	const subscoresI = at(state.candidateValidationSubscores, proposal.parentI, "validation subscores");
	const subscoresJ = at(state.candidateValidationSubscores, proposal.parentJ, "validation subscores");
	const sharedIds = [...subscoresI.keys()].filter((validationId) => subscoresJ.has(validationId));
	const subsample = buildMergeSubsample(sharedIds, subscoresI, subscoresJ, rng);
	const batch = subsample.map((validationId) => at(validationSet, validationId, "validationSet"));
	const mergedEval = await adapter.evaluate(batch, proposal.candidate, false);
	state.totalEvalsCount += batch.length;
	const sumMerged = sum(mergedEval.scores);
	const sumI = sum(subsample.map((validationId) => get(subscoresI, validationId, "subscores I")));
	const sumJ = sum(subsample.map((validationId) => get(subscoresJ, validationId, "subscores J")));
	if (sumMerged < Math.max(sumI, sumJ)) return "rejected";
	const fullEval = await adapter.evaluate(validationSet, proposal.candidate, false);
	const subscores = toSubscores(fullEval.scores);
	addCandidate(state, proposal.candidate, [proposal.parentI, proposal.parentJ], subscores, validationSet.length);
	await options.onAccepted?.(proposal.candidate, subscores);
	console.log(`GEPA iteration ${state.i}: accepted merge of ${proposal.parentI} and ${proposal.parentJ}`);
	return "accepted";
};
const selectParent = (state, options) => {
	const aggScores = state.candidateValidationSubscores.map(aggregateScore);
	if (options.candidateSelectionStrategy === "currentBest") return argmax(aggScores);
	return selectParetoParent(state.programAtParetoFrontValidationSet, aggScores, options.rng);
};
/** GEPAState after the seed candidate's full validationSet eval. */
const initGEPAState = (seedCandidate, seedScores) => {
	const state = {
		candidateValidationSubscores: [toSubscores(seedScores)],
		i: -1,
		metricCallCountsByDiscovery: [0],
		parentProgramForCandidate: [[null]],
		paretoFrontValidationSet: /* @__PURE__ */ new Map(),
		programAtParetoFrontValidationSet: /* @__PURE__ */ new Map(),
		programCandidates: [seedCandidate],
		stepIdToUpdateNextForCandidate: [0],
		totalEvalsCount: seedScores.length,
		validationSetEvalsCount: 1
	};
	updateParetoFront(state.paretoFrontValidationSet, state.programAtParetoFrontValidationSet, 0, at(state.candidateValidationSubscores, 0, "validation subscores"));
	return state;
};
const initGEPALoopState = (state) => ({
	bestAgg: aggregateScore(at(state.candidateValidationSubscores, 0, "validation subscores")),
	lastIterFoundNewProgram: false,
	mergeMemory: {
		producedByPair: /* @__PURE__ */ new Set(),
		triedTriplets: /* @__PURE__ */ new Set()
	},
	mergesDue: 0,
	samplerState: initEpochSamplerState(),
	totalMergesTested: 0
});
const noteImprovement = async (loop, onImprovement, candidate, subscores) => {
	const agg = aggregateScore(subscores);
	if (agg > loop.bestAgg) {
		loop.bestAgg = agg;
		await onImprovement?.(candidate);
	}
};
/**
* The reflection prologue: pick a parent, evaluate it on the epoch-shuffled
* minibatch with traces, advance the component cursor, and build the
* reflective dataset. Returns null when the iteration produces nothing to
* reflect on (no trajectories, all-perfect scores, or no component records) —
* budget is still billed, exactly like upstream.
*/
const prepareReflection = async (adapter, state, loop, options) => {
	const componentNames = Object.keys(options.seedCandidate);
	const parentIdx = selectParent(state, options);
	const parent = at(state.programCandidates, parentIdx, "candidates");
	const minibatchIds = sampleEpochShuffled(options.rng, options.trainingSet.length, options.reflectionMinibatchSize, loop.samplerState, state.i);
	const batch = minibatchIds.map((id) => at(options.trainingSet, id, "trainingSet"));
	const parentEval = await adapter.evaluate(batch, parent, true);
	state.totalEvalsCount += batch.length;
	if (!parentEval.trajectories || parentEval.trajectories.length === 0) return null;
	if (options.skipPerfectScore && parentEval.scores.every((score) => score >= options.perfectScore)) return null;
	let components;
	if (options.componentSelector === "all") components = componentNames;
	else {
		const cursor = at(state.stepIdToUpdateNextForCandidate, parentIdx, "step cursors");
		components = [at(componentNames, cursor % componentNames.length, "component names")];
		state.stepIdToUpdateNextForCandidate[parentIdx] = (cursor + 1) % componentNames.length;
	}
	const reflectiveDataset = await adapter.makeReflectiveDataset(parent, parentEval, components);
	const componentsWithData = components.filter((name) => (reflectiveDataset[name] ?? []).length > 0);
	if (componentsWithData.length === 0) return null;
	return {
		components: componentsWithData,
		minibatchIds,
		parentIdx,
		parentScores: parentEval.scores,
		reflectiveDataset
	};
};
/**
* The reflection epilogue: apply the proposed texts to the parent, evaluate
* the child on the same minibatch, and accept iff its score SUM strictly
* beats the parent's (upstream's rule) — acceptance bills a full
* validationSet eval, registers the candidate, and schedules a merge.
*/
const acceptReflection = async (adapter, state, loop, options, plan, newTexts) => {
	const parent = at(state.programCandidates, plan.parentIdx, "candidates");
	const applicable = Object.entries(newTexts).filter(([name]) => name in parent);
	if (applicable.length === 0) return false;
	const child = { ...parent };
	for (const [name, text] of applicable) child[name] = text;
	const batch = plan.minibatchIds.map((id) => at(options.trainingSet, id, "trainingSet"));
	const childEval = await adapter.evaluate(batch, child, true);
	state.totalEvalsCount += batch.length;
	if (sum(childEval.scores) <= sum(plan.parentScores)) return false;
	const fullEval = await adapter.evaluate(options.validationSet, child, false);
	const subscores = toSubscores(fullEval.scores);
	addCandidate(state, child, [plan.parentIdx], subscores, options.validationSet.length);
	await noteImprovement(loop, options.onImprovement, child, subscores);
	loop.lastIterFoundNewProgram = true;
	if (loop.totalMergesTested < options.maxMergeInvocations) loop.mergesDue += 1;
	console.log(`GEPA iteration ${state.i}: accepted child of ${plan.parentIdx} (component ${plan.components.join(", ")})`);
	return true;
};
/**
* The merge branch of one iteration, when it is due: returns the outcome and
* does the loop bookkeeping. "none" means the search was fruitless and the
* iteration should fall through to reflection.
*/
const runMergeBranch = async (adapter, state, loop, options) => {
	loop.lastIterFoundNewProgram = false;
	const outcome = await runMergeIteration(adapter, state, {
		onAccepted: (candidate, subscores) => noteImprovement(loop, options.onImprovement, candidate, subscores),
		rng: options.rng,
		validationSet: options.validationSet
	}, loop.mergeMemory);
	if (outcome === "accepted") {
		loop.mergesDue -= 1;
		loop.totalMergesTested += 1;
	}
	return outcome;
};
/** Whether the next iteration should try a merge before reflecting. */
const mergeDue = (loop, options) => options.useMerge && loop.mergesDue > 0 && loop.lastIterFoundNewProgram;
const serializeGEPAState = (state) => ({
	candidateValidationSubscores: state.candidateValidationSubscores.map((subscores) => [...subscores.entries()]),
	i: state.i,
	metricCallCountsByDiscovery: [...state.metricCallCountsByDiscovery],
	parentProgramForCandidate: state.parentProgramForCandidate.map((parents) => [...parents]),
	paretoFrontValidationSet: [...state.paretoFrontValidationSet.entries()],
	programAtParetoFrontValidationSet: [...state.programAtParetoFrontValidationSet.entries()].map(([validationId, programs]) => [validationId, [...programs]]),
	programCandidates: structuredClone(state.programCandidates),
	stepIdToUpdateNextForCandidate: [...state.stepIdToUpdateNextForCandidate],
	totalEvalsCount: state.totalEvalsCount,
	validationSetEvalsCount: state.validationSetEvalsCount
});
const deserializeGEPAState = (serialized) => ({
	candidateValidationSubscores: serialized.candidateValidationSubscores.map((subscores) => new Map(subscores)),
	i: serialized.i,
	metricCallCountsByDiscovery: [...serialized.metricCallCountsByDiscovery],
	parentProgramForCandidate: serialized.parentProgramForCandidate.map((parents) => [...parents]),
	paretoFrontValidationSet: new Map(serialized.paretoFrontValidationSet),
	programAtParetoFrontValidationSet: new Map(serialized.programAtParetoFrontValidationSet.map(([validationId, programs]) => [validationId, new Set(programs)])),
	programCandidates: structuredClone(serialized.programCandidates),
	stepIdToUpdateNextForCandidate: [...serialized.stepIdToUpdateNextForCandidate],
	totalEvalsCount: serialized.totalEvalsCount,
	validationSetEvalsCount: serialized.validationSetEvalsCount
});
const serializeGEPALoopState = (loop) => ({
	bestAgg: loop.bestAgg,
	lastIterFoundNewProgram: loop.lastIterFoundNewProgram,
	mergeMemory: {
		producedByPair: [...loop.mergeMemory.producedByPair],
		triedTriplets: [...loop.mergeMemory.triedTriplets]
	},
	mergesDue: loop.mergesDue,
	samplerState: structuredClone(loop.samplerState),
	totalMergesTested: loop.totalMergesTested
});
const deserializeGEPALoopState = (serialized) => ({
	bestAgg: serialized.bestAgg,
	lastIterFoundNewProgram: serialized.lastIterFoundNewProgram,
	mergeMemory: {
		producedByPair: new Set(serialized.mergeMemory.producedByPair),
		triedTriplets: new Set(serialized.mergeMemory.triedTriplets)
	},
	mergesDue: serialized.mergesDue,
	samplerState: structuredClone(serialized.samplerState),
	totalMergesTested: serialized.totalMergesTested
});
//#endregion
//#region src/optimizers/gepa/index.ts
const AUTO_CANDIDATES = {
	heavy: 18,
	light: 6,
	medium: 12
};
/**
* DSPy's auto-budget estimate. `fullEvalSteps` (m) exists only here — the
* engine has no periodic full-eval scheduling.
*/
const autoBudget = (stepsCount, candidates, validationSetSize, minibatchSize = 35, fullEvalSteps = 5) => {
	const trialsCount = Math.floor(Math.max(2 * (stepsCount * 2) * Math.log2(candidates), 1.5 * candidates));
	if (trialsCount < 0 || validationSetSize < 0 || minibatchSize < 0) throw new Error("autoBudget arguments must be non-negative");
	if (fullEvalSteps < 1) throw new Error("fullEvalSteps must be >= 1");
	let total = validationSetSize + candidates * 5 + trialsCount * minibatchSize;
	if (trialsCount === 0) return total;
	total += (Math.floor((trialsCount + 1) / fullEvalSteps) + 1 + (trialsCount < fullEvalSteps ? 1 : 0)) * validationSetSize;
	return total;
};
const VALIDATION_SET_SIZE_NOTE = 35;
const resolveBudget = (config, stepsCount, trainingSet, validationSetSize) => {
	if ([
		config.auto,
		config.maxFullEvals,
		config.maxMetricCalls
	].filter((value) => value !== void 0).length !== 1) throw new Error("Exactly one of auto, maxFullEvals, maxMetricCalls must be set");
	if (config.maxMetricCalls !== void 0) return config.maxMetricCalls;
	if (config.maxFullEvals !== void 0) return config.maxFullEvals * (trainingSet.length + (config.validationSet?.length ?? 0));
	if (config.auto === void 0) throw new Error("Exactly one of auto, maxFullEvals, maxMetricCalls must be set");
	return autoBudget(Math.max(stepsCount, 1), AUTO_CANDIDATES[config.auto], validationSetSize);
};
const buildResult = (state, validationSetSize, seed, buildProgram) => {
	const validationAggregateScores = state.candidateValidationSubscores.map(aggregateScore);
	const bestIdx = argmax(validationAggregateScores);
	return {
		bestIdx,
		candidates: state.programCandidates,
		discoveryEvalCounts: state.metricCallCountsByDiscovery,
		parents: state.parentProgramForCandidate,
		perValidationInstanceBestCandidates: state.programAtParetoFrontValidationSet,
		program: buildProgram(at(state.programCandidates, bestIdx, "candidates")),
		seed,
		totalMetricCalls: state.totalEvalsCount,
		validationAggregateScores,
		validationSetEvalsCount: state.validationSetEvalsCount,
		validationSubscores: Array.from({ length: state.programCandidates.length }, (_, idx) => Array.from({ length: validationSetSize }, (_2, validationId) => at(state.candidateValidationSubscores, idx, "candidate validation subscores").get(validationId) ?? NaN))
	};
};
/** The serialized engine state, as it crosses workflow step boundaries. */
const serializedStateSchema = z.object({
	candidateValidationSubscores: z.array(z.array(z.tuple([z.number(), z.number()]))),
	i: z.number(),
	metricCallCountsByDiscovery: z.array(z.number()),
	parentProgramForCandidate: z.array(z.array(z.number().nullable())),
	paretoFrontValidationSet: z.array(z.tuple([z.number(), z.number()])),
	programAtParetoFrontValidationSet: z.array(z.tuple([z.number(), z.array(z.number())])),
	programCandidates: z.array(z.record(z.string(), z.string())),
	stepIdToUpdateNextForCandidate: z.array(z.number()),
	totalEvalsCount: z.number(),
	validationSetEvalsCount: z.number()
});
const serializedLoopSchema = z.object({
	bestAgg: z.number(),
	lastIterFoundNewProgram: z.boolean(),
	mergeMemory: z.object({
		producedByPair: z.array(z.string()),
		triedTriplets: z.array(z.string())
	}),
	mergesDue: z.number(),
	samplerState: z.object({
		epoch: z.number(),
		shuffled: z.array(z.number())
	}),
	totalMergesTested: z.number()
});
const reflectiveExampleSchema = z.object({
	Feedback: z.string(),
	"Generated Outputs": z.union([z.record(z.string(), z.string()), z.string()]),
	Inputs: z.record(z.string(), z.string())
});
const reflectionPlanSchema = z.object({
	components: z.array(z.string()),
	minibatchIds: z.array(z.number()),
	parentIdx: z.number(),
	parentScores: z.array(z.number()),
	reflectiveDataset: z.record(z.string(), z.array(reflectiveExampleSchema))
});
const iterationSchema = z.object({
	loop: serializedLoopSchema,
	rng: z.object({
		adapter: z.number(),
		engine: z.number()
	}),
	state: serializedStateSchema,
	studentPrompts: promptsSchema
});
const reflectedSchema = iterationSchema.extend({ plan: reflectionPlanSchema.nullable() });
const proposedSchema$1 = reflectedSchema.extend({ newTexts: z.record(z.string(), z.string()).nullable() });
/** The description-only Candidate view of a student snapshot. */
const seedCandidateOf = (studentPrompts) => Object.fromEntries(Object.entries(studentPrompts.steps).map(([id, step]) => [id, step.description]));
/**
* Genetic-Pareto reflective prompt evolution as a Mastra workflow over the
* target `workflow`: a pre-pass step optionally bootstraps few-shot examples
* (its metric calls are not billed to GEPA's budget, matching DSPy), a
* seed-eval step scores the seed candidate over the validationSet, and a
* durable dountil loop runs one GEPA iteration per pass — split into a
* `reflect` step (parent selection, minibatch rollouts, reflective dataset,
* or the merge branch), a `propose` step that makes the reflection-LM calls,
* and an `accept` step (child evaluation and Pareto bookkeeping). Every
* candidate crosses step boundaries as a JSON snapshot and randomness as
* checkpointed RNG state, so a storage-backed run resumes mid-optimization
* without redoing completed iterations, and savePrompts checkpoints the best
* candidate whenever the aggregate score improves.
*/
const createGEPAWorkflow = (workflow, config) => {
	const { checkpoint, maxFewShotExamples = 0, maxLabeledExamples, maxScorerCalls, reflectionModel, savePrompts, scorer, trainingSet, ...tuning } = config;
	const examples = [...trainingSet];
	if (examples.length === 0) throw new Error("GEPA requires a non-empty trainingSet");
	if ([
		tuning.auto,
		tuning.maxFullEvals,
		maxScorerCalls
	].filter((value) => value !== void 0).length !== 1) throw new Error("Exactly one of auto, maxFullEvals, maxScorerCalls must be set");
	const validationSet = tuning.validationSet ?? examples;
	if (!tuning.validationSet) console.warn("GEPA: no validationSet provided; using the trainingSet for validation.");
	if (validationSet.length > VALIDATION_SET_SIZE_NOTE) console.warn(`GEPA: validationSet has ${validationSet.length} examples; every accepted candidate costs a full validationSet eval.`);
	const stepsCount = workflowToProgram(workflow).steps.length;
	const maxMetricCalls = resolveBudget({
		...tuning,
		maxMetricCalls: maxScorerCalls
	}, stepsCount, examples, validationSet.length);
	const seed = tuning.seed ?? 0;
	const metric = scorerMetric(resolveScorer(workflow, scorer));
	const cache = /* @__PURE__ */ new WeakMap();
	const cachedMetric = (gold, prediction, target) => {
		if (prediction === null) return metric(gold, void 0, target);
		const hit = cache.get(prediction);
		if (hit) return hit;
		const pending = (async () => {
			try {
				return await metric(gold, prediction, target);
			} catch (error) {
				cache.delete(prediction);
				throw error;
			}
		})();
		cache.set(prediction, pending);
		return pending;
	};
	const gepaMetric = (gold, prediction, _trace, _stepId, _stepTrace, target) => cachedMetric(gold, prediction, target);
	const buildAdapter = (studentPrompts, adapterRNGState) => {
		const student = programFromPrompts(workflowToProgram(workflow), studentPrompts);
		const adapterRNG = restoreRNG(adapterRNGState);
		return {
			adapter: createProgramAdapter({
				adapterRNG,
				addFormatFailureAsFeedback: tuning.addFormatFailureAsFeedback ?? false,
				failureScore: tuning.failureScore ?? 0,
				metric: gepaMetric,
				program: student,
				reflectionModel: reflectionModel ?? first(student.steps, "workflow steps").model,
				warnOnScoreMismatch: tuning.warnOnScoreMismatch ?? true
			}),
			adapterRNG,
			student
		};
	};
	const engineOptionsFor = (rng, adapter, seedCandidate) => ({
		candidateSelectionStrategy: tuning.candidateSelectionStrategy ?? "pareto",
		componentSelector: tuning.componentSelector ?? "roundRobin",
		maxMergeInvocations: tuning.maxMergeInvocations ?? 5,
		maxMetricCalls,
		onImprovement: async (candidate) => {
			await savePrompts(promptsOf(adapter.buildProgram(candidate)));
		},
		perfectScore: tuning.perfectScore ?? 1,
		reflectionMinibatchSize: tuning.reflectionMinibatchSize ?? 3,
		rng,
		seedCandidate,
		skipPerfectScore: tuning.skipPerfectScore ?? true,
		trainingSet: examples,
		useMerge: tuning.useMerge ?? true,
		validationSet
	});
	const prepass = createStep({
		description: "Optional BootstrapFewShot pre-pass installing few-shot examples",
		execute: async () => {
			let student = workflowToProgram(workflow);
			if (maxFewShotExamples > 0) student = await bootstrapFewShotProgram(student, examples, {
				maxFewShotExamples,
				maxLabeledExamples: maxLabeledExamples ?? maxFewShotExamples,
				metric: (gold, prediction) => cachedMetric(gold, prediction)
			});
			return { studentPrompts: promptsOf(student) };
		},
		id: "prepass",
		inputSchema: z.object({}),
		outputSchema: z.object({ studentPrompts: promptsSchema })
	});
	const seedEval = createStep({
		description: "Score the seed candidate over the validationSet",
		execute: async ({ inputData }) => {
			const { studentPrompts } = inputData;
			const { adapter } = buildAdapter(studentPrompts, seed);
			const seedCandidate = seedCandidateOf(studentPrompts);
			const evaluated = await adapter.evaluate(validationSet, seedCandidate, false);
			const state = initGEPAState(seedCandidate, evaluated.scores);
			const loop = initGEPALoopState(state);
			return {
				loop: serializeGEPALoopState(loop),
				rng: {
					adapter: createRNG(seed).state,
					engine: createRNG(seed).state
				},
				state: serializeGEPAState(state),
				studentPrompts
			};
		},
		id: "seed-eval",
		inputSchema: z.object({ studentPrompts: promptsSchema }),
		outputSchema: iterationSchema
	});
	const reflect = createStep({
		description: "One iteration's prologue: merge branch, or minibatch rollouts and the reflective dataset",
		execute: async ({ inputData, resumeData, suspend }) => {
			const payload = inputData;
			const state = deserializeGEPAState(payload.state);
			if (state.totalEvalsCount >= maxMetricCalls) return {
				...payload,
				plan: null
			};
			if (!resumeData && await checkpoint?.({ iteration: state.i + 1 })) return await suspend({ iteration: state.i + 1 });
			const loop = deserializeGEPALoopState(payload.loop);
			const engineRNG = restoreRNG(payload.rng.engine);
			const { adapter, adapterRNG } = buildAdapter(payload.studentPrompts, payload.rng.adapter);
			const options = engineOptionsFor(engineRNG, adapter, seedCandidateOf(payload.studentPrompts));
			state.i += 1;
			let plan = null;
			if (mergeDue(loop, options)) {
				if (await runMergeBranch(adapter, state, loop, options) === "none") plan = await prepareReflection(adapter, state, loop, options);
			} else plan = await prepareReflection(adapter, state, loop, options);
			return {
				loop: serializeGEPALoopState(loop),
				plan,
				rng: {
					adapter: adapterRNG.state,
					engine: engineRNG.state
				},
				state: serializeGEPAState(state),
				studentPrompts: payload.studentPrompts
			};
		},
		id: "reflect",
		inputSchema: iterationSchema,
		outputSchema: reflectedSchema,
		resumeSchema: z.object({}),
		suspendSchema: z.object({ iteration: z.number() })
	});
	const propose = createStep({
		description: "Reflection-LM calls proposing new instruction texts",
		execute: async ({ inputData }) => {
			const { plan } = inputData;
			if (!plan) return {
				...inputData,
				newTexts: null
			};
			const { adapter } = buildAdapter(inputData.studentPrompts, inputData.rng.adapter);
			const parent = at(inputData.state.programCandidates, plan.parentIdx, "candidates");
			const newTexts = await adapter.proposeNewTexts(parent, plan.reflectiveDataset, plan.components);
			return {
				...inputData,
				newTexts
			};
		},
		id: "propose",
		inputSchema: reflectedSchema,
		outputSchema: proposedSchema$1
	});
	const accept = createStep({
		description: "Child evaluation, acceptance, Pareto bookkeeping, checkpointing",
		execute: async ({ inputData }) => {
			const { newTexts, plan, ...payload } = inputData;
			if (!(plan && newTexts)) return payload;
			const state = deserializeGEPAState(payload.state);
			const loop = deserializeGEPALoopState(payload.loop);
			const engineRNG = restoreRNG(payload.rng.engine);
			const { adapter, adapterRNG } = buildAdapter(payload.studentPrompts, payload.rng.adapter);
			const options = engineOptionsFor(engineRNG, adapter, seedCandidateOf(payload.studentPrompts));
			await acceptReflection(adapter, state, loop, options, plan, newTexts);
			return {
				loop: serializeGEPALoopState(loop),
				rng: {
					adapter: adapterRNG.state,
					engine: engineRNG.state
				},
				state: serializeGEPAState(state),
				studentPrompts: payload.studentPrompts
			};
		},
		id: "accept",
		inputSchema: proposedSchema$1,
		outputSchema: iterationSchema
	});
	const iteration = createWorkflow({
		id: "iteration",
		inputSchema: iterationSchema,
		outputSchema: iterationSchema
	}).then(reflect).then(propose).then(accept).commit();
	const finalize = createStep({
		description: "Select the winner, persist it, and land it on the workflow",
		execute: async ({ inputData }) => {
			const payload = inputData;
			const state = deserializeGEPAState(payload.state);
			const { adapter, student } = buildAdapter(payload.studentPrompts, payload.rng.adapter);
			const result = buildResult(state, validationSet.length, seed, adapter.buildProgram);
			await savePrompts(promptsOf(result.program));
			applyProgram(workflow, result.program);
			return {
				candidates: result.candidates.map((candidate, idx) => {
					const snapshot = student.clone();
					for (const step of snapshot.steps) {
						const description = candidate[step.id];
						if (description !== void 0) step.description = description;
					}
					return [promptsOf(snapshot), { score: at(result.validationAggregateScores, idx, "aggregate scores") }];
				}),
				score: at(result.validationAggregateScores, result.bestIdx, "aggregate scores")
			};
		},
		id: "finalize",
		inputSchema: iterationSchema,
		outputSchema: optimizerResultSchema
	});
	return createWorkflow({
		id: `${workflow.id}.gepa`,
		inputSchema: z.object({}),
		outputSchema: optimizerResultSchema
	}).then(prepass).then(seedEval).dountil(iteration, ({ inputData }) => Promise.resolve(inputData.state.totalEvalsCount >= maxMetricCalls)).then(finalize).commit();
};
//#endregion
//#region src/optimizers/labeled-few-shot.ts
/**
* LabeledFewShot as a Mastra workflow over the target `workflow`: install up
* to `maxFewShotExamples` labeled trainingSet examples as few-shot examples
* on every step (dspy.teleprompt.vanilla.LabeledFewShot). Compiling makes no
* LM calls; the evaluate step runs the compiled workflow over the trainingSet
* once, and the apply step lands the compiled prompt state in place on the
* target workflow. All inter-step state is JSON, so a storage-backed run is
* durable and observable step by step.
*/
const createLabeledFewShotWorkflow = (workflow, config) => {
	const compile = createStep({
		description: "Install labeled examples as few-shot examples on every step",
		execute: () => {
			const compiled = labeledFewShotProgram(workflowToProgram(workflow), [...config.trainingSet], config.maxFewShotExamples ?? 16);
			return Promise.resolve({ prompts: promptsOf(compiled) });
		},
		id: "compile",
		inputSchema: z.object({}),
		outputSchema: compiledSchema
	});
	const { apply, evaluate, save } = finishingSteps(workflow, {
		metric: scorerMetric(resolveScorer(workflow, config.scorer)),
		savePrompts: config.savePrompts,
		trainingSet: config.trainingSet
	});
	return createWorkflow({
		id: `${workflow.id}.labeled-few-shot`,
		inputSchema: z.object({}),
		outputSchema: optimizerResultSchema
	}).then(compile).then(save).then(evaluate).then(apply).commit();
};
//#endregion
//#region src/optimizers/simba.ts
const mean = (values) => values.reduce((acc, v) => acc + v, 0) / values.length;
/** Python's round(): banker's rounding, used for final-selection spacing. */
const roundHalfEven = (x) => {
	const floor = Math.floor(x);
	const diff = x - floor;
	if (diff > .5) return floor + 1;
	if (diff < .5) return floor;
	return floor % 2 === 0 ? floor : floor + 1;
};
/**
* Sort program indices by average score descending (stable — ties break toward
* the lower index), take the first k, force the baseline (0) into the last
* slot if absent, then dedupe preserving order. May return fewer than k.
*/
const topKPlusBaseline = (avgScores, k) => {
	const sorted = avgScores.map((avg, idx) => ({
		avg,
		idx
	})).toSorted((a, b) => b.avg - a.avg).slice(0, k).map((entry) => entry.idx);
	if (sorted.length > 0 && !sorted.includes(0)) sorted[sorted.length - 1] = 0;
	return [...new Set(sorted)];
};
/**
* Sample an index weighted by exp(avg/temperature); uniform fallback when the
* weight sum is not positive. With all-zero scores this is uniform.
*/
const softmaxSample = (rng, programIdxs, avgScores, temperature) => {
	if (programIdxs.length === 0) throw new Error("No programs available for softmax sampling.");
	return weightedChoice(rng, programIdxs, programIdxs.map((idx) => Math.exp((avgScores[idx] ?? 0) / temperature)));
};
/** NumPy-default linear-interpolation percentile. */
const percentile = (values, p) => {
	const sorted = values.toSorted((a, b) => a - b);
	if (sorted.length === 0) return 0;
	const rank = p / 100 * (sorted.length - 1);
	const lo = Math.floor(rank);
	const hi = Math.ceil(rank);
	const low = at(sorted, lo, "sorted scores");
	return low + (at(sorted, hi, "sorted scores") - low) * (rank - lo);
};
/**
* Group model-major rollouts into per-example buckets (stride = batchSize),
* each sorted by score descending, then order buckets by
* (max−min gap, max score, max−avg gap) lexicographically descending.
* Rollout records are shallow-copied so strategies never mutate shared state.
*/
const makeBuckets = (rollouts, batchSize) => {
	const buckets = [];
	for (let exampleIdx = 0; exampleIdx < batchSize; exampleIdx += 1) {
		const bucket = [];
		for (let i = exampleIdx; i < rollouts.length; i += batchSize) bucket.push({ ...at(rollouts, i, "rollouts") });
		bucket.sort((a, b) => b.score - a.score);
		const maxScore = first(bucket, "bucket").score;
		const minScore = last(bucket, "bucket").score;
		const avgScore = mean(bucket.map((r) => r.score));
		buckets.push({
			maxScore,
			maxToAvgGap: maxScore - avgScore,
			maxToMinGap: maxScore - minScore,
			rollouts: bucket
		});
	}
	return buckets.toSorted((a, b) => b.maxToMinGap - a.maxToMinGap || b.maxScore - a.maxScore || b.maxToAvgGap - a.maxToAvgGap);
};
/**
* maxFewShotExamples enforced probabilistically: expected ~1 drop for a full
* step, at least one forced at/over the cap. Draws are with replacement, so
* the realized drop count can be lower than the sampled one. The single index
* set applies to every step of the candidate.
*/
const dropExamples = (candidate, maxFewShotExamples, rng, poissonRNG) => {
	const cap = maxFewShotExamples > 0 ? maxFewShotExamples : 3;
	const examplesCount = Math.max(0, ...candidate.steps.map((step) => step.examples.length));
	let toDrop = Math.max(samplePoisson(poissonRNG, examplesCount / cap), examplesCount >= cap ? 1 : 0);
	toDrop = Math.min(toDrop, examplesCount);
	const dropIdxs = /* @__PURE__ */ new Set();
	for (let i = 0; i < toDrop; i += 1) dropIdxs.add(Math.floor(rng() * examplesCount));
	for (const step of candidate.steps) step.examples = step.examples.filter((_, idx) => !dropIdxs.has(idx));
	return toDrop;
};
const appendAnExample = (bucket, candidate, opts) => {
	const good = first(bucket.rollouts, "bucket rollouts");
	if (good.score <= opts.p10) {
		console.log(`Skipping appending an example as good score ${good.score} is at or below the 10th percentile.`);
		return false;
	}
	const idToExample = /* @__PURE__ */ new Map();
	for (const traceStep of good.trace) {
		const inputData = { ...traceStep.inputData };
		for (const [key, value] of Object.entries(inputData)) {
			const text = String(value);
			if (opts.maxFewShotInputLength && text.length > opts.maxFewShotInputLength) inputData[key] = `${text.slice(0, opts.maxFewShotInputLength)}\n\t\t... <TRUNCATED FOR BREVITY>`;
		}
		idToExample.set(traceStep.stepId, {
			inputData,
			outputData: traceStep.outputData
		});
	}
	let added = 0;
	for (const [stepId, example] of idToExample) {
		const step = candidate.steps.find((s) => s.id === stepId);
		if (!step) continue;
		step.examples.push(example);
		added += 1;
	}
	console.log(`Added ${added} examples (one each) across all steps.`);
	return true;
};
const OFFER_FEEDBACK_INSTRUCTIONS = `You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
similar inputs in the future.

The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

In your advice:
- Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
- Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
- Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
- Ensure each unique module name appears exactly once as a key in the advice dictionary.`;
const TRAJECTORY_DESCRIPTION = "The trajectory of the program's execution, showing each module's I/O";
const OUTPUTS_DESCRIPTION = "The outputs of the program that we are analyzing";
const REWARD_VALUE_DESCRIPTION = "The reward value assigned to the program's outputs";
const REWARD_INFO_DESCRIPTION = "Additional information that might be helpful to understanding the assigned reward value.";
/** OfferFeedback input fields, in declaration order — order matters for prompt layout. */
const OFFER_FEEDBACK_INPUT_FIELDS = [
	["program_code", "The code of the program that we are analyzing"],
	["modules_defn", "The definition of each module in the program, including its I/O"],
	["program_inputs", "The inputs to the program that we are analyzing"],
	["oracle_metadata", "Any (hidden) metadata about the training set instance we're analyzing"],
	["worse_program_trajectory", TRAJECTORY_DESCRIPTION],
	["worse_program_outputs", OUTPUTS_DESCRIPTION],
	["worse_reward_value", REWARD_VALUE_DESCRIPTION],
	["worse_reward_info", REWARD_INFO_DESCRIPTION],
	["better_program_trajectory", TRAJECTORY_DESCRIPTION],
	["better_program_outputs", OUTPUTS_DESCRIPTION],
	["better_reward_value", REWARD_VALUE_DESCRIPTION],
	["better_reward_info", REWARD_INFO_DESCRIPTION],
	["module_names", "The names of the modules in the program, for which we seek advice"]
];
const MODULE_ADVICE_DESCRIPTION = "For each module, describe very concretely: If the module receives ${description of input or patterns therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. Basically, your advice be such that if the module has access to your tip, it would be much more likely to act like the successful trajectory rather than the lower-scoring trajectory.";
const offerFeedbackSchema = (moduleNames) => z.object({
	discussion: z.string().describe("Discussing blame of where each module went wrong, if it did"),
	moduleAdvice: z.object(Object.fromEntries(moduleNames.map((name) => [name, z.string()]))).describe(MODULE_ADVICE_DESCRIPTION)
});
const offerFeedback = async (promptModel, moduleNames, fields) => {
	const sections = OFFER_FEEDBACK_INPUT_FIELDS.map(([name, description]) => `[[ ## ${name} ## ]]\n${description}\n\n${serializeField(fields[name])}`);
	const { output } = await generateText({
		model: promptModel,
		output: Output.object({ schema: offerFeedbackSchema(moduleNames) }),
		prompt: [OFFER_FEEDBACK_INSTRUCTIONS, ...sections].join("\n\n")
	});
	return output;
};
const toTrajectory = (trace) => trace.map((traceStep) => ({
	inputs: traceStep.inputData,
	module_name: traceStep.stepId,
	outputs: traceStep.outputData
}));
/** Stand-in shown when a rollout carries no usable contrast. */
const BLANKED_CONTRAST = {
	prediction: { "N/A": "Prediction not available" },
	score: "N/A",
	trace: []
};
const appendARule = async (bucket, candidate, opts) => {
	const good = first(bucket.rollouts, "bucket rollouts");
	const bad = last(bucket.rollouts, "bucket rollouts");
	if (good.score <= opts.p10 || bad.score >= opts.p90) {
		console.log(`Skipping rule generation as good score ${good.score} is at or below the 10th percentile *or* bad score ${bad.score} is at or above the 90th percentile.`);
		return false;
	}
	let goodView = {
		outputMetadata: good.outputMetadata,
		prediction: good.prediction,
		score: good.score,
		trace: good.trace
	};
	let badView = {
		outputMetadata: bad.outputMetadata,
		prediction: bad.prediction,
		score: bad.score,
		trace: bad.trace
	};
	if (good.score <= bad.score) {
		if (good.score > opts.p90) badView = {
			...badView,
			...BLANKED_CONTRAST
		};
		else goodView = {
			...goodView,
			...BLANKED_CONTRAST
		};
	}
	const { example } = good;
	const result = await offerFeedback(opts.promptModel, candidate.steps.map((step) => step.id), {
		better_program_outputs: goodView.prediction ?? {},
		better_program_trajectory: toTrajectory(goodView.trace),
		better_reward_info: goodView.outputMetadata,
		better_reward_value: goodView.score,
		module_names: candidate.steps.map((step) => step.id),
		modules_defn: inspectModules(candidate),
		oracle_metadata: example.outputData,
		program_code: candidate.code,
		program_inputs: example.inputData,
		worse_program_outputs: badView.prediction ?? {},
		worse_program_trajectory: toTrajectory(badView.trace),
		worse_reward_info: badView.outputMetadata,
		worse_reward_value: badView.score
	});
	for (const step of candidate.steps) {
		const advice = result.moduleAdvice[step.id];
		if (advice !== void 0) {
			console.log(`Advice for ${step.id}: ${advice}`);
			step.description = `${step.description}\n\n${advice}`;
		}
	}
	return true;
};
const runRollout = async (program, example, metric, ctx) => {
	const trace = [];
	const runCtx = {
		...ctx,
		trace
	};
	let prediction;
	try {
		prediction = await program.run(example.inputData, runCtx);
	} catch (error) {
		console.warn(error);
	}
	let score = 0;
	let outputMetadata = {};
	try {
		const { score: metricScore, ...metadata } = await metric(example, prediction, runCtx.target);
		score = metricScore;
		outputMetadata = metadata;
	} catch (error) {
		console.warn(error);
	}
	return {
		example,
		outputMetadata,
		prediction,
		score,
		trace
	};
};
const nextBatch = (rng, cursor, trainingSet, batchSize) => {
	if (cursor.instanceIdx + batchSize > trainingSet.length) {
		shuffle(rng, cursor.dataIdxs);
		cursor.instanceIdx = 0;
	}
	const batch = cursor.dataIdxs.slice(cursor.instanceIdx, cursor.instanceIdx + batchSize).map((i) => at([...trainingSet], i, "trainingSet"));
	cursor.instanceIdx += batchSize;
	return batch;
};
const prepareModelsForResampling = (rt, nextRolloutId) => {
	const models = [];
	let id = nextRolloutId;
	if (rt.teacherSettings) {
		models.push({
			model: rt.teacherSettings.model,
			seed: id,
			temperature: rt.teacherSettings.temperature
		});
		id += 1;
	}
	while (models.length < rt.candidates) {
		models.push({
			seed: id,
			temperature: 1
		});
		id += 1;
	}
	return {
		models,
		nextRolloutId: id
	};
};
const sampleBatchRollouts = (rt, rng, programs, avg, batch, models) => {
	const topK = topKPlusBaseline(avg, rt.candidates);
	const runs = [];
	for (const modelCtx of models) for (const example of batch) {
		const srcIdx = softmaxSample(rng, topK, avg, rt.samplingTemperature);
		const rolloutProgram = at(programs, srcIdx, "programs");
		runs.push(() => runRollout(rolloutProgram, example, rt.metric, modelCtx));
	}
	return Promise.all(runs.map((run) => run()));
};
const generateCandidatesFromBuckets = async (rt, rng, poissonRNG, programs, avg, buckets, percentiles) => {
	const topK = topKPlusBaseline(avg, rt.candidates);
	const strategyCount = rt.maxFewShotExamples > 0 ? 2 : 1;
	const generated = [];
	for (const bucket of buckets) {
		const srcIdx = softmaxSample(rng, topK, avg, rt.candidateTemperature);
		const candidate = at(programs, srcIdx, "programs").clone();
		dropExamples(candidate, rt.maxFewShotExamples, rng, poissonRNG);
		const strategyIdx = Math.floor(rng() * strategyCount);
		const useExampleStrategy = rt.maxFewShotExamples > 0 && strategyIdx === 0;
		try {
			if (useExampleStrategy) appendAnExample(bucket, candidate, {
				maxFewShotInputLength: rt.maxFewShotInputLength,
				p10: percentiles.p10
			});
			else await appendARule(bucket, candidate, {
				p10: percentiles.p10,
				p90: percentiles.p90,
				promptModel: rt.promptModel
			});
		} catch (error) {
			console.error(`Strategy failed with error: ${error}`);
			continue;
		}
		generated.push(candidate);
		if (generated.length >= rt.candidates + 1) break;
	}
	return generated;
};
const evaluateOn = async (programsToScore, examples, metric) => {
	const rollouts = await Promise.all(programsToScore.flatMap((program) => examples.map((example) => runRollout(program, example, metric, {}))));
	return programsToScore.map((_, idx) => rollouts.slice(idx * examples.length, (idx + 1) * examples.length).map((rollout) => rollout.score));
};
/** Final selection: candidates+1 programs evenly spaced across the winner
* timeline, always including the untouched student and the last winner. */
const finalistIdxs = (winnersCount, candidates) => {
	const m = winnersCount - 1;
	const n = candidates + 1;
	const spacing = m < 1 ? Array.from({ length: n }, () => 0) : Array.from({ length: n }, (_, i) => roundHalfEven(i * m / (n - 1)));
	return [...new Set(spacing)];
};
const avgOf = (scoreLists) => scoreLists.map((scores) => scores.length > 0 ? mean(scores) : 0);
const trialLogSchema = z.object({
	baselineScore: z.number(),
	candidateScores: z.array(z.number()),
	step: z.number()
});
/** The loop's whole world between batches, as JSON. */
const simbaStateSchema = z.object({
	cursor: z.object({
		dataIdxs: z.array(z.number()),
		instanceIdx: z.number()
	}),
	nextRolloutId: z.number(),
	pool: z.array(promptsSchema),
	poolScores: z.array(z.array(z.number())),
	rng: z.object({
		main: z.number(),
		poisson: z.number()
	}),
	step: z.number(),
	trialLogs: z.array(trialLogSchema),
	winners: z.array(promptsSchema)
});
const traceStepSchema = z.object({
	inputData: fieldsSchema,
	outputData: fieldsSchema,
	stepId: z.string()
});
const rolloutSchema = z.object({
	example: exampleSchema,
	outputMetadata: fieldsSchema,
	prediction: fieldsSchema.optional(),
	score: z.number(),
	trace: z.array(traceStepSchema)
});
const bucketSchema = z.object({
	maxScore: z.number(),
	maxToAvgGap: z.number(),
	maxToMinGap: z.number(),
	rollouts: z.array(rolloutSchema)
});
const rolledOutSchema = simbaStateSchema.extend({
	baselineScore: z.number(),
	batch: z.array(exampleSchema),
	buckets: z.array(bucketSchema),
	p10: z.number(),
	p90: z.number()
});
const proposedSchema = simbaStateSchema.extend({
	baselineScore: z.number(),
	batch: z.array(exampleSchema),
	candidatePrompts: z.array(promptsSchema)
});
/**
* SIMBA (Stochastic Introspective Mini-Batch Ascent) as a Mastra workflow
* over the target `workflow`: each durable loop iteration is one mini-batch
* step, split into a `rollout` step (trajectory sampling through the engine),
* a `propose-candidates` step (the introspective phase — appendARule's
* offerFeedback LM calls live here), and a `score-candidates` step (winner
* selection, pool registration, savePrompts checkpointing). The candidate
* pool, winner timeline, batch cursor, and both RNG streams cross step
* boundaries as JSON, so a storage-backed run resumes mid-optimization
* without redoing completed batches.
*/
const createSIMBAWorkflow = (workflow, config) => {
	const { checkpoint, savePrompts, trainingSet } = config;
	const batchSize = config.batchSize ?? Math.min(32, trainingSet.length);
	const candidates = config.candidates ?? 6;
	const maxSteps = config.maxSteps ?? 8;
	const seed = config.seed ?? 0;
	if (trainingSet.length < batchSize) throw new Error(`TrainingSet too small: ${trainingSet.length} < ${batchSize}`);
	const metric = scorerMetric(resolveScorer(workflow, config.scorer));
	const base = () => workflowToProgram(workflow);
	const rt = {
		batchSize,
		candidateTemperature: config.candidateTemperature ?? .2,
		candidates,
		maxFewShotExamples: config.maxFewShotExamples ?? 4,
		maxFewShotInputLength: config.maxFewShotInputLength ?? 1e5,
		metric,
		promptModel: config.promptModel ?? first(base().steps, "steps").model,
		samplingTemperature: config.samplingTemperature ?? .2,
		teacherSettings: config.teacherSettings
	};
	const examples = [...trainingSet];
	const init = createStep({
		description: "Seed the pool with the baseline and shuffle the batch order",
		execute: () => {
			const rng = createRNG(seed);
			const poissonRNG = createRNG(seed);
			const baselinePrompts = promptsOf(base());
			const dataIdxs = examples.map((_, i) => i);
			shuffle(rng, dataIdxs);
			return Promise.resolve({
				cursor: {
					dataIdxs,
					instanceIdx: 0
				},
				nextRolloutId: 0,
				pool: [baselinePrompts],
				poolScores: [[]],
				rng: {
					main: rng.state,
					poisson: poissonRNG.state
				},
				step: 0,
				trialLogs: [],
				winners: [baselinePrompts]
			});
		},
		id: "init",
		inputSchema: z.object({}),
		outputSchema: simbaStateSchema
	});
	const rollout = createStep({
		description: "Sample program trajectories over the next mini-batch",
		execute: async ({ inputData, resumeData, suspend }) => {
			const state = inputData;
			if (!resumeData && await checkpoint?.({ iteration: state.step })) return await suspend({ iteration: state.step });
			const rng = restoreRNG(state.rng.main);
			const cursor = structuredClone(state.cursor);
			const programs = state.pool.map((prompts) => programFromPrompts(base(), prompts));
			console.log(`Starting batch ${state.step + 1} of ${maxSteps}.`);
			const batch = nextBatch(rng, cursor, examples, batchSize);
			console.log(`Sampling program trajectories on ${batchSize} examples x ${candidates} samples.`);
			const resampling = prepareModelsForResampling(rt, state.nextRolloutId);
			const rollouts = await sampleBatchRollouts(rt, rng, programs, avgOf(state.poolScores), batch, resampling.models);
			const allScores = rollouts.map((r) => r.score);
			const baselineScore = mean(allScores);
			console.log(`Batch ${state.step + 1}: Baseline mini-batch score: ${baselineScore}`);
			return {
				...state,
				baselineScore,
				batch,
				buckets: makeBuckets(rollouts, batchSize),
				cursor,
				nextRolloutId: resampling.nextRolloutId,
				p10: percentile(allScores, 10),
				p90: percentile(allScores, 90),
				rng: {
					...state.rng,
					main: rng.state
				}
			};
		},
		id: "rollout",
		inputSchema: simbaStateSchema,
		outputSchema: rolledOutSchema,
		resumeSchema: z.object({}),
		suspendSchema: z.object({ iteration: z.number() })
	});
	const propose = createStep({
		description: "Generate candidates from the buckets (introspective offerFeedback LM calls)",
		execute: async ({ inputData }) => {
			const { buckets, p10, p90, ...state } = inputData;
			const rng = restoreRNG(state.rng.main);
			const poissonRNG = restoreRNG(state.rng.poisson);
			const programs = state.pool.map((prompts) => programFromPrompts(base(), prompts));
			const generated = await generateCandidatesFromBuckets(rt, rng, poissonRNG, programs, avgOf(state.poolScores), buckets, {
				p10,
				p90
			});
			return {
				...state,
				candidatePrompts: generated.map((candidate) => promptsOf(candidate)),
				rng: {
					main: rng.state,
					poisson: poissonRNG.state
				}
			};
		},
		id: "propose-candidates",
		inputSchema: rolledOutSchema,
		outputSchema: proposedSchema
	});
	const score = createStep({
		description: "Score the candidates, register them, persist the winner",
		execute: async ({ inputData }) => {
			const { baselineScore, batch, candidatePrompts, ...state } = inputData;
			const stepCandidates = candidatePrompts.map((prompts) => programFromPrompts(base(), prompts));
			console.log(`Batch ${state.step + 1}: Evaluating ${stepCandidates.length} programs on ${batchSize} examples.`);
			const candidateScoreLists = await evaluateOn(stepCandidates, batch, metric);
			const candidateScores = candidateScoreLists.map(mean);
			console.log(`Scores after ${state.step + 1} batches: ${candidateScores}, Best: ${candidateScores.length ? Math.max(...candidateScores) : "N/A"}`);
			const winners = [...state.winners];
			if (candidateScores.length > 0) {
				const bestIdx = candidateScores.indexOf(Math.max(...candidateScores));
				const winner = at(candidatePrompts, bestIdx, "candidates");
				winners.push(winner);
				await savePrompts(winner);
			}
			return {
				...state,
				pool: [...state.pool, ...candidatePrompts],
				poolScores: [...state.poolScores, ...candidateScoreLists],
				step: state.step + 1,
				trialLogs: [...state.trialLogs, {
					baselineScore,
					candidateScores,
					step: state.step
				}],
				winners
			};
		},
		id: "score-candidates",
		inputSchema: proposedSchema,
		outputSchema: simbaStateSchema
	});
	const iteration = createWorkflow({
		id: "iteration",
		inputSchema: simbaStateSchema,
		outputSchema: simbaStateSchema
	}).then(rollout).then(propose).then(score).commit();
	const finalize = createStep({
		description: "Evaluate the finalists, persist and land the winner",
		execute: async ({ inputData }) => {
			const state = inputData;
			const finalists = finalistIdxs(state.winners.length, candidates).map((i) => at(state.winners, i, "winners"));
			console.log(`VALIDATION: Evaluating ${finalists.length} programs on the full trainingSet.`);
			const finalScores = (await evaluateOn(finalists.map((prompts) => programFromPrompts(base(), prompts)), examples, metric)).map(mean);
			const candidateData = finalists.map((prompts, idx) => ({
				prompts,
				score: at(finalScores, idx, "final scores")
			})).toSorted((a, b) => b.score - a.score);
			const bestIdx = finalScores.indexOf(Math.max(...finalScores));
			const bestScore = Math.max(...finalScores);
			console.log(`Final trainingSet scores: ${finalScores}, Best: ${bestScore} (at index ${bestIdx})`);
			const best = at(finalists, bestIdx, "finalists");
			await savePrompts(best);
			loadPrompts(workflow, best);
			return {
				candidates: candidateData.map(({ prompts, score: finalistScore }) => {
					return [prompts, { score: finalistScore }];
				}),
				score: bestScore
			};
		},
		id: "finalize",
		inputSchema: simbaStateSchema,
		outputSchema: optimizerResultSchema
	});
	return createWorkflow({
		id: `${workflow.id}.simba`,
		inputSchema: z.object({}),
		outputSchema: optimizerResultSchema
	}).then(init).dountil(iteration, ({ inputData }) => Promise.resolve(inputData.step >= maxSteps)).then(finalize).commit();
};
//#endregion
export { createBootstrapFewShotWorkflow, createExactMatchScorer, createGEPAWorkflow, createLabeledFewShotWorkflow, createSIMBAWorkflow, declareStep, loadPrompts };
