import { AnyWorkflow, Step } from "@mastra/core/workflows";
import { z } from "zod";
import { MastraScorer, Trajectory } from "@mastra/core/evals";
import { LanguageModel } from "ai";
//#region src/fields.d.ts
/**
 * A predictor's inputs or outputs as a flat map of named fields.
 *
 * Field names come from user-supplied Zod schemas and are only known at runtime,
 * so the optimizers — which shuffle demos, splice traces and rewrite instructions
 * without knowing any particular program's shape — genuinely operate on an open
 * map. Every consumer imports this name rather than respelling the dictionary, so
 * the dynamic boundary is one declaration wide instead of scattered everywhere.
 */
type Fields = Record<string, unknown>;
/**
 * A schema whose parsed value is a field map. Constraining predictors to this
 * rather than bare `z.ZodType` is what lets `z.infer<TSchema>` flow into `Fields`
 * positions without a cast: the compiler can prove the parsed value is an object
 * with string keys, which bare `z.ZodType` (satisfiable by `z.string()`) cannot.
 */
type FieldSchema = z.ZodType<Fields>;
//#endregion
//#region src/step.d.ts
/**
 * A training example: inputData plus the expected output fields. Doubles as a
 * few-shot example rendered into the prompt between the description and the
 * live input — there is no separate demo type.
 */
type Example<TInput = Fields, TOutput = Fields> = {
  inputData: TInput;
  outputData: TOutput;
};
type TraceStep = {
  inputData: Fields;
  outputData: Fields;
  stepId: string;
};
/** A rollout's Mastra trace linkage, for attaching scores to it in Studio. */
type ScoreTarget = {
  spanId?: string;
  traceId?: string;
};
/** Per-run overrides and trace capture, threaded through a whole program run. */
type RunContext = {
  model?: LanguageModel;
  seed?: number;
  /** Written by the engine runner after a successful run. */
  target?: ScoreTarget;
  temperature?: number;
  trace?: TraceStep[];
  /** Every engine-executed step as Mastra's Trajectory (agents, tools, plain
   * steps included) — written by the engine runner after a successful run. */
  trajectory?: Trajectory;
};
/** AI SDK call settings supported first-class on a step, forwarded verbatim to generateText. */
type StepSettings = {
  abortSignal?: AbortSignal;
  frequencyPenalty?: number;
  headers?: Record<string, string | undefined>;
  maxOutputTokens?: number;
  maxRetries?: number;
  presencePenalty?: number;
  seed?: number;
  stopSequences?: string[];
  temperature?: number;
  topK?: number;
  topP?: number;
};
type StepConfig<TStepId extends string, TInputSchema extends FieldSchema, TOutputSchema extends FieldSchema> = StepSettings & {
  /** The step's instruction text — this is what the optimizers tune. */
  description: string;
  examples?: Example[];
  id: TStepId;
  inputSchema: TInputSchema;
  model: LanguageModel;
  outputSchema: TOutputSchema;
  /** Mastra live-eval scorers, forwarded to the step verbatim. Optimizer
   * rollouts disable these (they run with `disableScorers`) and score through
   * the optimizer's own scorer instead. */
  scorers?: Step["scorers"];
};
/**
 * A Mastra step that is also the unit the optimizers tune: `description` and
 * `examples` are mutable prompt state, everything else is fixed config.
 *
 * `execute` is declared with method syntax on purpose. Mastra's own execute
 * takes a much larger params object (runId, mastra, request context…) that it
 * supplies when running the step inside a workflow, while callers here only
 * ever pass `inputData` — and the optimizers additionally thread a RunContext
 * for per-rollout model/seed/temperature overrides and trace capture, which
 * Mastra simply never passes. Method syntax makes the parameter bivariant, so
 * the real Mastra step satisfies this narrower view without an assertion.
 */
type DeclarativeStep<TStepId extends string = string, TInputSchema extends FieldSchema = FieldSchema, TOutputSchema extends FieldSchema = FieldSchema> = {
  description: string;
  examples: Example[];
  execute(params: {
    inputData: z.infer<TInputSchema>;
  }, ctx?: RunContext): Promise<z.infer<TOutputSchema>>;
  id: TStepId;
  inputSchema: TInputSchema;
  model: LanguageModel;
  outputSchema: TOutputSchema;
  scorers?: Step["scorers"];
  settings: StepSettings;
};
declare const declareStep: <TStepId extends string, TInputSchema extends FieldSchema, TOutputSchema extends FieldSchema>(config: StepConfig<TStepId, TInputSchema, TOutputSchema>) => DeclarativeStep<TStepId, TInputSchema, TOutputSchema>;
//#endregion
//#region src/scorers.d.ts
type AnyScorer = MastraScorer<any, any, any, any>;
/**
 * The optimization objective: a Mastra scorer, or the registration key of one
 * on the Mastra instance the workflow is registered with. Define a scorer once
 * with `createScorer`, attach it for live evals, register it on Mastra — and
 * hand the same scorer (or its key) to an optimizer.
 */
type ScorerRef = AnyScorer | string;
/**
 * A scorer where every expected output field must strictly equal the
 * prediction's for a score of 1, else 0. A factory rather than a shared
 * instance: registered scorers carry a mutable Mastra backpointer, so sharing
 * one across Mastra instances would cross-wire their observability.
 */
declare const createExactMatchScorer: () => MastraScorer<"exact-match", any, any, Record<"generateScoreStepResult", number>>;
//#endregion
//#region src/optimizers/utils.d.ts
/**
 * The tuned prompt state of a workflow: everything an optimizer changes and
 * nothing it doesn't. JSON-safe, so callers can persist it wherever they like.
 */
type Prompts = {
  steps: Record<string, {
    description: string;
    examples: Example[];
  }>;
  version: 1;
};
/**
 * Where tuned prompts go. Required on every optimizer config — the types make
 * forgetting to persist impossible. Called with the current best prompts every
 * time an optimizer improves on them, so a crashed run still leaves its best
 * result behind (last write wins).
 */
type SavePrompts = (prompts: Prompts) => Promise<unknown>;
/** Every optimizer workflow ends in this shape: candidate snapshots with their
 * scores (best first for the search optimizers), plus the winner's score. */
declare const optimizerResultSchema: z.ZodObject<{
  candidates: z.ZodArray<z.ZodTuple<[z.ZodObject<{
    steps: z.ZodRecord<z.ZodString, z.ZodObject<{
      description: z.ZodString;
      examples: z.ZodArray<z.ZodObject<{
        inputData: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        outputData: z.ZodRecord<z.ZodString, z.ZodUnknown>;
      }, z.core.$strip>>;
    }, z.core.$strip>>;
    version: z.ZodLiteral<1>;
  }, z.core.$strip>, z.ZodObject<{
    score: z.ZodNumber;
  }, z.core.$strip>], null>>;
  score: z.ZodNumber;
}, z.core.$strip>;
type OptimizerResult = z.infer<typeof optimizerResultSchema>;
/**
 * Pause hook shared by the optimizer workflows: called at the top of every
 * loop iteration with the optimizer's progress; returning true suspends the
 * run (durably, via Mastra's suspend), to be continued later with
 * `run.resume()`. Human-in-the-loop checkpointing for long optimizations.
 */
type OptimizerCheckpoint = (progress: {
  iteration: number;
}) => boolean | Promise<boolean>;
/**
 * Apply saved prompts to a workflow's live steps, in place, and return the
 * same workflow. Throws when the prompts' step ids don't exactly match the
 * workflow's, so a stale snapshot fails loudly instead of half-applying.
 * Parsing and storage are the caller's problem.
 */
declare const loadPrompts: <TWorkflow extends AnyWorkflow>(workflow: TWorkflow, prompts: Prompts) => TWorkflow;
//#endregion
//#region src/optimizers/bootstrap-few-shot.d.ts
type BootstrapFewShotConfig = {
  /** Pause hook: called before every teacher attempt; returning true suspends
   * the run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint;
  /** Optional trajectory gate: a Mastra `type: "trajectory"` scorer (or its
   * registration key) that sees each teacher rollout as a Trajectory in
   * `run.output` — one workflow_step entry per engine-executed step (agents,
   * tools, and plain steps included; a nested workflow is a single entry;
   * loop iterations collapse to one entry per step id). Per-step inputs are
   * not recorded — the workflow input sits at
   * `rawWorkflowResult.stepResults.input`, and each later step's input is the
   * prior entry's output. The gate decides demo acceptance
   * in place of the objective scorer. Accepted when its score reaches
   * `threshold`; with no threshold, any score above zero. A gate throw counts
   * toward maxErrors, same as a rollout failure. */
  gate?: {
    scorer: ScorerRef;
    threshold?: number;
  };
  /** Caught per-attempt errors allowed before the run aborts. */
  maxErrors?: number;
  maxFewShotExamples?: number;
  maxLabeledExamples?: number;
  maxRounds?: number;
  savePrompts: SavePrompts;
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Scores the compiled workflow, and — unless
   * a `gate` is set — also gates teacher-trace acceptance; a scorer whose
   * generateScore returns 1 accepts every trace. */
  scorer: ScorerRef;
  /** Accept a teacher trace when the objective score reaches this; default:
   * score > 0. Only meaningful without a `gate` (set `gate.threshold` there). */
  scoreThreshold?: number;
  teacher?: AnyWorkflow;
  trainingSet: readonly Example[];
  teacherSettings?: {
    model?: LanguageModel;
    temperature?: number;
  };
};
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
declare const createBootstrapFewShotWorkflow: (workflow: AnyWorkflow, config: BootstrapFewShotConfig) => import("@mastra/core/workflows").Workflow<import("@mastra/core/workflows").DefaultEngineType, import("@mastra/core/workflows").Step<string, unknown, unknown, unknown, unknown, unknown, any, unknown>[], `${any}.bootstrap-few-shot`, unknown, Record<string, never>, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, unknown>;
//#endregion
//#region src/optimizers/gepa/adapter.d.ts
type ReflectionModel = LanguageModel | ((prompt: string) => Promise<string>);
//#endregion
//#region src/optimizers/gepa/index.d.ts
declare const AUTO_CANDIDATES: {
  readonly heavy: 18;
  readonly light: 6;
  readonly medium: 12;
};
type GEPAAuto = keyof typeof AUTO_CANDIDATES;
type EngineTuning<TInput, TOutput> = {
  addFormatFailureAsFeedback?: boolean;
  candidateSelectionStrategy?: "currentBest" | "pareto";
  componentSelector?: "all" | "roundRobin";
  failureScore?: number;
  maxMergeInvocations?: number;
  perfectScore?: number;
  reflectionMinibatchSize?: number;
  seed?: number;
  skipPerfectScore?: boolean;
  useMerge?: boolean;
  validationSet?: Example<TInput, TOutput>[];
  warnOnScoreMismatch?: boolean;
};
type GEPAConfig = EngineTuning<Fields, Fields> & {
  /** Exactly one of `auto`, `maxFullEvals`, `maxScorerCalls` must be set. */
  auto?: GEPAAuto;
  /** Pause hook: called before every iteration; returning true suspends the
   * run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint;
  /** When > 0, a bootstrapFewShot pre-pass installs few-shot examples first. */
  maxFewShotExamples?: number;
  /** Labeled backfill cap for the pre-pass; defaults to maxFewShotExamples. */
  maxLabeledExamples?: number;
  maxFullEvals?: number;
  /** Budget cap counted in scorer runs — DSPy's maxMetricCalls. */
  maxScorerCalls?: number;
  /** LM used to propose new descriptions; defaults to the first step's model. */
  reflectionModel?: ReflectionModel;
  savePrompts: SavePrompts;
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Its `reason` (generateReason step) feeds
   * GEPA's reflection LM as feedback. */
  scorer: ScorerRef;
  trainingSet: readonly Example[];
};
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
declare const createGEPAWorkflow: (workflow: AnyWorkflow, config: GEPAConfig) => import("@mastra/core/workflows").Workflow<import("@mastra/core/workflows").DefaultEngineType, import("@mastra/core/workflows").Step<string, unknown, unknown, unknown, unknown, unknown, any, unknown>[], `${any}.gepa`, unknown, Record<string, never>, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, unknown>;
//#endregion
//#region src/optimizers/labeled-few-shot.d.ts
type LabeledFewShotConfig = {
  maxFewShotExamples?: number;
  savePrompts: SavePrompts;
  /** Scores the compiled workflow over the trainingSet: a Mastra scorer, or
   * its registration key on the workflow's Mastra instance. */
  scorer: ScorerRef;
  trainingSet: readonly Example[];
};
/**
 * LabeledFewShot as a Mastra workflow over the target `workflow`: install up
 * to `maxFewShotExamples` labeled trainingSet examples as few-shot examples
 * on every step (dspy.teleprompt.vanilla.LabeledFewShot). Compiling makes no
 * LM calls; the evaluate step runs the compiled workflow over the trainingSet
 * once, and the apply step lands the compiled prompt state in place on the
 * target workflow. All inter-step state is JSON, so a storage-backed run is
 * durable and observable step by step.
 */
declare const createLabeledFewShotWorkflow: (workflow: AnyWorkflow, config: LabeledFewShotConfig) => import("@mastra/core/workflows").Workflow<import("@mastra/core/workflows").DefaultEngineType, import("@mastra/core/workflows").Step<string, unknown, unknown, unknown, unknown, unknown, any, unknown>[], `${any}.labeled-few-shot`, unknown, Record<string, never>, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, unknown>;
//#endregion
//#region src/optimizers/simba.d.ts
type SIMBAConfig = {
  batchSize?: number;
  /** Pause hook: called before every mini-batch; returning true suspends the
   * run durably, to be continued with `run.resume()`. */
  checkpoint?: OptimizerCheckpoint;
  candidates?: number;
  candidateTemperature?: number;
  maxFewShotExamples?: number;
  maxFewShotInputLength?: number;
  maxSteps?: number;
  /** LM used to write rules; defaults to the first step's model. */
  promptModel?: LanguageModel;
  samplingTemperature?: number;
  savePrompts: SavePrompts;
  /** The optimization objective: a Mastra scorer, or its registration key on
   * the workflow's Mastra instance. Its `reason` rides along as reward info
   * for SIMBA's reflection. */
  scorer: ScorerRef;
  seed?: number;
  teacherSettings?: {
    model: LanguageModel;
    temperature?: number;
  };
  trainingSet: readonly Example[];
};
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
declare const createSIMBAWorkflow: (workflow: AnyWorkflow, config: SIMBAConfig) => import("@mastra/core/workflows").Workflow<import("@mastra/core/workflows").DefaultEngineType, import("@mastra/core/workflows").Step<string, unknown, unknown, unknown, unknown, unknown, any, unknown>[], `${any}.simba`, unknown, Record<string, never>, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, {
  candidates: [{
    steps: Record<string, {
      description: string;
      examples: {
        inputData: Record<string, unknown>;
        outputData: Record<string, unknown>;
      }[];
    }>;
    version: 1;
  }, {
    score: number;
  }][];
  score: number;
}, unknown>;
//#endregion
export { type BootstrapFewShotConfig, type Example, type GEPAConfig, type LabeledFewShotConfig, type OptimizerCheckpoint, type OptimizerResult, type Prompts, type SIMBAConfig, type SavePrompts, type ScorerRef, type StepSettings, createBootstrapFewShotWorkflow, createExactMatchScorer, createGEPAWorkflow, createLabeledFewShotWorkflow, createSIMBAWorkflow, declareStep, loadPrompts };