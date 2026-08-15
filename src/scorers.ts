import { createScorer } from "@mastra/core/evals"
import type { MastraScorer, Trajectory } from "@mastra/core/evals"
import type { AnyWorkflow } from "@mastra/core/workflows"

import type { Fields } from "./fields"
import type { Example, ScoreTarget } from "./step"

// oxlint-disable-next-line typescript/no-explicit-any -- MastraScorer's own defaults collapse `score` to never; `any` is the widest scorer Mastra itself uses (see Mastra's Workflow typing)
export type AnyScorer = MastraScorer<any, any, any, any>

/**
 * The optimization objective: a Mastra scorer, or the registration key of one
 * on the Mastra instance the workflow is registered with. Define a scorer once
 * with `createScorer`, attach it for live evals, register it on Mastra — and
 * hand the same scorer (or its key) to an optimizer.
 */
export type ScorerRef = AnyScorer | string

/**
 * What every metric hands back internally: a score, plus whatever else rides
 * along as metadata — `feedback` text for GEPA's reflection LM, sub-scores.
 * The public API speaks Mastra scorers; this is the function form the
 * optimizer internals run on.
 */
export type MetricResult = { score: number } & Fields

/** A metric's return, or a promise of one. */
export type MetricOutput = MetricResult | Promise<MetricResult>

export type Metric<TInput = Fields, TOutput = Fields> = (
  example: Example<TInput, TOutput>,
  prediction: TOutput | undefined,
  /** The rollout's trace linkage, when the engine produced one. */
  target?: ScoreTarget
) => MetricOutput

/**
 * Resolve a ScorerRef against the workflow. A string is strictly a Mastra
 * registration key — resolved through the workflow's Mastra instance, never a
 * fuzzy name or step lookup — so an unregistered workflow fails loudly.
 */
export const resolveScorer = (
  workflow: AnyWorkflow,
  ref: ScorerRef
): AnyScorer => {
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- ScorerRef is a published `scorer | registration-key` union; the string form is only distinguishable at runtime
  if (typeof ref !== "string") {
    return ref
  }
  const { mastra } = workflow
  if (!mastra) {
    throw new Error(
      `Cannot resolve scorer "${ref}": workflow ${workflow.id} is not registered on a Mastra instance`
    )
  }
  return mastra.getScorer(ref)
}

/**
 * A Mastra scorer as the internal metric contract. Runs the scorer with
 * Mastra's own runEvals mapping — input: the example's inputData, output: the
 * prediction, groundTruth: the expected outputData — and links the score to
 * the rollout's trace when one exists, so registered scorers surface each
 * evaluation in Mastra observability. The scorer's `reason` becomes `feedback`
 * (GEPA's reflection LM reads it; SIMBA carries it as reward info).
 */
type ScorerRunInput = {
  groundTruth: unknown
  input: unknown
  output: unknown
  scoreSource: "experiment"
  targetScope?: "span"
  targetSpanId?: string
  targetTraceId?: string
}

/**
 * Narrow a scorer run's result to the internal metric shape, rejecting
 * non-finite scores; the scorer's `reason` rides along as `feedback`.
 */
const metricResultOf = (
  scorer: AnyScorer,
  result: { reason?: unknown; score?: unknown }
): MetricResult => {
  // SAFETY: MastraScorer<…, any> erases the score type; a scorer without a
  // generateScore step is rejected by Mastra at run time, so this narrows
  // what Mastra already guarantees is a number — and the finite check below
  // re-verifies it.
  const score = result.score as number
  if (!Number.isFinite(score)) {
    throw new TypeError(`Scorer ${scorer.id} returned a non-finite score`)
  }
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- `reason` only exists when the scorer defines a generateReason step; Mastra types the result loosely, so the runtime shape is the only evidence
  return typeof result.reason === "string"
    ? { feedback: result.reason, score }
    : { score }
}

export const scorerMetric =
  (scorer: AnyScorer): Metric =>
  async (example, prediction, target) => {
    if (prediction === undefined) {
      throw new Error(
        `Scorer ${scorer.id} has no prediction to score (the rollout failed)`
      )
    }
    const runInput: ScorerRunInput = {
      groundTruth: example.outputData,
      input: example.inputData,
      output: prediction,
      scoreSource: "experiment",
    }
    if (target?.traceId) {
      runInput.targetScope = "span"
      runInput.targetSpanId = target.spanId
      runInput.targetTraceId = target.traceId
    }
    return metricResultOf(scorer, await scorer.run(runInput))
  }

/** A metric over the rollout's trajectory rather than its final prediction. */
export type TrajectoryMetric<TInput = Fields, TOutput = Fields> = (
  example: Example<TInput, TOutput>,
  trajectory: Trajectory
) => MetricOutput

/**
 * A Mastra trajectory scorer as a trajectory metric. Runs the scorer the way
 * Mastra's runEvals runs `type: "trajectory"` scorers — the Trajectory as
 * `output`, the example's inputData as `input`, expected outputData as
 * `groundTruth`. The trajectory is cloned first: its rawWorkflowResult aliases
 * the very step outputs that become few-shot demos after acceptance, so a
 * scorer mutating its run input must not reach them. No trace linkage: these
 * runs happen mid-bootstrap, before any score target exists.
 */
export const trajectoryScorerMetric =
  (scorer: AnyScorer): TrajectoryMetric =>
  async (example, trajectory) =>
    metricResultOf(
      scorer,
      await scorer.run({
        groundTruth: example.outputData,
        input: example.inputData,
        output: structuredClone(trajectory),
        scoreSource: "experiment",
        targetScope: "trajectory",
      })
    )

/**
 * A scorer where every expected output field must strictly equal the
 * prediction's for a score of 1, else 0. A factory rather than a shared
 * instance: registered scorers carry a mutable Mastra backpointer, so sharing
 * one across Mastra instances would cross-wire their observability.
 */
export const createExactMatchScorer = () =>
  createScorer({
    description:
      "1 when every expected output field strictly equals the prediction's, else 0.",
    id: "exact-match",
  }).generateScore(({ run }) =>
    // SAFETY: the plain createScorer overload types run.groundTruth and
    // run.output as any; this scorer is documented for record-shaped workflow
    // fields, and Object.entries / optional indexing tolerate anything else.
    Object.entries((run.groundTruth ?? {}) as Fields).every(
      ([key, value]) => (run.output as Fields | undefined)?.[key] === value
    )
      ? 1
      : 0
  )
