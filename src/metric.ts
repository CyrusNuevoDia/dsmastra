import type { Fields } from "@/fields"

/**
 * What every metric hands back: a score, plus whatever else the metric wants to
 * report alongside it — feedback text for GEPA's reflection LM, sub-scores, or
 * any other field an optimizer carries through as metadata.
 *
 * Always an object, never a bare number. A metric that only has a score writes
 * `{ score }`, which costs one pair of braces and means no caller ever has to ask
 * which of two shapes it received.
 */
export type MetricResult = { score: number } & Fields

/** A metric's return, or a promise of one. */
export type MetricOutput = MetricResult | Promise<MetricResult>
