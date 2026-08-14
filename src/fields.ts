import type { z } from "zod"

/**
 * A predictor's inputs or outputs as a flat map of named fields.
 *
 * Field names come from user-supplied Zod schemas and are only known at runtime,
 * so the optimizers — which shuffle demos, splice traces and rewrite instructions
 * without knowing any particular program's shape — genuinely operate on an open
 * map. Every consumer imports this name rather than respelling the dictionary, so
 * the dynamic boundary is one declaration wide instead of scattered everywhere.
 */
// oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type -- the single named contract for the domain's one genuinely dynamic value; see above
export type Fields = Record<string, unknown>

/**
 * A schema whose parsed value is a field map. Constraining predictors to this
 * rather than bare `z.ZodType` is what lets `z.infer<TSchema>` flow into `Fields`
 * positions without a cast: the compiler can prove the parsed value is an object
 * with string keys, which bare `z.ZodType` (satisfiable by `z.string()`) cannot.
 */
// oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type -- `Fields` here is the constraint that makes predictor IO checkable at all; see its declaration above
export type FieldSchema = z.ZodType<Fields>
