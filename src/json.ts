/**
 * Predictor outputs are whatever the model produced and the schema accepted, so
 * the code that renders them into prompts genuinely meets an open JSON value. The
 * classification happens exactly once, here, and everything downstream branches on
 * the returned `kind` rather than re-interrogating the representation.
 */

/** A value that survives `JSON.stringify` unchanged. */
export type JSONData =
  | JSONData[]
  | boolean
  | number
  | string
  | { [key: string]: JSONData }
  | null

/** An open value sorted into the JSON case it belongs to. */
export type JSONCase =
  | { entries: [string, unknown][]; kind: "object" }
  | { items: unknown[]; kind: "array" }
  | { kind: "null" }
  | { kind: "primitive"; value: boolean | number }
  | { kind: "string"; value: string }
  | { description: string; kind: "unserializable" }

/** Sort an open value into the JSON case it will serialize as. */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- `unknown` is the input by definition: this function IS the boundary parser that turns an unparsed value into `JSONCase`, which is the named domain type every caller then branches on
export const classifyJSON = (value: unknown): JSONCase => {
  if (value === null) {
    return { kind: "null" }
  }
  if (Array.isArray(value)) {
    return { items: value, kind: "array" }
  }
  /* oxlint-disable anti-slop/no-runtime-typeof -- this IS the parse step the rule asks for: the one place an open value becomes a domain case, so every caller branches on `kind` instead of re-interrogating the representation */
  if (typeof value === "string") {
    return { kind: "string", value }
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return { kind: "primitive", value }
  }
  if (typeof value === "object") {
    return { entries: Object.entries(value), kind: "object" }
  }
  return {
    description: `<non-serializable: ${typeof value}>`,
    kind: "unserializable",
  }
  /* oxlint-enable anti-slop/no-runtime-typeof */
}
