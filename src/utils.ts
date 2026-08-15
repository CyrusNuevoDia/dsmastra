/** Structural equality via JSON rendering — for JSON-safe values only (the
 * optimizer currency: examples, prompts, fields). Key order matters, which is
 * fine here because compared values come from the same construction sites. */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- a generic JSON comparator: callers guarantee JSON-safe values, and any named type would just restate `unknown`
export const isEqualJSON = (a: unknown, b: unknown): boolean =>
  JSON.stringify(a) === JSON.stringify(b)
