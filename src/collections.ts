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
export const at = <T>(items: readonly T[], index: number, what: string): T => {
  const item = items[index]
  if (item === undefined) {
    throw new Error(
      `${what}: index ${index} is out of range (length ${items.length})`
    )
  }
  return item
}

/** The first element, or a thrown error naming what was empty. */
export const first = <T>(items: readonly T[], what: string): T =>
  at(items, 0, what)

/** The last element, or a thrown error naming what was empty. */
export const last = <T>(items: readonly T[], what: string): T =>
  at(items, items.length - 1, what)

/** The property at `key`, or a thrown error naming the record that lacked it. */
export const prop = <V>(
  record: Readonly<Record<string, V>>,
  key: string,
  what: string
): V => {
  const value = record[key]
  if (value === undefined) {
    throw new Error(`${what}: no entry for key ${key}`)
  }
  return value
}

/** The popped element, or a thrown error naming the stack that was empty. */
export const pop = <T>(items: T[], what: string): T => {
  const item = items.pop()
  if (item === undefined) {
    throw new Error(`${what}: popped an empty stack`)
  }
  return item
}

/** The value for `key`, or a thrown error naming the map that lacked it. */
export const get = <K, V>(map: ReadonlyMap<K, V>, key: K, what: string): V => {
  const value = map.get(key)
  if (value === undefined) {
    throw new Error(`${what}: no entry for key ${String(key)}`)
  }
  return value
}
