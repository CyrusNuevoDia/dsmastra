/** Seeded RNG + sampling utilities shared by the optimizers. */

import { at, last } from "#src/collections"

export type RNG = () => number

/**
 * A seeded RNG whose entire internal state is the single uint32 `state` —
 * read it to checkpoint the stream, write it to restore. `createRNG(seed)`
 * and a restored `state` produce byte-identical sequences, which is what
 * lets the durable optimizer workflows serialize randomness between steps.
 */
export type SerializableRNG = RNG & { state: number }

/** mulberry32 — small seeded PRNG; only the distributions matter for the port. */
export const createRNG = (seed: number): SerializableRNG => {
  /* oxlint-disable no-bitwise -- mulberry32 is bit-twiddling by design */
  let state = seed >>> 0
  const rng = () => {
    // oxlint-disable-next-line unicorn/prefer-math-trunc -- `| 0` wraps to 32 bits here, which is the point; Math.trunc would leave the value above 2^31 and break the sequence
    state = (state + 0x6d_2b_79_f5) | 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296
  }
  // SAFETY: the defineProperty call right here installs the `state` accessor
  // the SerializableRNG shape adds on top of the bare function.
  return Object.defineProperty(rng, "state", {
    get: () => state,
    set: (next: number) => {
      state = next >>> 0
    },
  }) as SerializableRNG
  /* oxlint-enable no-bitwise */
}

/** An RNG resumed from a checkpointed `state` (uint32), mid-stream. */
export const restoreRNG = (state: number): SerializableRNG => {
  const rng = createRNG(0)
  rng.state = state
  return rng
}

/** Knuth's Poisson sampler. */
export const samplePoisson = (rng: RNG, lambda: number): number => {
  if (lambda <= 0) {
    return 0
  }
  const limit = Math.exp(-lambda)
  let k = 0
  let p = 1
  do {
    k += 1
    p *= rng()
  } while (p > limit)
  return k - 1
}

/** In-place Fisher–Yates shuffle. */
export const shuffle = <T>(rng: RNG, items: T[]): void => {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1))
    const a = at(items, i, "shuffle")
    items[i] = at(items, j, "shuffle")
    items[j] = a
  }
}

/** Python `random.sample` parity: k items without replacement. */
export const sample = <T>(rng: RNG, items: T[], k: number): T[] => {
  if (k > items.length) {
    throw new Error("Sample larger than population")
  }
  const pool = [...items]
  shuffle(rng, pool)
  return pool.slice(0, k)
}

/**
 * Pick an item with probability proportional to its weight; uniform fallback
 * when the weight sum is not positive or not finite.
 */
export const weightedChoice = <T>(
  rng: RNG,
  items: T[],
  weights: number[]
): T => {
  const total = weights.reduce((acc, w) => acc + w, 0)
  if (total <= 0 || !Number.isFinite(total)) {
    return at(items, Math.floor(rng() * items.length), "weightedChoice pool")
  }
  let r = rng() * total
  for (const [idx, weight] of weights.entries()) {
    r -= weight
    if (r <= 0) {
      return at(items, idx, "weightedChoice pool")
    }
  }
  return last(items, "weightedChoice pool")
}

/**
 * Python `random.choices` parity: throws when the weight total is not
 * positive, exactly like the reference (which aborts the run under
 * raise_on_exception). Use where the Python side passes score weights.
 */
export const weightedChoiceStrict = <T>(
  rng: RNG,
  items: T[],
  weights: number[]
): T => {
  const total = weights.reduce((acc, w) => acc + w, 0)
  if (total <= 0) {
    throw new Error("Total of weights must be greater than zero")
  }
  return weightedChoice(rng, items, weights)
}
