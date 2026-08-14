/** Seeded RNG + sampling utilities shared by the optimizers. */

export type RNG = () => number

/** mulberry32 — small seeded PRNG; only the distributions matter for the port. */
export function createRNG(seed: number): RNG {
  // biome-ignore-start lint/suspicious/noBitwiseOperators: mulberry32 is bit-twiddling by design
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d_2b_79_f5) | 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296
  }
  // biome-ignore-end lint/suspicious/noBitwiseOperators: mulberry32 is bit-twiddling by design
}

/** Knuth's Poisson sampler. */
export function samplePoisson(rng: RNG, lambda: number): number {
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
export function shuffle<T>(rng: RNG, items: T[]): void {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1))
    const a = items[i] as T
    items[i] = items[j] as T
    items[j] = a
  }
}

/** Python `random.sample` parity: k items without replacement. */
export function sample<T>(rng: RNG, items: T[], k: number): T[] {
  if (k > items.length) {
    throw new Error("Sample larger than population")
  }
  const pool = [...items]
  shuffle(rng, pool)
  return pool.slice(0, k)
}

/**
 * Python `random.choices` parity: throws when the weight total is not
 * positive, exactly like the reference (which aborts the run under
 * raise_on_exception). Use where the Python side passes score weights.
 */
export function weightedChoiceStrict<T>(
  rng: RNG,
  items: T[],
  weights: number[]
): T {
  const total = weights.reduce((acc, w) => acc + w, 0)
  if (total <= 0) {
    throw new Error("Total of weights must be greater than zero")
  }
  return weightedChoice(rng, items, weights)
}

/**
 * Pick an item with probability proportional to its weight; uniform fallback
 * when the weight sum is not positive or not finite.
 */
export function weightedChoice<T>(rng: RNG, items: T[], weights: number[]): T {
  const total = weights.reduce((acc, w) => acc + w, 0)
  if (total <= 0 || !Number.isFinite(total)) {
    return items[Math.floor(rng() * items.length)] as T
  }
  let r = rng() * total
  for (const [idx, weight] of weights.entries()) {
    r -= weight
    if (r <= 0) {
      return items[idx] as T
    }
  }
  return items.at(-1) as T
}
