import type { z } from "zod"

import type { FieldSchema, Fields } from "./fields"
import { schemaProperties } from "./schema"
import type { Example } from "./step"

/**
 * Prompt-template formatting shared across the steps and optimizers: rendering
 * values, schemas, and few-shot examples into prompt text, and extracting
 * instruction text back out of LM responses. Optimizer-specific prompt
 * templates stay with their optimizers.
 */

/** The prompt a step sends: description, few-shot examples, then the live input. */
export const renderPrompt = (
  description: string,
  examples: Example[],
  inputData: Fields
): string => {
  const parts = [description]
  for (const example of examples) {
    parts.push(
      `Example:\nInput:\n${JSON.stringify(example.inputData)}\nOutput:\n${JSON.stringify(example.outputData)}`
    )
  }
  parts.push(`Input:\n${JSON.stringify(inputData)}`)
  return parts.join("\n\n")
}

/** A value that survives `JSON.stringify` unchanged. */
export type JSONData =
  | JSONData[]
  | boolean
  | number
  | string
  | { [key: string]: JSONData }
  | null

/** Replace non-serializable values recursively, like dspy's recursive_mask. */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- a step output is whatever the model produced and the schema accepted, so this genuinely meets an open value; `JSONData` is the named type it returns
export const recursiveMask = (value: unknown): JSONData => {
  if (value === null) {
    return null
  }
  if (Array.isArray(value)) {
    return value.map(recursiveMask)
  }
  /* oxlint-disable anti-slop/no-runtime-typeof -- this IS the parse step the rule asks for: an open value becomes `JSONData`, with anything unserializable replaced by a printable placeholder */
  switch (typeof value) {
    case "boolean":
    case "number":
    case "string": {
      return value
    }
    case "object": {
      return Object.fromEntries(
        Object.entries(value).map(([k, v]) => [k, recursiveMask(v)])
      )
    }
    default: {
      return `<non-serializable: ${typeof value}>`
    }
  }
  /* oxlint-enable anti-slop/no-runtime-typeof */
}

/** A field value as prompt text: strings raw, everything else masked JSON. */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- same open boundary as `recursiveMask`: a field value is whatever the step produced
export const serializeField = (value: unknown): string =>
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- strings go into the prompt raw; everything else routes through `recursiveMask`, which is the parse
  typeof value === "string"
    ? value
    : JSON.stringify(recursiveMask(value), null, 2)

/** Every field rendered with String(), for records shown verbatim in prompts. */
export const stringifyFields = (fields: Fields): Record<string, string> =>
  Object.fromEntries(
    Object.entries(fields).map(([key, value]) => [key, String(value)])
  )

/** One `name (type): description` line per schema property. */
export const fieldDescriptionLines = (schema: z.ZodType): string =>
  Object.entries(schemaProperties(schema))
    .map(
      ([name, prop]) =>
        `${name} (${prop.type ?? "unknown"})${prop.description ? `: ${prop.description}` : ""}`
    )
    .join("\n")

/** One `name: type` line per schema property — the compact structure sketch. */
export const expectedStructure = (schema: z.ZodType | undefined): string =>
  Object.entries(schemaProperties(schema))
    .map(([name, prop]) => `${name}: ${prop.type ?? "unknown"}`)
    .join("\n")

export const indentContinuations = (text: string): string =>
  ["", ...text.split("\n")].join("\n\t\t")

const MODULE_SEPARATOR = "-".repeat(80)

/** Every module's I/O fields and current instructions, DSPy inspect-style. */
export const inspectModules = (
  modules: readonly {
    description: string
    id: string
    inputSchema: FieldSchema
    outputSchema: FieldSchema
  }[]
): string => {
  const blocks = [MODULE_SEPARATOR]
  for (const step of modules) {
    blocks.push(
      `Module ${step.id}`,
      `\n\tInput Fields:${indentContinuations(fieldDescriptionLines(step.inputSchema))}`,
      `\tOutput Fields:${indentContinuations(fieldDescriptionLines(step.outputSchema))}`,
      `\tOriginal Instructions: ${indentContinuations(step.description)}`,
      MODULE_SEPARATOR
    )
  }
  return blocks.map((block) => block.replaceAll(/^\n+|\n+$/gu, "")).join("\n")
}

const MAX_HEADER_DEPTH = 6

/**
 * Byte-for-byte port of the Python renderer: scalars end with a blank line
 * (`value\n\n`), headers with a single newline, empty dicts/lists add a bare
 * newline, and the depth cap applies on recursion.
 */
// oxlint-disable-next-line anti-slop/no-unknown-parameters -- a step output is whatever the model produced and the schema accepted, so the renderer genuinely meets an open value
const renderValue = (value: unknown, level: number): string => {
  const header = "#".repeat(level)
  const nextLevel = Math.min(level + 1, MAX_HEADER_DEPTH)
  if (Array.isArray(value)) {
    let s = ""
    for (const [k, item] of value.entries()) {
      s += `${header} Item ${k + 1}\n${renderValue(item, nextLevel)}`
    }
    if (value.length === 0) {
      s += "\n"
    }
    return s
  }
  // oxlint-disable-next-line anti-slop/no-runtime-typeof -- objects nest into headers; every other JSON case renders as a scalar below
  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value)
    let s = ""
    for (const [key, sub] of entries) {
      s += `${header} ${key}\n${renderValue(sub, nextLevel)}`
    }
    if (entries.length === 0) {
      s += "\n"
    }
    return s
  }
  return `${String(value).trim()}\n\n`
}

/** Markdown rendering of keyed example records, one `# Example n` per entry. */
export const renderSideInfo = (examples: Fields[]): string =>
  examples
    .map((example, n) => {
      let s = `# Example ${n + 1}\n`
      for (const [key, value] of Object.entries(example)) {
        s += `## ${key}\n${renderValue(value, 3)}`
      }
      return s
    })
    .join("\n\n")

const LANGUAGE_TAG = /^\S*\n/u
const LEADING_FENCE = /^```\S*\n?/u

/**
 * Byte-for-byte port of Python's output_extractor (which receives the
 * response pre-stripped): text between the first and last fences with a
 * leading language tag stripped; incomplete blocks fall back to stripping a
 * leading fence (+ optional language tag and newline) or a trailing fence,
 * else the whole trimmed response.
 */
export const extractInstructionText = (response: string): string => {
  const lmOut = response.trim()
  const start = lmOut.indexOf("```") + 3
  const end = lmOut.lastIndexOf("```")
  if (start >= end) {
    if (lmOut.startsWith("```")) {
      const match = LEADING_FENCE.exec(lmOut)
      if (match) {
        return lmOut.slice(match[0].length).trim()
      }
      return lmOut
    }
    if (lmOut.endsWith("```")) {
      return lmOut.slice(0, -3).trim()
    }
    return lmOut
  }
  let content = lmOut.slice(start, end)
  const tag = LANGUAGE_TAG.exec(content)
  if (tag) {
    content = content.slice(tag[0].length)
  }
  return content.trim()
}
