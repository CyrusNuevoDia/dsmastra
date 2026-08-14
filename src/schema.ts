import { z } from "zod"

/** One top-level field of a predictor schema, as rendered into prompts. */
export type SchemaProperty = {
  description?: string
  type?: string
}

/**
 * A predictor schema's top-level fields, keyed by field name. Like `Fields`, the
 * keys come from a user-supplied schema and are only known at runtime; naming the
 * contract keeps that openness in one place.
 */
// oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type -- keys are user schema field names, known only at runtime; see above
export type SchemaProperties = Record<string, SchemaProperty>

/**
 * The slice of a JSON-schema property we render. `z.toJSONSchema` hands back the
 * full JSON-schema union (an object, or the `true`/`false` shorthand), so this
 * parses the two fields we use instead of narrowing the representation by hand.
 */
const schemaPropertySchema = z.object({
  description: z.string().optional(),
  type: z.string().optional(),
})

/** Top-level properties of a zod object schema; `{}` when the schema can't convert. */
export const schemaProperties = (schema: z.ZodType | undefined) => {
  if (!schema) {
    return {}
  }
  try {
    const { properties } = z.toJSONSchema(schema)
    if (!properties) {
      return {}
    }
    const result: SchemaProperties = {}
    for (const [name, property] of Object.entries(properties)) {
      const parsed = schemaPropertySchema.safeParse(property)
      if (parsed.success) {
        result[name] = parsed.data
      }
    }
    return result
  } catch {
    return {}
  }
}
