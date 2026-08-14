import { defineConfig } from "oxfmt"
import base from "ultracite/oxfmt"

export default defineConfig({
  ...base,
  ignorePatterns: [
    ...(base.ignorePatterns ?? []),
    "dspy/**",
    "gepa/**",
    ".agents/**",
    ".claude/**",
    "tools/oxlint/anti-slop/**",
  ],
  semi: false,
})
