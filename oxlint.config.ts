import { defineConfig } from "oxlint"
import antiSlop from "ultracite/oxlint/anti-slop"
import core from "ultracite/oxlint/core"

export default defineConfig({
  extends: [core, antiSlop],
  ignorePatterns: [
    ...(core.ignorePatterns ?? []),
    // dspy and gepa are vendored upstream clones, not our code.
    "dspy/**",
    "gepa/**",
    ".agents/**",
    ".claude/**",
  ],
  overrides: [
    ...(core.overrides ?? []),
    {
      // src/index.ts is the package entrypoint, so it is a barrel file by design.
      files: ["src/index.ts"],
      rules: { "import/no-namespace": "off" },
    },
    {
      // Tests build deliberately malformed fixtures to prove the optimizers cope
      // with them, so the rules that demand a parsed, named contract at every
      // boundary are asking for ceremony that would obscure what each test is
      // actually asserting. Everything under src/ is still held to all of them.
      files: ["tests/**"],
      rules: {
        "anti-slop/no-conditional-empty-object-spread": "off",
        "anti-slop/no-known-value-widening": "off",
        "anti-slop/no-runtime-typeof": "off",
        "anti-slop/no-unsafe-dictionary-type": "off",
        "anti-slop/require-safety-comment-for-type-assertion": "off",
        "unicorn/consistent-function-scoping": "off",
      },
    },
  ],
  rules: {
    "typescript/consistent-type-definitions": ["error", "type"],
  },
})
