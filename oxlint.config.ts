import { defineConfig } from "oxlint"
import core from "ultracite/oxlint/core"

export default defineConfig({
  ...core,
  ignorePatterns: [
    ...(core.ignorePatterns ?? []),
    // dspy and gepa are vendored upstream clones, not our code.
    "dspy/**",
    "gepa/**",
    ".agents/**",
    ".claude/**",
    "tools/oxlint/anti-slop/**",
  ],
  jsPlugins: [
    { name: "anti-slop", specifier: "./tools/oxlint/anti-slop/index.ts" },
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
    ...core.rules,
    "anti-slop/no-chained-type-assertions": "error",
    "anti-slop/no-conditional-empty-object-spread": "error",
    "anti-slop/no-known-value-widening": "error",
    "anti-slop/no-module-mocking": "error",
    "anti-slop/no-object-parameters": "error",
    "anti-slop/no-reflect-apply": "error",
    "anti-slop/no-reflect-get": "error",
    "anti-slop/no-runtime-typeof": "error",
    "anti-slop/no-shape-in-symbol-names": "error",
    "anti-slop/no-unknown-parameters": "error",
    "anti-slop/no-unknown-returns": "error",
    "anti-slop/no-unknown-type-aliases": "error",
    "anti-slop/no-unsafe-dictionary-type": "error",
    "anti-slop/no-widen-then-assert": "error",
    "anti-slop/require-safety-comment-for-type-assertion": "error",
    "typescript/consistent-type-definitions": ["error", "type"],
  },
})
