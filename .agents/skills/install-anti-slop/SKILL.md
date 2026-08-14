---
name: install-anti-slop
description: Install or migrate anti-slop lint rules through Ultracite's built-in Oxlint preset. Use when a TypeScript or JavaScript repository should enable anti-slop, replace a copied anti-slop plugin, adopt Ultracite 7.10.4 or newer, or wire Ultracite lint and format commands into a justfile.
---

# Install anti-slop

Use Ultracite's bundled `ultracite/oxlint/anti-slop` preset. Do not copy anti-slop source into the target repository or install `@oxlint/plugins`; Ultracite 7.10.4 and newer ship a self-contained plugin bundle.

## Procedure

1. Inspect the repository before changing it:
   - Read its agent instructions and check `git status`.
   - Detect the package manager from `packageManager` and lockfiles.
   - Read the existing Oxlint, Oxfmt, package, and task-runner configuration.
   - Find copied anti-slop plugins and project-specific rule overrides. Preserve unrelated work and intentional overrides.

2. Verify the current Ultracite release with `npm view ultracite version`. Install Ultracite 7.10.4 or newer with compatible `oxlint` and `oxfmt` development dependencies using the repository's package manager. Remove `@oxlint/plugins` when no other local plugin imports it.

3. Extend the built-in presets in `oxlint.config.ts`:

   ```ts
   import { defineConfig } from "oxlint"
   import antiSlop from "ultracite/oxlint/anti-slop"
   import core from "ultracite/oxlint/core"

   export default defineConfig({
     extends: [core, antiSlop],
     ignorePatterns: core.ignorePatterns,
   })
   ```

   Keep `antiSlop` after `core`: it enables all anti-slop rules and disables two core rules that otherwise create conflicting fixes. Merge repository-specific ignores, overrides, and rules without restating the preset's rule list or `jsPlugins` entry.

4. Keep `oxfmt.config.ts` extending `ultracite/oxfmt`. Remove ignores that existed only for a copied plugin, but preserve source, generated-file, and agent-tooling ignores that still apply.

5. Delete the copied plugin and its installer assets after the config resolves the Ultracite preset. Remove stale lock metadata that would identify a rewritten repo-owned skill as an unchanged external install.

6. Wire the repository's task runner through Ultracite. For a justfile, use `ultracite fix` for formatting/autofixes and `ultracite check` for non-writing format and lint checks; keep typechecking and tests as separate recipes or steps.

7. Run the package install, `ultracite doctor`, the repository's full check command, and the skill validator. Fix owned-source findings when the user requested migration or cleanup, but do not weaken rules or add unsafe casts to make the check pass.

Review the final diff and report the Ultracite version, removed local plugin paths and dependency, configuration changes, and exact verification results.
