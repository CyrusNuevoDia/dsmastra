import { defineConfig } from "tsdown"

// Peer and dev deps (@mastra/*, @ai-sdk/*, ai, zod) are externalized by default,
// so there's nothing to declare here beyond pinning the emitted extensions to
// the .js/.d.ts names package.json points at.
export default defineConfig({
  clean: true,
  dts: true,
  entry: ["src/index.ts"],
  format: "esm",
  outDir: "dist",
  outExtensions: () => ({ dts: ".d.ts", js: ".js" }),
  platform: "node",
})
