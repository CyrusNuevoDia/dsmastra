set dotenv-load := true
set dotenv-path := ".env"
set ignore-comments := true

_default:
    @just --list

# Format everything.
fmt:
    bunx oxfmt .

# Lint, including the local anti-slop plugin.
lint:
    bunx oxlint

# Autofix what the linter and formatter can fix on their own.
fix:
    bunx oxlint --fix
    bunx oxfmt .

# Typecheck and bundle to dist/. Deps stay external; `@/*` paths get inlined.
build:
    bunx tsc --noEmit
    bun build src/index.ts --outdir dist --target node --format esm \
      --external '@ai-sdk/*' --external '@mastra/*' --external ai --external zod

# Unit tests, run concurrently.
test:
    bun test --concurrent tests/unit

# Integration tests; these hit the API and need OPENAI_API_KEY in .env.
test-int:
    bun test --concurrent tests/int

# Everything CI should care about: format check, lint, typecheck, unit tests.
check:
    bunx oxfmt --check .
    just lint
    bunx tsc --noEmit
    just test
