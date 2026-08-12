set dotenv-load := true
set dotenv-path := ".env"
set ignore-comments := true

claude *args:
  bunx @anthropic-ai/claude-code {{args}}

gemini *args:
  bunx @google/gemini-cli {{args}}
