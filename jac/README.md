# trustandverify — Jac Interface

Thin Jac walker that delegates the full verification pipeline to the Python
`trustandverify` library. All scoring, search, LLM calls, and export happen
in Python — the Jac layer provides the graph structure and walker interface.

## Usage

```bash
# Basic verification
jac run trustgraph.jac "Is remote work more productive?"

# With options
jac run trustgraph.jac "Is coffee healthy?" --claims 5 --format markdown

# All options
jac run trustgraph.jac "Is nuclear energy safe?" \
    --claims 4 \
    --search tavily \
    --llm gemini \
    --storage sqlite \
    --db reports.db \
    --format html \
    --byzantine \
    --verbose
```

## Options

| Flag | Values | Default | Description |
|---|---|---|---|
| `--claims` | 0-8 | 0 (auto) | Number of claims to decompose into |
| `--search` | tavily, brave, bing, serpapi, multi | tavily | Search backend |
| `--llm` | gemini, openai, anthropic, ollama | gemini | LLM backend |
| `--storage` | memory, sqlite | memory | Storage backend |
| `--db` | path | trustandverify.db | SQLite database path |
| `--format` | jsonld, markdown, html, none | jsonld | Export format |
| `--byzantine` | flag | off | Enable Byzantine-resistant fusion |
| `--verbose` / `-v` | flag | off | Print step-by-step progress |

## Architecture

The Jac walker is a thin interface — it calls `jac_verify()` and `jac_export()`
from `trustandverify.jac_interop`, which delegates to `TrustAgent.verify()`.
The graph structure (Query → VerifiedClaim, ReportNode) is built from the
returned result dict.

For the full-featured hackathon walker with its own pipeline, see
[trustgraph-jac](https://github.com/jemsbhai/trustgraph-jac).
