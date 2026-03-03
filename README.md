# 🔍 trustandverify

[![PyPI](https://img.shields.io/pypi/v/trustandverify)](https://pypi.org/project/trustandverify/)
[![Python](https://img.shields.io/pypi/pyversions/trustandverify)](https://pypi.org/project/trustandverify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/jemsbhai/trustandverify/actions/workflows/ci.yml/badge.svg)](https://github.com/jemsbhai/trustandverify/actions)

**Agentic knowledge verification using Subjective Logic confidence algebra.**

`trustandverify` decomposes research questions into verifiable claims, gathers evidence from multiple search backends, scores confidence using formal Subjective Logic mathematics (Jøsang 2016), and produces provenance-rich reports with per-claim uncertainty quantification.

---

## The Problem

Every major AI system treats confidence as a single scalar — or ignores it entirely. A `confidence = 0.5` is meaningless: it could mean "strong evidence that the probability is 50%", or "we have literally no evidence and are guessing." These require fundamentally different downstream decisions.

`trustandverify` uses **Subjective Logic opinion tuples** `(belief, disbelief, uncertainty, base_rate)` so you always know *not just what the AI thinks, but how much evidence it has.*

---

## Quick Start

```bash
pip install trustandverify[tavily,gemini]
```

```python
import asyncio
from trustandverify import verify

report = asyncio.run(verify("Is remote work more productive than office work?"))
print(report.summary)
for claim in report.claims:
    op = claim.opinion
    print(f"  [{claim.verdict}] {claim.text}")
    print(f"    P={op.projected_probability():.3f}  b={op.belief:.3f} d={op.disbelief:.3f} u={op.uncertainty:.3f}")
```

---

## Configured Usage

```python
from trustandverify import TrustAgent, TrustConfig
from trustandverify.search import TavilySearch
from trustandverify.llm import GeminiBackend
from trustandverify.storage import SQLiteStorage
from trustandverify.export import JsonLdExporter, MarkdownExporter

config = TrustConfig(num_claims=5, max_sources_per_claim=3)

agent = TrustAgent(
    config=config,
    search=TavilySearch(),
    llm=GeminiBackend(),
    storage=SQLiteStorage("reports.db"),
)

report = await agent.verify("Is nuclear energy safer than solar?")

JsonLdExporter().render_to_file(report, "report.jsonld")
MarkdownExporter().render_to_file(report, "report.md")
```

---

## CLI

```bash
trustandverify "Is coffee healthy?"
trustandverify "Is coffee healthy?" --claims 5 --llm gemini --format markdown
trustandverify ui   # Launch Streamlit dashboard
```

---

## Why Subjective Logic?

| Scenario | Scalar Confidence | `trustandverify` Opinion |
|---|---|---|
| Strong evidence it's 50/50 | 0.5 | `b=0.45, d=0.45, u=0.10` |
| No evidence at all | 0.5 | `b=0.00, d=0.00, u=1.00` |
| Sources violently disagree | 0.5 | `b=0.40, d=0.40, u=0.20` |

Confidence algebra (from [`jsonld-ex`](https://pypi.org/project/jsonld-ex/)):
- **Cumulative fusion** — more independent agreeing sources → lower uncertainty
- **Trust discount** — `.gov`/`.edu` sources weighted higher than Reddit
- **Pairwise conflict detection** — surfaces where sources disagree, quantified

---

## Install Options

```bash
pip install trustandverify                              # core only
pip install trustandverify[tavily,gemini]               # minimal working setup
pip install trustandverify[tavily,brave,openai,sqlite]  # typical setup
pip install trustandverify[all]                         # everything
```

---

## Architecture

```
trustandverify/
├── core/        — TrustAgent, pipeline, models, config
├── scoring/     — Subjective Logic algebra (wraps jsonld-ex)
├── search/      — SearchBackend protocol + Tavily, Brave, Bing, SerpAPI
├── llm/         — LLMBackend protocol + Gemini, OpenAI, Anthropic, Ollama
├── storage/     — StorageBackend protocol + memory, SQLite, Postgres, Neo4j
├── cache/       — CacheBackend protocol + file cache, Redis
├── export/      — ExportBackend protocol + JSON-LD, Markdown, HTML, PDF
├── cli/         — Typer CLI
└── ui/          — Streamlit dashboard
```

---

## References

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.
- [`jsonld-ex`](https://pypi.org/project/jsonld-ex/) — JSON-LD 1.2 extensions with Subjective Logic confidence algebra
- W3C [PROV-O](https://www.w3.org/TR/prov-o/) — Provenance Ontology

---

## License

MIT — see [LICENSE](LICENSE).
