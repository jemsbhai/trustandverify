# Getting Started

## Installation

**Minimum working setup** (Tavily search + Gemini LLM):

```bash
pip install trustandverify[tavily,gemini]
```

**Typical setup** (multiple search backends + SQLite persistence):

```bash
pip install trustandverify[tavily,brave,openai,sqlite]
```

**Everything**:

```bash
pip install trustandverify[all]
```

Requires Python 3.10+.

## Environment Variables

Set API keys for your chosen backends:

```bash
export TAVILY_API_KEY="tvly-..."     # Required for TavilySearch
export GEMINI_API_KEY="AIza..."      # Required for GeminiBackend
export OPENAI_API_KEY="sk-..."       # Required for OpenAIBackend
export ANTHROPIC_API_KEY="sk-ant-..."  # Required for AnthropicBackend
export BRAVE_API_KEY="BSA..."        # Required for BraveSearch
export BING_API_KEY="..."            # Required for BingSearch
export SERPAPI_API_KEY="..."         # Required for SerpAPISearch
```

On Windows PowerShell:

```powershell
$env:TAVILY_API_KEY = "tvly-..."
$env:GEMINI_API_KEY = "AIza..."
```

## Quick Start (One-Liner)

```python
import asyncio
from trustandverify import verify

report = asyncio.run(verify("Is remote work more productive than office work?"))

print(report.summary)
for claim in report.claims:
    op = claim.opinion
    print(f"  [{claim.verdict.value}] {claim.text}")
    print(f"    P={op.projected_probability():.3f}  "
          f"b={op.belief:.3f} d={op.disbelief:.3f} u={op.uncertainty:.3f}")
```

## Configured Usage

For full control over backends, scoring, and persistence:

```python
import asyncio
from trustandverify import TrustAgent, TrustConfig
from trustandverify.search import TavilySearch
from trustandverify.llm import GeminiBackend
from trustandverify.storage import SQLiteStorage
from trustandverify.export import JsonLdExporter, MarkdownExporter

config = TrustConfig(
    num_claims=5,              # decompose query into exactly 5 claims
    max_sources_per_claim=3,   # fetch up to 3 search results per claim
    conflict_threshold=0.2,    # flag conflicts above this score
    enable_byzantine=False,    # opt-in Byzantine fusion (off by default)
)

agent = TrustAgent(
    config=config,
    search=TavilySearch(),
    llm=GeminiBackend(),
    storage=SQLiteStorage("reports.db"),
)

report = asyncio.run(agent.verify("Is nuclear energy safer than solar?"))

# Export to multiple formats
JsonLdExporter().render_to_file(report, "report.jsonld")
MarkdownExporter().render_to_file(report, "report.md")
```

## CLI

```bash
# Basic verification
trustandverify verify "Is coffee healthy?"

# With options
trustandverify verify "Is coffee healthy?" --claims 5 --format markdown --output report.md

# Available formats: jsonld, markdown, html, pdf
trustandverify verify "Is coffee healthy?" --format html --output report.html

# Launch Streamlit dashboard
trustandverify ui

# Show version
trustandverify version
```

## Understanding the Output

Each claim in a report has:

- **Verdict**: `supported` (P ≥ 0.7), `contested` (0.3 < P < 0.7), or `refuted` (P ≤ 0.3).
- **Opinion tuple** `(b, d, u)`: belief, disbelief, and uncertainty. Always sum to 1.0.
- **Projected probability**: `P = b + a · u` where `a` is the base rate (default 0.5).
- **Assessment**: LLM-written 2–3 sentence summary of the evidence.
- **Evidence**: Per-source details including trust scores and support/contradiction labels.
- **Conflicts**: Flagged when supporting and contradicting evidence exceed the conflict threshold.

The key insight: two claims can both have P = 0.5 but differ dramatically in uncertainty. `(b=0.45, d=0.45, u=0.10)` means "strong evidence for a 50/50 split." `(b=0.00, d=0.00, u=1.00)` means "we have no evidence at all."

## Jac Integration

For Jaseci/Jac users, a thin walker interface delegates to the Python pipeline:

```bash
# From the jac/ directory
jac run trustgraph.jac "Is remote work more productive?"
jac run trustgraph.jac "Is coffee healthy?" --claims 5 --format markdown
```

The Python interop module can also be used directly:

```python
from trustandverify.jac_interop import jac_verify, jac_export

result = jac_verify("Is coffee healthy?", num_claims=3, search_backend="tavily")
jac_export(result, format="markdown", output_path="report.md")
```

For the full-featured hackathon walker with its own pipeline, see [trustgraph-jac](https://github.com/jemsbhai/trustgraph-jac).

## Next Steps

- [Backends Guide](backends.md) — configure search, LLM, storage, and export backends
- [Confidence Algebra](confidence-algebra.md) — the mathematics behind opinion scoring
- [API Reference](api-reference.md) — full module and function documentation
