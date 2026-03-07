# API Reference

## Top-Level API

### `verify(query, *, num_claims=0, verbose=False) → Report`

*Async.* One-liner verification using auto-configured backends from environment variables. Requires `TAVILY_API_KEY` and `GEMINI_API_KEY`.

```python
import asyncio
from trustandverify import verify

report = asyncio.run(verify("Is coffee healthy?", num_claims=3))
```

### `TrustAgent`

Main orchestrator. Wires config, backends, and the pipeline.

```python
from trustandverify import TrustAgent, TrustConfig

agent = TrustAgent(
    config=TrustConfig(),       # optional, defaults to TrustConfig()
    search=search_backend,      # required — any SearchBackend
    llm=llm_backend,            # required — any LLMBackend
    storage=storage_backend,    # optional, defaults to InMemoryStorage
    cache=cache_backend,        # optional, defaults to FileCache if enable_cache=True
)
```

#### `agent.verify(query, verbose=False) → Report`

*Async.* Runs the full 5-stage pipeline: Plan → Search → Extract → Score → Report. Persists the report and individual claims to the storage backend.

Raises `RuntimeError` if `search` or `llm` is `None`.

### `TrustConfig`

All pipeline settings in one dataclass.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_claims` | `int` | `0` | Claims to decompose into (0 = LLM decides, typically 3–5) |
| `max_sources_per_claim` | `int` | `3` | Max search results per claim |
| `base_uncertainty` | `float` | `0.3` | Starting uncertainty per evidence piece |
| `conflict_threshold` | `float` | `0.2` | Conflict score above which a conflict is flagged |
| `default_source_trust` | `float` | `0.5` | Fallback trust for unrecognised domains |
| `cache_ttl` | `int` | `3600` | Cache TTL in seconds |
| `enable_cache` | `bool` | `True` | Use the cache layer |
| `enable_byzantine` | `bool` | `False` | Use Byzantine-resistant fusion |
| `byzantine_threshold` | `float` | `0.15` | Discord score cutoff for Byzantine filtering |
| `byzantine_min_agents` | `int` | `2` | Minimum evidence pieces after Byzantine filtering |
| `export_formats` | `list[str]` | `["jsonld"]` | Default export formats |

---

## Data Models

All models are in `trustandverify.core.models`. `Opinion` is re-exported from `jsonld_ex.confidence_algebra`.

### `Opinion`

Subjective Logic opinion tuple. Imported from jsonld-ex.

| Attribute | Type | Description |
|---|---|---|
| `belief` | `float` | Degree of belief [0, 1] |
| `disbelief` | `float` | Degree of disbelief [0, 1] |
| `uncertainty` | `float` | Degree of uncertainty [0, 1] |
| `base_rate` | `float` | Prior probability (default 0.5) |

**Constraint**: `belief + disbelief + uncertainty = 1.0`

**Method**: `projected_probability() → float` — returns `belief + base_rate * uncertainty`.

### `Verdict`

Enum: `SUPPORTED`, `CONTESTED`, `REFUTED`, `NO_EVIDENCE`.

### `SearchResult`

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Page title |
| `url` | `str` | Source URL |
| `content` | `str` | Page content/snippet |
| `score` | `float` | Relevance score (default 0.0) |

### `Source`

| Field | Type | Description |
|---|---|---|
| `url` | `str` | Source URL |
| `title` | `str` | Page title |
| `content_snippet` | `str` | Truncated content |
| `trust_score` | `float` | Heuristic trust [0, 1] |
| `source_type` | `str` | Default `"web"` |

### `Evidence`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Evidence text (max 300 chars) |
| `supports_claim` | `bool` | True if supporting, False if contradicting |
| `relevance` | `float` | LLM-assessed relevance [0, 1] |
| `confidence_raw` | `float` | LLM-assessed confidence [0, 1] |
| `source` | `Source` | The source this evidence came from |
| `opinion` | `Opinion \| None` | Optional pre-computed opinion |

### `Conflict`

| Field | Type | Description |
|---|---|---|
| `claim_text` | `str` | Claim text (truncated) |
| `conflict_degree` | `float` | Pairwise conflict score |
| `num_supporting` | `int` | Count of supporting evidence |
| `num_contradicting` | `int` | Count of contradicting evidence |

### `Claim`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Claim text |
| `evidence` | `list[Evidence]` | Collected evidence |
| `opinion` | `Opinion \| None` | Fused opinion after scoring |
| `verdict` | `Verdict` | Derived verdict |
| `assessment` | `str` | LLM-written assessment |

### `ReportSummary`

Lightweight listing entry for `list_reports()`.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Report ID |
| `query` | `str` | Original query |
| `created_at` | `datetime` | Creation timestamp |
| `num_claims` | `int` | Number of claims |

### `Report`

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique report ID (UUID) |
| `query` | `str` | Original research question |
| `claims` | `list[Claim]` | Verified claims |
| `conflicts` | `list[Conflict]` | Detected conflicts |
| `summary` | `str` | Executive summary |
| `created_at` | `datetime` | Creation timestamp (UTC) |
| `metadata` | `dict` | Extensible metadata |

---

## Scoring Module

`trustandverify.scoring` — all public functions.

### `scalar_to_opinion(confidence, evidence_weight=1.0) → Opinion`

Map a scalar confidence [0, 1] to an opinion. Base uncertainty = 0.30 (decreases with `evidence_weight`).

### `flip_opinion(op) → Opinion`

Swap belief and disbelief. Use for contradicting evidence before fusion.

### `opinion_summary(op) → dict`

Returns `{belief, disbelief, uncertainty, base_rate, projected_probability, verdict}` with values rounded to 4 decimal places.

### `estimate_source_trust(url, title="") → float`

Domain-based heuristic trust score in [0, 1].

### `apply_trust_discount(opinion, source_trust) → Opinion`

Jøsang's trust discount operator. `trust=1.0` → unchanged; `trust=0.0` → vacuous.

### `fuse_evidence(opinions) → Opinion`

Cumulative fusion of independent opinions. Empty list returns vacuous opinion.

### `fuse_evidence_byzantine(opinions, trust_weights=None, threshold=0.15, min_agents=2) → dict`

Byzantine-resistant fusion. Returns `{fused, filtered, cohesion, surviving_indices, used_byzantine}`.

### `diagnose_byzantine(opinions, threshold=0.15) → dict`

Lightweight diagnostic. Returns `{recommended, num_discordant, reason, cohesion}`.

### `cohesion_score(opinions) → float`

Overall source agreement [0, 1]. Based on mean pairwise opinion distance.

### `opinion_distance(op_a, op_b) → float`

Normalised Euclidean distance on the opinion simplex [0, 1]. Proper metric.

### `detect_conflicts_within_claim(supporting, contradicting, threshold=0.2) → dict | None`

Within-claim conflict between fused supporting and fused contradicting opinions.

### `pairwise_conflict(op_a, op_b) → float`

Jøsang's evidential conflict measure. Not a metric (`pairwise_conflict(A, A) ≠ 0`).

### `build_evidence_opinion(ev) → Opinion`

Convert an Evidence object to a trust-discounted, possibly-flipped Opinion.

### `score_claim(evidence_list, conflict_threshold=0.2, enable_byzantine=False, byzantine_threshold=0.15) → tuple`

Main scoring entry point. Returns `(Opinion, Verdict, conflict_dict | None, meta_dict)`.

---

## Jac Interop

`trustandverify.jac_interop` — synchronous bridge for Jac walkers.

### `jac_verify(query, **kwargs) → dict`

Runs the full pipeline synchronously. Returns a plain dict (not a Report dataclass). Accepts all `TrustConfig` parameters plus `search_backend`, `llm_backend`, `storage_backend`, `db_path`, and `verbose`.

### `jac_export(report_dict, format="jsonld", output_path=None) → str | bytes`

Export a dict from `jac_verify()` to the requested format. Writes to file if `output_path` is provided.

### `jac_configure_agent(**kwargs) → TrustAgent`

Build a TrustAgent from string backend names (`"tavily"`, `"gemini"`, `"sqlite"`, etc.).

---

## Pipeline Stages

`trustandverify.core.pipeline` — individual async stages (used by TrustAgent internally, available for advanced use).

### `plan(query, config, llm) → list[str]`

*Async.* Decompose a query into claim text strings via the LLM.

### `search_for_claim(claim_text, config, search, llm, cache=None) → list[SearchResult]`

*Async.* Generate a search query for a claim and fetch results.

### `extract(claim_text, results, llm, cache=None) → list[Evidence]`

*Async.* Extract structured evidence from search results.

### `score(claim, config) → tuple[Claim, Conflict | None]`

*Sync.* Score a claim using Subjective Logic. Mutates `claim.opinion` and `claim.verdict`.

### `assess(claim, llm, cache=None) → str`

*Async.* Write a 2–3 sentence assessment for a scored claim.

### `summarise(query, claims, llm, cache=None) → str`

*Async.* Write an executive summary for the full report.

### `run_pipeline(query, config, search, llm, cache=None, verbose=False) → Report`

*Async.* Run all 5 stages and return a Report.
