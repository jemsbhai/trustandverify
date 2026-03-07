# Confidence Algebra

This document describes the Subjective Logic mathematics used by trustandverify. All operators are implemented in [`jsonld-ex`](https://pypi.org/project/jsonld-ex/) (v0.7.0+); trustandverify wraps them with application-level helpers in the `scoring/` module.

## Subjective Logic Opinions

A **Subjective Logic opinion** (Jøsang 2016) is a 4-tuple:

```
ω = (b, d, u, a)
```

where:

- **b** (belief): degree of belief that the proposition is true
- **d** (disbelief): degree of belief that the proposition is false
- **u** (uncertainty): degree of uncommitted belief (ignorance)
- **a** (base rate): prior probability assumed when uncertainty is maximal

**Additivity constraint**: `b + d + u = 1` (always enforced).

**Projected probability**: `P(ω) = b + a · u` — the expected probability given current evidence and the prior.

### Why Not Scalars?

Two propositions can both have P = 0.5 but carry very different epistemological weight:

| Scenario | b | d | u | P | Meaning |
|---|---|---|---|---|---|
| Strong evidence, evenly split | 0.45 | 0.45 | 0.10 | 0.50 | Well-studied, genuinely contested |
| No evidence at all | 0.00 | 0.00 | 1.00 | 0.50 | Total ignorance; prior is doing all the work |
| One weak source, slightly positive | 0.25 | 0.15 | 0.60 | 0.55 | Tentatively positive, high uncertainty |

A scalar confidence score of 0.5 conflates all three. The opinion tuple distinguishes them, enabling different downstream decisions.

## Scalar-to-Opinion Mapping

LLMs produce scalar confidence scores. `scalar_to_opinion()` maps these to opinions:

```python
from trustandverify.scoring import scalar_to_opinion

op = scalar_to_opinion(0.85)  # high confidence
# Opinion(b≈0.595, d≈0.105, u=0.300, a=0.5) → P≈0.745
```

The base uncertainty is 0.30 (configurable via `evidence_weight`). Even a fully confident source gets u = 0.30, reflecting that a single web search result is not ground truth. Uncertainty decreases through fusion as independent sources corroborate each other.

## Trust Discount

Not all sources are equally reliable. Jøsang's trust discount operator (§14.3) adjusts an opinion based on source trustworthiness:

```python
from trustandverify.scoring import apply_trust_discount

discounted = apply_trust_discount(opinion, source_trust=0.85)
```

- `trust = 1.0` → opinion unchanged
- `trust = 0.0` → opinion collapses to pure uncertainty (vacuous)
- `trust = 0.5` → belief and disbelief halved, uncertainty absorbs the rest

Trust scores are assigned by `estimate_source_trust()`, a domain-based heuristic:

| Domain | Trust | Rationale |
|---|---|---|
| `.gov`, `.edu` | 0.90 | Government and academic institutions |
| `nature.com`, `pubmed`, `arxiv.org`, `nber.org` | 0.85 | Major research platforms |
| `reuters.com`, `bbc.com`, `nytimes.com` | 0.75 | Established news organisations |
| `wikipedia.org` | 0.60 | Useful but crowd-edited |
| `reddit.com`, `quora.com` | 0.35 | Social/opinion platforms |
| Other | 0.50 | Default |

These are deliberately conservative — even `.gov` is not 1.0 because domain alone cannot guarantee page-level reliability.

## Opinion Flipping

When evidence **contradicts** a claim, the opinion must be flipped (b ↔ d) before fusion. Without flipping, both supporting and contradicting evidence would have high belief, and pairwise conflict would see no disagreement.

```python
from trustandverify.scoring import flip_opinion

flipped = flip_opinion(opinion)
# belief ↔ disbelief; uncertainty and base_rate unchanged
```

## Cumulative Fusion

The core fusion operator for independent sources observing the same proposition (Jøsang 2016, §12.3):

```python
from trustandverify.scoring import fuse_evidence

fused = fuse_evidence([op1, op2, op3])
```

Properties:

- **Uncertainty reduction**: each additional agreeing source reduces u.
- **Associative and commutative**: order doesn't matter.
- **Vacuous neutral element**: fusing with (0, 0, 1, a) yields the other opinion unchanged.
- **Disagreement preservation**: opposing sources balance b against d rather than cancelling.

An empty list returns the vacuous opinion `(0, 0, 1, 0.5)`. A single opinion is returned unchanged.

## Byzantine-Resistant Fusion

When sources may be adversarial or unreliable, standard fusion can be corrupted by outliers. Byzantine-resistant fusion (introduced in jsonld-ex 0.7.0) filters discordant sources before fusing:

```python
from trustandverify.scoring import fuse_evidence_byzantine

result = fuse_evidence_byzantine(
    opinions,
    trust_weights=[0.9, 0.85, 0.35],  # per-source trust
    threshold=0.15,                     # discord cutoff
    min_agents=2,                       # never reduce below 2
)
# result["fused"] — the fused opinion
# result["filtered"] — removed sources with reasons
# result["cohesion"] — overall source agreement [0, 1]
```

Strategy: `combined` — discord × (1 − trust). Sources that are both highly discordant AND lowly trusted are removed first.

### Opt-In Design

Byzantine fusion is **off by default** in trustandverify (`TrustConfig.enable_byzantine=False`). When off, a lightweight diagnostic runs and returns:

- `meta["byzantine_recommended"]` — whether filtering would help
- `meta["cohesion"]` — source agreement score
- `meta["num_discordant"]` — how many sources exceed the threshold

This lets you surface a recommendation in your UI without altering the scoring.

## Cohesion Score

Measures overall agreement among evidence sources, in [0, 1]:

```python
from trustandverify.scoring import cohesion_score

coh = cohesion_score(opinions)  # 1.0 = perfect agreement, 0.0 = maximum discord
```

Based on mean pairwise opinion distance on the belief-disbelief-uncertainty simplex.

## Opinion Distance

Normalised Euclidean distance on the 2-simplex of (b, d, u):

```python
from trustandverify.scoring import opinion_distance

dist = opinion_distance(op_a, op_b)  # 0.0 = identical, 1.0 = maximally far
```

This is a proper metric (satisfies identity, symmetry, and triangle inequality). It replaced `pairwise_conflict` as the primary distance measure because `pairwise_conflict(A, A) = 2·b·d ≠ 0`, violating the identity axiom of a metric space. `pairwise_conflict` is still available for measuring evidential tension (which is a different quantity from distance).

## Within-Claim Conflict Detection

Conflict is detected between the fused supporting opinion and the fused contradicting opinion for a single claim:

```python
from trustandverify.scoring import detect_conflicts_within_claim

conflict = detect_conflicts_within_claim(
    supporting_opinions,
    contradicting_opinions,
    threshold=0.2,
)
# Returns dict with conflict_degree, opinion_distance, counts — or None
```

Cross-claim comparison is meaningless because different claims are about different propositions. Conflict detection is always within a single claim.

## The score_claim() Pipeline

`score_claim()` is the main entry point. For each claim:

1. Convert each Evidence to a trust-discounted, possibly-flipped Opinion via `build_evidence_opinion()`.
2. Fuse all opinions (cumulative or Byzantine).
3. Derive the Verdict from projected probability: `supported` (P ≥ 0.7), `contested` (0.3 < P < 0.7), `refuted` (P ≤ 0.3).
4. Run within-claim conflict detection.
5. Return `(fused_opinion, verdict, conflict_dict_or_None, meta_dict)`.

```python
from trustandverify.scoring import score_claim

opinion, verdict, conflict, meta = score_claim(
    evidence_list,
    conflict_threshold=0.2,
    enable_byzantine=False,
)
```

## References

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.
- Jøsang, A. (2016). §12.3: Cumulative fusion of independent opinions.
- Jøsang, A. (2016). §14.3: Trust discount of functional trust.
- [`jsonld-ex`](https://pypi.org/project/jsonld-ex/) — Subjective Logic implementation with Byzantine fusion, cohesion, and pluggable distance metrics.
