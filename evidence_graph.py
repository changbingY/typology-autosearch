"""
evidence_graph.py — Structured Evidence Knowledge Graph
=========================================================
Replaces the flat list of evidence passages in the original agent.
Tracks claims, their sources, contradictions, and multi-hop inferences
as a proper graph structure.

Key improvements over flat list:
  - Claims are typed (grammar_statement / igt_pattern / inference / absence)
  - Contradiction detection across the whole graph
  - Confidence propagates through inference chains
  - Provides structured summaries that show the LLM the full evidence picture
    rather than only recent passages
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ClaimType(str, Enum):
    GRAMMAR_STATEMENT = "grammar_statement"   # explicit author claim from prose
    IGT_PATTERN       = "igt_pattern"         # derived from quantitative IGT analysis
    ABSENCE_EVIDENCE  = "absence_evidence"    # confirmed absence from IGT
    INFERENCE         = "inference"           # derived by combining other claims
    COUNTER_EVIDENCE  = "counter_evidence"    # evidence against current hypothesis
    AUTHOR_CAVEAT     = "author_caveat"       # author hedges or qualifies a claim


@dataclass
class Claim:
    claim_id: str
    text: str                       # the actual claim text
    claim_type: ClaimType
    source: str                     # e.g. "§4.2 TMA system" or "IGT[PST]: 140 examples"
    confidence: float               # 0.0–1.0 (how strongly does this claim hold?)
    supports_hypothesis: Optional[bool] = None   # True=supports, False=contradicts, None=neutral
    derived_from: list = field(default_factory=list)  # claim_ids this was inferred from
    igt_examples: list = field(default_factory=list)  # IGT example IDs cited
    raw_evidence: str = ""          # the raw text/observation that led to this claim


@dataclass
class Contradiction:
    claim_a_id: str
    claim_b_id: str
    description: str
    resolved: bool = False
    resolution: str = ""


class EvidenceGraph:
    """
    A structured, typed graph of evidence for a single feature.
    """

    def __init__(self, feature_question: str):
        self.feature_question = feature_question
        self.claims: dict[str, Claim]        = {}
        self.contradictions: list[Contradiction] = []
        self._counter = 0

    # ── Adding evidence ───────────────────────────────────────────

    def add_claim(
        self,
        text: str,
        claim_type: ClaimType,
        source: str,
        confidence: float,
        supports_hypothesis: Optional[bool] = None,
        igt_examples: list = None,
        raw_evidence: str = "",
        derived_from: list = None,
    ) -> str:
        """Add a claim and return its ID. Automatically detects contradictions."""
        self._counter += 1
        cid = f"C{self._counter:03d}"
        claim = Claim(
            claim_id=cid,
            text=text,
            claim_type=claim_type,
            source=source,
            confidence=confidence,
            supports_hypothesis=supports_hypothesis,
            derived_from=derived_from or [],
            igt_examples=igt_examples or [],
            raw_evidence=raw_evidence,
        )
        self.claims[cid] = claim
        self._detect_contradictions(claim)
        return cid

    def add_inference(
        self,
        text: str,
        derived_from: list[str],
        confidence: float,
        supports_hypothesis: Optional[bool] = None,
    ) -> str:
        """Add an inference derived from combining existing claims."""
        sources = [self.claims[cid].source for cid in derived_from if cid in self.claims]
        return self.add_claim(
            text=text,
            claim_type=ClaimType.INFERENCE,
            source="inferred from: " + "; ".join(sources),
            confidence=confidence,
            supports_hypothesis=supports_hypothesis,
            derived_from=derived_from,
        )

    # ── Contradiction detection ───────────────────────────────────

    # Minimum confidence for both claims before flagging a contradiction.
    # Low-confidence claims often represent weak hints rather than genuine conflicts,
    # and flagging every supporting+opposing pair floods the graph with false positives
    # (e.g. "prose says marker exists" at 0.6 + "IGT rate 1%" at 0.55 is not a
    # contradiction — it is the normal dual-evidence pattern that requires synthesis).
    CONTRADICTION_MIN_CONFIDENCE: float = 0.60

    def _detect_contradictions(self, new_claim: Claim):
        """Scan existing claims for potential contradictions with the new one."""
        if new_claim.supports_hypothesis is None:
            return
        # Only flag genuine conflicts: both claims must be sufficiently confident
        if new_claim.confidence < self.CONTRADICTION_MIN_CONFIDENCE:
            return
        for cid, existing in self.claims.items():
            if cid == new_claim.claim_id:
                continue
            if existing.supports_hypothesis is None:
                continue
            if existing.confidence < self.CONTRADICTION_MIN_CONFIDENCE:
                continue
            # Direct contradiction: one supports, one contradicts
            if existing.supports_hypothesis != new_claim.supports_hypothesis:
                # Avoid duplicate contradiction entries
                pair = tuple(sorted([cid, new_claim.claim_id]))
                already = any(
                    tuple(sorted([c.claim_a_id, c.claim_b_id])) == pair
                    for c in self.contradictions
                )
                if not already:
                    self.contradictions.append(Contradiction(
                        claim_a_id=cid,
                        claim_b_id=new_claim.claim_id,
                        description=(
                            f"Conflicting evidence: "
                            f"[{cid}] '{existing.text[:80]}' ({existing.source}) "
                            f"vs [{new_claim.claim_id}] '{new_claim.text[:80]}' ({new_claim.source})"
                        ),
                    ))

    # ── Confidence aggregation ─────────────────────────────────────

    def aggregate_confidence(self) -> tuple[str, float, str]:
        """
        Compute overall hypothesis and confidence from the graph.
        Uses log-odds combination with dampening to avoid overconfidence.

        Grammar prose (GRAMMAR_STATEMENT) is weighted higher than IGT counts
        because the IGT corpus may be incomplete — the author's explicit
        description is the authoritative source.

        Returns (hypothesis, confidence, igt_note) where:
          hypothesis : "Yes" / "No" / "Partial" / "Unclear"
          confidence : 0.0–1.0
          igt_note   : non-empty warning string when IGT coverage is
                       absent or too sparse to be relied upon; empty
                       string when IGT evidence is sufficient.
        """
        supporting    = [c for c in self.claims.values() if c.supports_hypothesis is True]
        contradicting = [c for c in self.claims.values() if c.supports_hypothesis is False]

        if not supporting and not contradicting:
            return "Unclear", 0.0, self._igt_coverage_note()

        # Log-odds aggregation with dampening
        def log_odds(p: float) -> float:
            p = max(0.01, min(0.99, p))
            return math.log(p / (1 - p))

        def from_log_odds(lo: float) -> float:
            return 1 / (1 + math.exp(-lo))

        prior_lo   = 0.0   # start at 50/50
        cumulative = prior_lo

        for claim in supporting:
            weight = self._claim_weight(claim)
            cumulative += weight * log_odds(claim.confidence)

        for claim in contradicting:
            weight = self._claim_weight(claim)
            cumulative -= weight * log_odds(claim.confidence)

        # Normalize: dampen only when there is genuinely mixed (opposing) evidence.
        # If all claims point in the same direction, dividing by sqrt(n) would
        # artificially push a well-supported conclusion back toward 0.5.
        # When both sides are present, dampening prevents a flood of weak
        # same-direction claims from swamping one strong contrary claim.
        n_supporting_count    = len(supporting)
        n_contradicting_count = len(contradicting)
        n_total               = n_supporting_count + n_contradicting_count
        if n_contradicting_count > 0 and n_supporting_count > 0 and n_total > 0:
            cumulative /= math.sqrt(n_total)

        raw_conf = from_log_odds(cumulative)

        # Determine hypothesis
        unresolved_contradictions = [c for c in self.contradictions if not c.resolved]
        has_strong_counter = any(c.confidence > 0.7 for c in contradicting)

        if unresolved_contradictions and has_strong_counter:
            hyp, conf = "Partial", min(0.6, raw_conf)
        elif raw_conf >= 0.75:
            hyp, conf = "Yes", raw_conf
        elif raw_conf <= 0.30:
            hyp, conf = "No", 1.0 - raw_conf
        elif 0.45 <= raw_conf <= 0.60:
            hyp, conf = "Unclear", 0.4
        else:
            hyp, conf = "Partial", raw_conf

        return hyp, conf, self._igt_coverage_note()

    def _igt_coverage_note(self) -> str:
        """
        Assess whether IGT evidence is sufficient to be relied upon.

        Returns a warning string in three tiers:
          - absent  : no IGT claims at all
          - limited : IGT claims present but all have confidence < 0.6,
                      or there is only a single IGT claim
          - ""      : IGT coverage is adequate (silent — no warning needed)

        The note is intentionally short so callers can append it to
        human-readable output or inject it into prompts without noise.
        """
        igt_claims = [
            c for c in self.claims.values()
            if c.claim_type in (ClaimType.IGT_PATTERN, ClaimType.ABSENCE_EVIDENCE)
        ]

        if not igt_claims:
            return (
                "IGT coverage: none — conclusion relies solely on grammar prose; "
                "the annotated corpus may be absent or not yet loaded."
            )

        high_conf_igt = [c for c in igt_claims if c.confidence >= 0.6]
        if not high_conf_igt or len(igt_claims) == 1:
            return (
                f"IGT coverage: limited ({len(igt_claims)} claim(s), "
                f"{len(high_conf_igt)} with conf ≥ 0.6) — "
                "corpus may be incomplete; treat quantitative figures with caution."
            )

        return ""  # sufficient — no warning

    def _claim_weight(self, claim: Claim) -> float:
        weights = {
            ClaimType.GRAMMAR_STATEMENT: 1.2,   # author's explicit prose > IGT counts
            ClaimType.IGT_PATTERN:       1.0,   # quantitative, but corpus may be incomplete
            ClaimType.ABSENCE_EVIDENCE:  1.1,
            ClaimType.INFERENCE:         0.7,   # derived claims get less weight
            ClaimType.COUNTER_EVIDENCE:  1.0,
            ClaimType.AUTHOR_CAVEAT:     0.5,   # hedged claims get less weight
        }
        return weights.get(claim.claim_type, 0.8)

    # ── Summaries for LLM ─────────────────────────────────────────

    def summarize(self) -> str:
        """Full evidence picture, structured for LLM consumption."""
        hypothesis, confidence, igt_note = self.aggregate_confidence()
        lines = [
            f"EVIDENCE GRAPH: {self.feature_question}",
            f"Current aggregate: {hypothesis} (confidence={confidence:.2f})",
        ]
        if igt_note:
            lines.append(f"⚠ {igt_note}")
        lines += [
            f"Claims: {len(self.claims)} total, "
            f"{sum(1 for c in self.claims.values() if c.supports_hypothesis is True)} supporting, "
            f"{sum(1 for c in self.claims.values() if c.supports_hypothesis is False)} contradicting, "
            f"{sum(1 for c in self.claims.values() if c.supports_hypothesis is None)} neutral",
            f"Unresolved contradictions: {sum(1 for c in self.contradictions if not c.resolved)}",
            "",
        ]

        # Claim types where raw prose text adds meaningful context for the LLM
        PROSE_TYPES = {ClaimType.GRAMMAR_STATEMENT, ClaimType.COUNTER_EVIDENCE, ClaimType.AUTHOR_CAVEAT}
        # IGT types — show the tool output so the conclusion LLM has position stats,
        # co-occurrents etc. (without this, those details are completely invisible)
        IGT_TYPES   = {ClaimType.IGT_PATTERN, ClaimType.ABSENCE_EVIDENCE}

        def fmt_claim(claim):
            type_label = claim.claim_type.value.upper()

            raw_str = ""
            if claim.claim_type in PROSE_TYPES and claim.raw_evidence:
                snippet = claim.raw_evidence[:200].replace("\n", " ").strip()
                if len(claim.raw_evidence) > 200:
                    snippet += "..."
                raw_str = f"\n    Grammar text: \"{snippet}\""
            elif claim.claim_type in IGT_TYPES and claim.raw_evidence:
                # Show up to 500 chars of the tool output so the conclusion LLM
                # can see frequency, position, co-occurrents, construction patterns.
                snippet = claim.raw_evidence[:500].replace("\n", " | ").strip()
                if len(claim.raw_evidence) > 500:
                    snippet += "..."
                raw_str = f"\n    Tool output: {snippet}"

            # For IGT claims, show example IDs and count
            igt_str = ""
            if claim.igt_examples:
                ex_ids = [e for e in claim.igt_examples if isinstance(e, str) and e.startswith("ex:")]
                chunk_ids = [e for e in claim.igt_examples if isinstance(e, str) and e.startswith("chunk_")]
                if ex_ids:
                    igt_str = f"\n    IGT examples ({len(ex_ids)}): {ex_ids[:3]}"
                if chunk_ids:
                    igt_str += f"\n    Chunks: {chunk_ids[:4]}"

            return (
                f"  [claim_id={claim.claim_id}] [{type_label}] conf={claim.confidence:.2f}\n"
                f"    Claim: {claim.text[:200]}\n"
                f"    Source: {claim.source}"
                f"{raw_str}{igt_str}"
            )

        if any(c.supports_hypothesis is True for c in self.claims.values()):
            lines.append("SUPPORTING EVIDENCE:")
            for claim in self.claims.values():
                if claim.supports_hypothesis is True:
                    lines.append(fmt_claim(claim))

        if any(c.supports_hypothesis is False for c in self.claims.values()):
            lines.append("\nCONTRADICTING EVIDENCE:")
            for claim in self.claims.values():
                if claim.supports_hypothesis is False:
                    lines.append(fmt_claim(claim))

        if any(c.supports_hypothesis is None for c in self.claims.values()):
            lines.append("\nNEUTRAL/CONTEXTUAL EVIDENCE:")
            for claim in self.claims.values():
                if claim.supports_hypothesis is None:
                    raw_str = ""
                    if claim.claim_type in PROSE_TYPES and claim.raw_evidence:
                        snippet = claim.raw_evidence[:120].replace("\n", " ").strip()
                        raw_str = f" | text: \"{snippet}...\""
                    igt_str = f" IGT: {[e for e in claim.igt_examples if isinstance(e,str) and e.startswith('ex:')][:2]}" if claim.igt_examples else ""
                    lines.append(
                        f"  [claim_id={claim.claim_id}] {claim.text[:120]}"
                        f"  Source: {claim.source}{raw_str}{igt_str}"
                    )

        if self.contradictions:
            lines.append("\nCONTRADICTIONS DETECTED:")
            for contra in self.contradictions:
                status = "RESOLVED" if contra.resolved else "UNRESOLVED"
                lines.append(f"  [{status}] {contra.description}")
                if contra.resolved:
                    lines.append(f"    Resolution: {contra.resolution}")

        return "\n".join(lines)

    def get_gap_analysis(self, has_igt: bool = True) -> str:
        """
        What evidence is still missing?
        Guides the agent on what to search for next.

        has_igt=False: IGT corpus not loaded; suppress all IGT-related gaps
        so the agent does not waste iterations on unavailable tools.
        """
        has_grammar_statement = any(
            c.claim_type == ClaimType.GRAMMAR_STATEMENT
            for c in self.claims.values()
        )
        has_igt_pattern = any(
            c.claim_type == ClaimType.IGT_PATTERN
            for c in self.claims.values()
        )
        has_absence = any(
            c.claim_type == ClaimType.ABSENCE_EVIDENCE
            for c in self.claims.values()
        )
        has_counter = any(
            c.supports_hypothesis is False
            for c in self.claims.values()
        )
        unresolved = [c for c in self.contradictions if not c.resolved]

        gaps = []
        if not has_grammar_statement:
            gaps.append("NO GRAMMAR PROSE EVIDENCE: Search for the author's explicit description of this category")
        if has_igt and not has_igt_pattern:
            gaps.append("NO IGT QUANTITATIVE EVIDENCE: Run tag analysis to ground claims in data")
        if has_igt and not has_absence:
            gaps.append("NO ABSENCE CHECK: Verify whether absence of the feature is confirmed by IGT")
        if not has_counter:
            gaps.append("NO COUNTER-EVIDENCE SEARCH: A genuine search for disconfirming evidence has not been done")
        if unresolved:
            gaps.append(
                f"UNRESOLVED CONTRADICTIONS ({len(unresolved)}): "
                + "; ".join(c.description[:80] for c in unresolved[:2])
            )

        if not gaps:
            if has_igt:
                return "Evidence is comprehensive: grammar prose, IGT pattern, absence check, and counter-evidence all present."
            else:
                return "Evidence is comprehensive: grammar prose and counter-evidence present (no IGT corpus loaded)."
        return "EVIDENCE GAPS:\n" + "\n".join(f"  - {g}" for g in gaps)

    def get_igt_example_ids(self) -> list:
        """
        Collect all real IGT example IDs cited in any claim.
        Excludes chunk IDs (used for prose section references) which are stored
        in the same igt_examples field for mechanical reasons.
        Returns plain ID strings; full data is looked up by the caller via the toolkit.
        """
        ids = []
        for claim in self.claims.values():
            for eid in claim.igt_examples:
                # chunk_ids (chunk_XXXX) are section references, not IGT examples
                if isinstance(eid, str) and not eid.startswith("chunk_") and eid not in ids:
                    ids.append(eid)
        return ids

    def get_igt_example_notes(self) -> dict:
        """
        Build an annotation dict for every IGT example cited in this graph.

        Returns: {example_id: {"note": str, "source": str, "claim_type": str}}
          note       — the claim text that cited this example (why it was selected)
          source     — the tool call or section that produced the claim
          claim_type — e.g. "igt_pattern", "grammar_statement"

        If multiple claims cite the same example, the highest-confidence one wins.
        """
        best: dict = {}   # example_id → (confidence, claim)
        for claim in self.claims.values():
            for eid in claim.igt_examples:
                if not (isinstance(eid, str) and not eid.startswith("chunk_")):
                    continue
                if eid not in best or claim.confidence > best[eid][0]:
                    best[eid] = (claim.confidence, claim)

        notes = {}
        for eid, (_, claim) in best.items():
            notes[eid] = {
                "note":       claim.text[:120],
                "source":     claim.source,
                "claim_type": claim.claim_type.value,
            }
        return notes

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        hyp, conf, igt_note = self.aggregate_confidence()
        return {
            "feature": self.feature_question,
            "aggregate_hypothesis": hyp,
            "aggregate_confidence": round(conf, 3),
            "igt_coverage_note": igt_note,
            "n_claims": len(self.claims),
            "n_contradictions": len(self.contradictions),
            "unresolved_contradictions": sum(1 for c in self.contradictions if not c.resolved),
            "claims": [
                {
                    "id": c.claim_id,
                    "type": c.claim_type.value,
                    "text": c.text,
                    "source": c.source,
                    "confidence": round(c.confidence, 3),
                    "supports": c.supports_hypothesis,
                    "igt_examples": c.igt_examples,
                }
                for c in self.claims.values()
            ],
            "contradictions": [
                {
                    "claims": [c.claim_a_id, c.claim_b_id],
                    "description": c.description,
                    "resolved": c.resolved,
                    "resolution": c.resolution,
                }
                for c in self.contradictions
            ],
        }