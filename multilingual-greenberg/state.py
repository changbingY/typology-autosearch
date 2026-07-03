"""
Data structures for the deep search agent's epistemic state.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Feature:
    feature_id: str
    question: str
    domain: str
    linguistic_definition: str       # precise definition of the feature as a linguistic category
    structural_description: str      # how it is realised: forms, markers, positions
    value: str                       # Yes / No / Partial / Unclear / ?
    value_detail: str                # precise one-sentence elaboration
    confidence: float                # 0.0 – 1.0 (post-audit)
    key_evidence: list               # evidence paragraph strings
    igt_examples_used: list          # [{"example_id": ..., "source": ..., "gloss": ..., "translation": ...}]
    igt_support: bool                # whether IGT was consulted
    search_trace: list               # full ReAct trace
    typological_notes: str           # cross-linguistic significance
    needs_human_review: bool
    review_reason: str
    audit_verdict: str               # "upheld" | "weakened" | "overturned"
    audit_objections: list           # objections raised by auditor
    token_usage: dict                # {"input_tokens", "output_tokens", "total_tokens", "llm_calls"}

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class QueryResult:
    """Structured result for a free-form deep query — same evidence rigour as Feature."""
    query_id: str
    query: str
    phenomena: list              # linguistic phenomena the query was about
    linguistic_definition: str   # language-agnostic definition of the queried phenomenon
    structural_description: str  # how the phenomenon is realised in this language
    answer: str                  # main prose answer to the query
    key_evidence: list           # cited evidence paragraphs (same format as Feature)
    igt_examples_used: list      # [{"example_id", "source", "morpheme", "gloss", "translation"}]
    igt_support: bool
    search_trace: list
    confidence: float
    needs_human_review: bool
    review_reason: str
    audit_verdict: str           # "upheld" | "weakened" | "overturned"
    audit_objections: list
    token_usage: dict            # {"input_tokens", "output_tokens", "total_tokens", "llm_calls"}

    # ── Grambank coding fields (populated only when query includes a
    #    GRAMBANK CODING REQUIREMENT section; empty strings otherwise) ──
    grambank_label: str = ""         # "0" | "1" | "?" | ""
    grambank_reasoning: str = ""     # one-sentence explanation of the label

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class EpistemicState:
    language: str
    domains: list = field(default_factory=list)
    confirmed_features: list = field(default_factory=list)
    uncertain_features: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)

    def has_open_questions(self, domain: dict) -> bool:
        investigated = {
            f.question
            for f in self.confirmed_features + self.uncertain_features
        }
        return any(
            c["question"] not in investigated
            for c in domain.get("candidate_features", [])
        )

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "summary": {
                "domains": len(self.domains),
                "confirmed_features": len(self.confirmed_features),
                "uncertain_features": len(self.uncertain_features),
                "features_needing_review": sum(
                    1 for f in self.confirmed_features if f.needs_human_review
                ),
                "audit_overturned": sum(
                    1 for f in self.confirmed_features
                    if f.audit_verdict == "overturned"
                ),
                "igt_supported": sum(
                    1 for f in self.confirmed_features if f.igt_support
                ),
            },
            "confirmed_features": [f.to_dict() for f in self.confirmed_features],
            "uncertain_features": [f.to_dict() for f in self.uncertain_features],
            "contradictions": self.contradictions,
        }