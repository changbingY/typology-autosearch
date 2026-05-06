"""
feature_synthesizer.py — Per-Question Cross-Linguistic Feature Synthesizer
===========================================================================
After all languages have answered the same question, this module synthesizes
their answers into a single structured FeatureEntry:

  Feature name  : canonical typological name (e.g. "Order of Subject, Object and Verb")
  Definition    : precise language-neutral definition
  Types         : the types attested across the sample (emerged from data, not pre-specified)
  Per type      : description + which languages + how each language realizes it + evidence

Called from multi_main.py once per question, immediately after all languages
have run their ReAct search for that question.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Output data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class LanguageRealization:
    """How one specific language realizes a feature type."""
    language:    str
    realization: str    # 1-2 sentence description of realization in this language
    evidence:    str    # key evidence passage (cited from QueryResult)
    confidence:  float
    raw_answer:  str    # first 200 chars of the full prose answer


@dataclass
class FeatureType:
    """One attested type of a cross-linguistic feature."""
    type_label:   str               # e.g. "SOV", "Yes", "Suffixing", "Ergative"
    description:  str               # what this type means linguistically
    languages:    list[LanguageRealization] = field(default_factory=list)

    def language_names(self) -> list[str]:
        return [lr.language for lr in self.languages]


@dataclass
class FeatureEntry:
    """
    The full synthesized cross-linguistic entry for one question/feature.
    This is the primary output unit of the system.
    """
    question:               str         # original research question
    feature_name:           str         # canonical typological name
    definition:             str         # language-neutral definition
    types:                  list[FeatureType] = field(default_factory=list)
    cross_linguistic_notes: str = ""    # patterns, dominance, implications
    typological_significance: str = ""  # relation to known universals (Greenberg, WALS, etc.)
    languages_covered:      list[str] = field(default_factory=list)
    raw_llm_output:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question":               self.question,
            "feature_name":           self.feature_name,
            "definition":             self.definition,
            "languages_covered":      self.languages_covered,
            "n_languages":            len(self.languages_covered),
            "types": [
                {
                    "type_label":  t.type_label,
                    "description": t.description,
                    "n_languages": len(t.languages),
                    "languages": [
                        {
                            "language":    lr.language,
                            "realization": lr.realization,
                            "evidence":    lr.evidence,
                            "confidence":  lr.confidence,
                        }
                        for lr in t.languages
                    ],
                }
                for t in self.types
            ],
            "cross_linguistic_notes":     self.cross_linguistic_notes,
            "typological_significance":   self.typological_significance,
        }

    def to_markdown(self, question_number: int = 0) -> str:
        """Render this feature entry as a Markdown section."""
        num_prefix = f"{question_number}. " if question_number else ""
        lines = [
            f"## {num_prefix}{self.feature_name}",
            "",
            f"**Research question:** _{self.question}_",
            "",
            f"**Definition:** {self.definition}",
            "",
            f"**Languages analyzed ({len(self.languages_covered)}):** "
            + ", ".join(self.languages_covered),
            "",
        ]

        # Value distribution summary
        dist_parts = [
            f"{t.type_label} ({len(t.languages)})"
            for t in sorted(self.types, key=lambda t: -len(t.languages))
        ]
        if dist_parts:
            lines += [
                f"**Type distribution:** " + " · ".join(dist_parts),
                "",
            ]

        # Each type
        for t in sorted(self.types, key=lambda t: -len(t.languages)):
            pct = int(100 * len(t.languages) / max(len(self.languages_covered), 1))
            lines += [
                f"### Type: {t.type_label} "
                f"({len(t.languages)}/{len(self.languages_covered)} languages, {pct}%)",
                "",
                f"_{t.description}_",
                "",
            ]
            for lr in t.languages:
                lines += [
                    f"**{lr.language}** (confidence {lr.confidence:.0%})",
                    f"> {lr.realization}",
                    "",
                ]
                if lr.evidence:
                    # Show evidence as blockquote, truncated
                    ev = lr.evidence.strip()[:400].replace("\n", " ")
                    lines += [f"> *Evidence:* {ev}", ""]

        # Notes
        if self.cross_linguistic_notes:
            lines += ["**Cross-linguistic patterns:**", "", self.cross_linguistic_notes, ""]
        if self.typological_significance:
            lines += ["**Typological significance:**", "", self.typological_significance, ""]

        lines.append("---")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Synthesis prompt
# ════════════════════════════════════════════════════════════════

FEATURE_SYNTHESIS_PROMPT = """\
You are a linguistic typologist. You have collected answers from {n_languages} languages
about the following research question:

RESEARCH QUESTION: {question}

════════════════════════════════════════════════════════════════
LANGUAGE ANSWERS
════════════════════════════════════════════════════════════════

{language_answers_block}

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════

Synthesize these language-specific answers into ONE structured feature entry.

1. FEATURE NAME
   Give the canonical typological name for this feature — the standard term used
   in typological databases (WALS, Grambank, WOLD). Examples:
     "Order of Subject, Object and Verb"
     "Grammatical Tense"
     "Negative Morpheme Position"
     "Presence of Evidentiality"
   Do NOT use the question wording. Use the standard typological label.

2. DEFINITION
   Write a precise, language-neutral definition (2-3 sentences) of what this
   feature measures. Focus on the structural/grammatical property, not on any
   one language.

3. TYPES (CRITICAL RULE)
   Identify the distinct types based ONLY on what is actually attested in this
   sample. Do NOT list possible types that appear in no language.
   - Each type gets a short label (e.g. "SOV", "Yes", "Prefixing", "Ergative-Absolutive")
   - Each type gets a description of what that value means linguistically

4. LANGUAGE REALIZATIONS
   For each type, list every language with that type and describe:
   (a) realization: how the feature is specifically realized in that language
       (morphological form, position, interaction with other features, etc.)
       Write 1-2 sentences grounded in the evidence.
   (b) evidence: COPY the most specific piece of evidence from the language's
       answer below (a grammar section citation or IGT observation). Do not
       invent or paraphrase — quote or closely summarize the actual evidence.
   (c) confidence: use the confidence score from the language's answer.

5. CROSS-LINGUISTIC NOTES
   1-2 sentences on patterns: is one type dominant? Are there co-occurrences
   or implications with other features? Note if any language is anomalous.

6. TYPOLOGICAL SIGNIFICANCE
   1-2 sentences connecting the findings to known typological generalizations
   (e.g., Greenberg's universals, WALS tendencies, known areal patterns).
   Only write this if you can make a specific, accurate claim.

Output ONLY valid JSON (no markdown, no extra text):
{{
  "feature_name": "Order of Subject, Object and Verb",
  "definition": "The canonical linear order of Subject (S), Object (O), and Verb (V) in transitive declarative main clauses. It is one of the most studied variables in cross-linguistic typology.",
  "types": [
    {{
      "type_label": "SOV",
      "description": "The verb is the final element; both subject and object precede it. This is the most common word order cross-linguistically.",
      "language_realizations": [
        {{
          "language": "Aguaruna",
          "realization": "SOV order is the strong default in main clauses and obligatory in subordinate clauses; postverbal constituents are marked as afterthoughts.",
          "evidence": "Grammar §4.2: 'The verb consistently follows both its subject and object in declarative main clauses; in subordinate clauses this order is strictly obligatory.'",
          "confidence": 0.95
        }}
      ]
    }},
    {{
      "type_label": "VSO",
      "description": "The verb appears clause-initially, before both subject and object.",
      "language_realizations": [
        {{
          "language": "Yagua",
          "realization": "The verb is the first constituent in most transitive clauses, with subject and object following.",
          "evidence": "IGT analysis: 78% of transitive clauses show V in initial position (analyse_tag: V positional profile mean_position=0.08).",
          "confidence": 0.82
        }}
      ]
    }}
  ],
  "cross_linguistic_notes": "SOV is the majority type in this sample (2/3 languages). The one VSO language (Yagua) shows flexible order in marked contexts, suggesting VSO may be surface-level.",
  "typological_significance": "SOV is the most common word order globally (~44% of languages; Dryer 2013, WALS Chapter 81). The sample's SOV preference is consistent with Amazonian areal tendencies."
}}"""


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _safe_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _format_query_result(name: str, qr: dict) -> str:
    """Format one language's QueryResult dict into the prompt block."""
    answer      = qr.get("answer", "").strip()
    struct_desc = qr.get("structural_description", "").strip()
    confidence  = qr.get("confidence", 0.0)
    audit       = qr.get("audit_verdict", "")

    # Top 2 evidence passages
    evidence_passages = qr.get("key_evidence", [])[:2]
    evidence_str = ""
    for ev in evidence_passages:
        if isinstance(ev, str):
            evidence_str += f"\n  Evidence: {ev[:300]}"
        elif isinstance(ev, dict):
            evidence_str += f"\n  Evidence: {str(ev)[:300]}"

    # Top 2 IGT examples
    igt_examples = qr.get("igt_examples_used", [])[:2]
    igt_str = ""
    for ex in igt_examples:
        if isinstance(ex, dict):
            gloss = ex.get("gloss", "")
            trans = ex.get("translation", "")
            if gloss or trans:
                igt_str += f"\n  IGT: {gloss}  '{trans}'"

    block = (
        f"── {name.upper()} (confidence={confidence:.2f}, audit={audit}) ──\n"
        f"Answer summary: {answer[:400]}\n"
    )
    if struct_desc:
        block += f"Structural description: {struct_desc[:250]}\n"
    block += evidence_str + igt_str
    return block


# ════════════════════════════════════════════════════════════════
# Main synthesizer class
# ════════════════════════════════════════════════════════════════

class FeatureSynthesizer:
    """
    Given a question and the raw QueryResult from each language, calls the LLM
    to produce a structured FeatureEntry with definition, types, and
    per-language realizations.
    """

    def __init__(self, llm):
        self.llm = llm

    def synthesize(
        self,
        question:     str,
        lang_results: dict[str, dict],   # {language_name: QueryResult.to_dict()}
    ) -> FeatureEntry:
        """
        Synthesize all language answers for one question into a FeatureEntry.

        lang_results : dict mapping language name → QueryResult.to_dict()
        """
        if not lang_results:
            return self._empty_entry(question)

        # Build the language answers block for the prompt
        blocks = []
        for name, qr in lang_results.items():
            blocks.append(_format_query_result(name, qr))
        language_answers_block = "\n\n".join(blocks)

        prompt = FEATURE_SYNTHESIS_PROMPT.format(
            n_languages=len(lang_results),
            question=question,
            language_answers_block=language_answers_block,
        )

        raw    = self.llm.generate(prompt, max_new_tokens=2048, json_mode=True)
        parsed = _safe_json(raw)

        if not parsed:
            logger.warning(f"LLM synthesis failed for '{question}'; using fallback.")
            return self._fallback_entry(question, lang_results)

        return self._build_entry(question, lang_results, parsed)

    def _build_entry(
        self,
        question:     str,
        lang_results: dict[str, dict],
        parsed:       dict,
    ) -> FeatureEntry:
        """Build a FeatureEntry from the parsed LLM JSON output."""
        types: list[FeatureType] = []

        for t_data in parsed.get("types", []):
            realizations: list[LanguageRealization] = []

            for lr_data in t_data.get("language_realizations", []):
                lang_name   = lr_data.get("language", "")
                raw_answer  = lang_results.get(lang_name, {}).get("answer", "")[:200]
                realizations.append(LanguageRealization(
                    language=lang_name,
                    realization=lr_data.get("realization", ""),
                    evidence=lr_data.get("evidence", ""),
                    confidence=float(lr_data.get("confidence", 0.0)),
                    raw_answer=raw_answer,
                ))

            types.append(FeatureType(
                type_label=t_data.get("type_label", "?"),
                description=t_data.get("description", ""),
                languages=realizations,
            ))

        # Collect all language names that appear in any type
        accounted = {lr.language for t in types for lr in t.languages}

        # Any language whose result was not placed in a type gets an "Unclear" bucket
        unaccounted = [n for n in lang_results if n not in accounted]
        if unaccounted:
            unclear_realizations = []
            for name in unaccounted:
                qr = lang_results[name]
                unclear_realizations.append(LanguageRealization(
                    language=name,
                    realization=qr.get("answer", "")[:150],
                    evidence="",
                    confidence=float(qr.get("confidence", 0.0)),
                    raw_answer=qr.get("answer", "")[:200],
                ))
            types.append(FeatureType(
                type_label="Unclear",
                description="Insufficient evidence to classify.",
                languages=unclear_realizations,
            ))

        return FeatureEntry(
            question=question,
            feature_name=parsed.get("feature_name", question),
            definition=parsed.get("definition", ""),
            types=types,
            cross_linguistic_notes=parsed.get("cross_linguistic_notes", ""),
            typological_significance=parsed.get("typological_significance", ""),
            languages_covered=list(lang_results.keys()),
            raw_llm_output=parsed,
        )

    def _fallback_entry(
        self,
        question:     str,
        lang_results: dict[str, dict],
    ) -> FeatureEntry:
        """Rule-based fallback when LLM synthesis fails."""
        # Group languages by their reported value
        by_value: dict[str, list] = {}
        for name, qr in lang_results.items():
            val = qr.get("value", qr.get("answer", "Unclear")[:30])
            by_value.setdefault(val, []).append((name, qr))

        types = []
        for val, lang_qr_pairs in by_value.items():
            realizations = [
                LanguageRealization(
                    language=name,
                    realization=qr.get("structural_description", qr.get("answer", ""))[:150],
                    evidence=(qr.get("key_evidence", [""])[0] if qr.get("key_evidence") else ""),
                    confidence=float(qr.get("confidence", 0.0)),
                    raw_answer=qr.get("answer", "")[:200],
                )
                for name, qr in lang_qr_pairs
            ]
            types.append(FeatureType(type_label=val, description="", languages=realizations))

        return FeatureEntry(
            question=question,
            feature_name=question,
            definition="(LLM synthesis failed — raw values shown)",
            types=types,
            languages_covered=list(lang_results.keys()),
        )

    def _empty_entry(self, question: str) -> FeatureEntry:
        return FeatureEntry(
            question=question,
            feature_name=question,
            definition="No language data available.",
            types=[],
            languages_covered=[],
        )


# ════════════════════════════════════════════════════════════════
# Report generator
# ════════════════════════════════════════════════════════════════

def build_report(
    entries:        list[FeatureEntry],
    language_names: list[str],
) -> str:
    """Render all FeatureEntries as a single Markdown report."""
    lines = [
        "# Cross-Linguistic Typological Report",
        "",
        f"**Languages ({len(language_names)}):** " + ", ".join(language_names),
        f"**Features analyzed:** {len(entries)}",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(entries, 1):
        lines.append(entry.to_markdown(question_number=i))
        lines.append("")

    # Summary table
    lines += [
        "---",
        "",
        "## Summary Table",
        "",
        "| # | Feature | Types (n languages each) | Languages |",
        "|---|---------|--------------------------|-----------|",
    ]
    for i, entry in enumerate(entries, 1):
        type_summary = " · ".join(
            f"{t.type_label} ({len(t.languages)})"
            for t in sorted(entry.types, key=lambda t: -len(t.languages))
        )
        langs = ", ".join(entry.languages_covered)
        lines.append(f"| {i} | {entry.feature_name} | {type_summary} | {langs} |")

    lines += ["", "---", "_Generated by Typology Autosearch_"]
    return "\n".join(lines)
