"""
grambank_labeler.py — Grambank Feature Labeler
===============================================
Applies Grambank coding policies to evidence produced by
DeepLanguageResearchAgent.answer_query(), yielding a structured label.

Design goals
------------
  - Always produce a definitive label from the model's own output.
  - All outputs are fully traceable: label, reasoning, confidence, and
    which stage produced the final label are stored together.

Standalone usage
----------------
  from grambank_labeler import GrambankLabeler, load_grambank_csv
  features = load_grambank_csv("grambank.csv")
  labeler  = GrambankLabeler(llm)
  label    = labeler.label(query_result, features["GB020"])
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GrambankFeature:
    """One row from the Grambank CSV."""
    grambank_id:    str          # e.g. "GB020"
    query:          str          # the research question
    coding_policy:  str          # full coding rules text
    definition:     str          # linguistic definition of the feature
    id_desc:        str          # e.g. "GB020 ARTDef"


@dataclass
class LabelResult:
    """Output of GrambankLabeler.label()."""
    grambank_id:    str
    query:          str
    label:          str           # e.g. "A", "B", "C"
    reasoning:      str           # one-paragraph justification
    confidence:     float         # 0.0–1.0
    stage:          int           # always 1; kept for downstream compatibility
    answer_summary: str           # first 200 chars of the underlying QueryResult.answer
    token_usage:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ═══════════════════════════════════════════════════════════════════
# CSV loader
# ═══════════════════════════════════════════════════════════════════

def load_grambank_csv(path: str | Path) -> dict[str, GrambankFeature]:
    """
    Load the Grambank CSV and return a dict keyed by grambank_id (e.g. "GB020").
    """
    path = Path(path)
    features: dict[str, GrambankFeature] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = row["ID"].strip()
            if not gid:
                continue
            feat = GrambankFeature(
                grambank_id   = gid,
                query         = row.get("Query", "").strip(),
                coding_policy = row.get("Coding", "").strip(),
                definition    = row.get("Definition", "").strip(),
                id_desc       = row.get("Grambank_ID_desc", "").strip(),
            )
            features[gid] = feat
    logger.info(f"Loaded {len(features)} Grambank features from {path}")
    return features


# ═══════════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════════

_PRIMARY_PROMPT = """\
You are a typology expert and field annotator applying a coding scheme to a reference grammar.

══════════════════════════════════════════════════════
FEATURE: {grambank_id}  —  {query}
══════════════════════════════════════════════════════

LINGUISTIC DEFINITION
{definition}

CODING POLICY (apply this EXACTLY)
{coding_policy}

══════════════════════════════════════════════════════
EVIDENCE FROM GRAMMAR + IGT CORPUS (language: {language})
══════════════════════════════════════════════════════

ANSWER SUMMARY:
{answer}

KEY EVIDENCE ITEMS:
{key_evidence}

STRUCTURAL DESCRIPTION OF THE PHENOMENON IN {language_upper}:
{structural_description}

IGT EXAMPLES CITED:
{igt_examples}

══════════════════════════════════════════════════════
TASK
══════════════════════════════════════════════════════

Step 1 — Read the evidence carefully.
  For each piece of evidence, note what grammatical behaviour it demonstrates. You need to make judgement because not all evidence are relevent or precise.

Step 2 — Match evidence to coding criteria.
  Go through every criterion in the Coding Policy and check which one the
  evidence most directly satisfies.

Step 3 — Select the label defined in the Coding Policy whose criterion is
  best supported by the evidence above.

IMPORTANT RULES:
  - The label MUST be one of the codes defined in the Coding Policy — copy it
    exactly as written there (e.g. 0, 1, 2, 3, ? ...).
  - Base your decision on what the evidence SHOWS, not on what is absent.
  - Use "?" when the evidence is genuinely insufficient to distinguish
    between any of the substantive codes.
  - If two codes are equally plausible, prefer the one whose criterion is most
    directly demonstrated by a concrete IGT example or explicit grammar statement.
  - No extra text, no explanation outside the JSON.

Output ONLY valid JSON, no prose outside it:
{{
  "label": "<code from Coding Policy>",
  "reasoning": "Two to four sentences citing specific evidence from the answer summary, key evidence items, or IGT examples that justify this code under the coding policy.",
  "confidence": <0.0–1.0>
}}"""


# ═══════════════════════════════════════════════════════════════════
# Labeler
# ═══════════════════════════════════════════════════════════════════

class GrambankLabeler:
    """
    Applies Grambank coding policies to QueryResult objects using an LLM.

    Parameters
    ----------
    llm : QwenLLM (or any object with a .generate(prompt, max_new_tokens) method)
        The same LLM instance used by the research agent; reusing it avoids
        loading a second model.
    """

    def __init__(self, llm):
        self.llm = llm

    # ── Public API ────────────────────────────────────────────────

    def label(
        self,
        query_result,
        feature: GrambankFeature,
        language: str = "",
    ) -> LabelResult:
        """
        Apply a Grambank coding policy to a QueryResult and return a LabelResult.
        The label is whatever single uppercase letter the model produces.
        """
        lang = language or "the language"

        answer      = (query_result.answer or "").strip()
        key_ev_list = query_result.key_evidence or []
        key_ev      = "\n".join(f"  • {ev}" for ev in key_ev_list[:4]) or "  (none)"
        struct_desc = (query_result.structural_description or "").strip() or "(not described)"

        igt_lines = []
        for ex in (query_result.igt_examples_used or [])[:4]:
            tid = ex.get("example_id", "?")
            tr  = ex.get("translation", "")[:70]
            gl  = ex.get("gloss", "")[:70]
            mo  = ex.get("morpheme", "")[:70]
            parts = [f"  [{tid}] '{tr}'"]
            if mo:
                parts.append(f"    morphemes: {mo}")
            if gl:
                parts.append(f"    gloss:     {gl}")
            igt_lines.append("\n".join(parts))
        igt_block = "\n".join(igt_lines) if igt_lines else "  (no IGT examples cited)"

        prompt = _PRIMARY_PROMPT.format(
            grambank_id            = feature.grambank_id,
            query                  = feature.query,
            definition             = feature.definition[:1200],
            coding_policy          = feature.coding_policy[:2000],
            language               = lang,
            language_upper         = lang.upper(),
            answer                 = answer[:1500],
            key_evidence           = key_ev[:1500],
            structural_description = struct_desc[:400],
            igt_examples           = igt_block[:600],
        )

        raw  = self.llm.generate(prompt, max_new_tokens=512)
        out  = self._parse_json(raw)
        label      = str(out.get("label", "")).strip()
        confidence = float(out.get("confidence", 0.5))
        reasoning  = out.get("reasoning", "")

        return LabelResult(
            grambank_id    = feature.grambank_id,
            query          = feature.query,
            label          = label,
            reasoning      = reasoning,
            confidence     = confidence,
            stage          = 1,
            answer_summary = answer[:200],
        )

    def label_batch(
        self,
        query_results: list,
        language: str = "",
    ) -> list[LabelResult]:
        """Label a batch of (QueryResult, GrambankFeature) pairs."""
        results = []
        for i, (qr, feat) in enumerate(query_results, 1):
            logger.info(f"Labeling {i}/{len(query_results)}: {feat.grambank_id}")
            result = self.label(qr, feat, language=language)
            results.append(result)
            _label_print(
                f"[{feat.grambank_id}] label={result.label}  "
                f"conf={result.confidence:.2f}  "
                f"{feat.query[:55]}"
            )
        return results

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse the first JSON object found in the LLM output."""
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
            else:
                return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for i in range(len(text) - 1, max(len(text) - 300, 0), -1):
                if text[i] == "}":
                    try:
                        return json.loads(text[: i + 1])
                    except json.JSONDecodeError:
                        continue
            return {}


def _label_print(msg: str) -> None:
    import sys
    print(msg, flush=True, file=sys.stdout)