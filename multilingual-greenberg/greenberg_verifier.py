"""
greenberg_verifier.py — Greenberg Universal Verifier
=====================================================
Tests one or more Greenberg universals against a multi-language sample.
Input: only a natural-language statement (no pre-parsed antecedent/consequent).

Pipeline (per universal):

  PHASE 0 — Planning
    Read TOC + IGT digests from all already-loaded agents simultaneously.
    One LLM call parses the statement into antecedent / consequent / logic
    and designs language-specific search strategies based on what each
    language's data actually contains.
    Saved to: {uid}/plan.json

  PHASE A — Per-language investigation
    For each language, run answer_query() with a query that includes the
    language's tailored search strategy from the plan.
    Full ReAct loop — grammar sections + IGT corpus.
    Saved to: {uid}/{Language}.json

  PHASE B — Verdict extraction
    One LLM call per language parses the prose QueryResult into:
      antecedent_holds, consequent_holds, violates, confidence, evidence

  PHASE C — Aggregation
    Count support / violation / N/A across all languages.
    Verdict logic uses the logic type inferred in Phase 0:
      ABSOLUTE    → FALSE if ANY consequent = False
      IMPLICATION → FALSE if ANY (antecedent=True, consequent=False)
      CORRELATION → rate-based TRUE / FALSE
    One LLM call writes conclusion + cross-linguistic notes.
    Saved to: {uid}/universal_verdict.json

  PHASE D — Report
    universals_report.md + universals_report.json
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


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GreenbergUniversal:
    """One Greenberg universal to test. Loaded directly from the CSV."""
    uid:        str
    statement:  str
    antecedent: str = ""              # "if" clause — from CSV
    consequent: str = ""              # "then" clause — from CSV
    logic:      str = "implication"   # absolute | implication | correlation — from CSV
    domain:     str = "UNKNOWN"
    source:     str = "Greenberg 1963"

    def short_label(self) -> str:
        return f"{self.uid} [{self.domain}]"


@dataclass
class LanguageStrategy:
    """Per-language search strategy produced by the planner."""
    target_sections: list[str] = field(default_factory=list)
    diagnostic_tags:  list[str] = field(default_factory=list)
    search_focus:     str = ""

    def to_dict(self) -> dict:
        return {
            "target_sections": self.target_sections,
            "diagnostic_tags": self.diagnostic_tags,
            "search_focus":    self.search_focus,
        }


@dataclass
class UniversalPlan:
    """
    Output of Phase 0 — the planner's interpretation of the universal
    and its per-language search strategies.
    """
    antecedent:          str
    consequent:          str
    logic:               str                         # absolute | implication | correlation
    parsing_rationale:   str
    language_strategies: dict[str, LanguageStrategy] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "antecedent":          self.antecedent,
            "consequent":          self.consequent,
            "logic":               self.logic,
            "parsing_rationale":   self.parsing_rationale,
            "language_strategies": {
                lang: s.to_dict() for lang, s in self.language_strategies.items()
            },
        }


@dataclass
class LanguageVerdict:
    """Evidence and verdict for one language against one universal."""
    language:             str
    antecedent_holds:     Optional[bool]
    consequent_holds:     Optional[bool]
    violates:             bool
    confidence:           float
    antecedent_evidence:  str
    consequent_evidence:  str
    notes:                str
    raw_answer:           str

    def to_dict(self) -> dict:
        return {
            "language":             self.language,
            "antecedent_holds":     self.antecedent_holds,
            "consequent_holds":     self.consequent_holds,
            "violates":             self.violates,
            "confidence":           round(self.confidence, 3),
            "antecedent_evidence":  self.antecedent_evidence,
            "consequent_evidence":  self.consequent_evidence,
            "notes":                self.notes,
            "raw_answer_preview":   self.raw_answer[:400],
        }


@dataclass
class UniversalVerdict:
    """Aggregated verdict for one universal across all languages."""
    uid:                    str
    statement:              str
    domain:                 str
    plan:                   UniversalPlan
    verdict:                str     # TRUE | FALSE
    conclusion:             str     # language-neutral typological claim
    cross_linguistic_notes: str     # patterns, names specific languages
    confidence:             float
    language_assessments:   dict    # {lang: {"assessment": ..., "reason": ...}}
    n_valid_support:        int     # after quality review
    n_valid_violation:      int
    n_antecedent_na:        int
    n_irrelevant:           int
    n_insufficient:         int
    language_verdicts:      list[LanguageVerdict]

    def to_dict(self) -> dict:
        return {
            "uid":                    self.uid,
            "statement":              self.statement,
            "domain":                 self.domain,
            "plan":                   self.plan.to_dict(),
            "verdict":                self.verdict,
            "conclusion":             self.conclusion,
            "cross_linguistic_notes": self.cross_linguistic_notes,
            "confidence":             round(self.confidence, 3),
            "n_languages_total":      len(self.language_verdicts),
            "language_assessments":   self.language_assessments,
            "n_valid_support":        self.n_valid_support,
            "n_valid_violation":      self.n_valid_violation,
            "n_antecedent_na":        self.n_antecedent_na,
            "n_irrelevant":           self.n_irrelevant,
            "n_insufficient":         self.n_insufficient,
            "language_verdicts":      [lv.to_dict() for lv in self.language_verdicts],
        }

    def to_markdown(self, index: int = 0) -> str:
        """
        Structure:
          1. Header + statement
          2. Verdict badge + plan (what was searched)
          3. Conclusion  (language-neutral)
          4. Stats + cross-linguistic notes
          5. Compact roster (grouped by verdict)
          6. Per-language details (violating first)
        """
        verdict_emoji = {"TRUE": "✅", "FALSE": "❌"}.get(self.verdict, "❓")

        num = f"{index}. " if index else ""
        p   = self.plan
        lines = [
            f"## {num}{self.uid} — {self.domain}",
            "",
            f"> _{self.statement}_",
            "",
            f"### {verdict_emoji} {self.verdict}  (confidence {self.confidence:.0%})",
            "",
        ]

        if self.conclusion:
            lines += [self.conclusion, ""]

        # Plan summary (what the model decided to look for)
        lines += [
            "**Interpreted as:**",
            f"- Logic: `{p.logic}`",
            f"- Antecedent: _{p.antecedent}_",
            f"- Consequent: _{p.consequent}_",
            "",
        ]

        # Stats bar — uses reviewed counts, not raw Phase B counts
        lines += [
            f"**Valid support:** {self.n_valid_support} · "
            f"**Valid violation:** {self.n_valid_violation} · "
            f"**Antecedent N/A:** {self.n_antecedent_na} · "
            f"**Irrelevant:** {self.n_irrelevant} · "
            f"**Insufficient:** {self.n_insufficient}",
            "",
        ]

        if self.cross_linguistic_notes:
            lines += [
                "**Cross-linguistic patterns:**",
                "",
                self.cross_linguistic_notes,
                "",
            ]

        # Compact roster grouped by assessment
        by_label: dict[str, list[str]] = {}
        for lang, a in self.language_assessments.items():
            lbl = a.get("assessment", "INSUFFICIENT")
            by_label.setdefault(lbl, []).append(lang)

        label_icon = {
            "VALID_SUPPORT":    ("✅", "Valid support"),
            "VALID_VIOLATION":  ("❌", "Valid violation"),
            "ANTECEDENT_NA":    ("–",  "Antecedent N/A"),
            "IRRELEVANT":       ("○",  "Irrelevant"),
            "INSUFFICIENT":     ("?",  "Insufficient"),
        }
        for lbl, (icon, display) in label_icon.items():
            langs = by_label.get(lbl, [])
            if langs:
                lines += [f"**{display}:** " + " · ".join(f"{icon} {l}" for l in langs), ""]

        # Per-language details — violating first
        lines += ["---", "", "### Language details", ""]

        _sort_order = {
            "VALID_VIOLATION": 0, "VALID_SUPPORT": 1,
            "INSUFFICIENT": 2, "ANTECEDENT_NA": 3, "IRRELEVANT": 4,
        }

        def _sort_key(lv: LanguageVerdict) -> int:
            a = self.language_assessments.get(lv.language, {})
            return _sort_order.get(a.get("assessment", "INSUFFICIENT"), 5)

        for lv in sorted(self.language_verdicts, key=_sort_key):
            a_data     = self.language_assessments.get(lv.language, {})
            assessment = a_data.get("assessment", "INSUFFICIENT")
            reason     = a_data.get("reason", "")
            icon, _    = label_icon.get(assessment, ("?", ""))

            ant_str = _tri(lv.antecedent_holds)
            con_str = _tri(lv.consequent_holds)
            strat   = p.language_strategies.get(lv.language)

            lines += [
                f"#### {icon} {lv.language}  ·  `{assessment}`  (conf {lv.confidence:.0%})",
                "",
            ]
            if reason:
                lines += [f"*{reason}*", ""]
            lines += [
                f"| | |",
                f"|---|---|",
                f"| Antecedent | {ant_str} |",
                f"| Consequent | {con_str} |",
                "",
            ]
            if strat and strat.search_focus:
                lines += [f"*Search focus: {strat.search_focus}*", ""]
            if lv.notes:
                lines += [f"_{lv.notes}_", ""]
            if lv.antecedent_evidence:
                lines += [f"**Antecedent evidence:** {lv.antecedent_evidence.strip()[:300]}", ""]
            if lv.consequent_evidence:
                lines += [f"**Consequent evidence:** {lv.consequent_evidence.strip()[:300]}", ""]

        lines.append("---")
        return "\n".join(lines)


def _tri(val: Optional[bool]) -> str:
    if val is True:  return "Yes"
    if val is False: return "No"
    return "Unclear"


# ═══════════════════════════════════════════════════════════════════════════════
# CSV loader  (only id + statement required)
# ═══════════════════════════════════════════════════════════════════════════════

def load_greenberg_csv(path: str | Path) -> dict[str, GreenbergUniversal]:
    """
    Load Greenberg universals CSV.  Required columns: id, statement.
    Optional: domain, source.
    The old antecedent/consequent/logic columns are ignored — the planner
    derives these from the statement at runtime.
    """
    path = Path(path)
    universals: dict[str, GreenbergUniversal] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row.get("id", "").strip()
            if not uid:
                continue
            universals[uid] = GreenbergUniversal(
                uid        = uid,
                statement  = row.get("statement",  "").strip(),
                antecedent = row.get("antecedent", "").strip(),
                consequent = row.get("consequent", "").strip(),
                logic      = (row.get("logic", "implication") or "implication").strip().lower(),
                domain     = row.get("domain",    "UNKNOWN").strip(),
                source     = row.get("source",    "Greenberg 1963").strip(),
            )
    logger.info(f"Loaded {len(universals)} Greenberg universals from {path}")
    return universals


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_PLANNING_PROMPT = """\
You are a linguistic typologist preparing to verify a universal rule
against a multi-language corpus sample.

UNIVERSAL {uid} ({source}):
  Statement  : "{statement}"
  Antecedent : {antecedent}
  Consequent : {consequent}
  Logic      : {logic}

Below are DATA DIGESTS for each language in the sample — either a Table of
Contents (TOC) from the reference grammar, a quantitative IGT tag summary,
or both.

════════════════════════════════════════════════════════════
LANGUAGE DIGESTS
════════════════════════════════════════════════════════════

{digests_block}

════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════

Design a language-specific search strategy for EACH language.

For each language, use its digest to decide:
  target_sections : grammar chapter/section headings most likely to contain
                    direct evidence (copy from TOC, 1–3 headings).
                    Leave [] if the language is IGT-only.
  diagnostic_tags : IGT gloss tags whose presence/absence/position directly
                    tests the antecedent or consequent.
                    Leave [] if the language has no IGT.
  search_focus    : One sentence on exactly what to look for in this language
                    to test whether the antecedent holds AND whether the
                    consequent holds.

Output ONLY valid JSON — no markdown, no extra text:
{{
  "{example_lang}": {{
    "target_sections": ["chapter > section", ...],
    "diagnostic_tags": ["TAG1", "TAG2"],
    "search_focus": "one sentence on what to look for."
  }}
}}"""


_QUERY_TEMPLATE = """\
UNIVERSAL VERIFICATION TASK
======================================
Universal : {uid} ({source})
Statement : "{statement}"

Interpreted structure (from the planning phase):
  Antecedent : {antecedent}
  Consequent : {consequent}
  Logic type : {logic}

Search strategy for {language}:
  Target grammar sections : {target_sections}
  Diagnostic IGT tags     : {diagnostic_tags}
  Focus                   : {search_focus}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: WHAT "HOLDS" MEANS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rules describe DOMINANT typological patterns, not exceptionless rules.

  Antecedent holds  = the described pattern is the DOMINANT or default order in the
                      language, even if minority exceptions exist.
  Consequent holds  = the predicted pattern is likewise the DOMINANT or default order.
  Consequent fails  = the DOMINANT order clearly contradicts the prediction.

Do NOT mark the consequent as failing just because some exceptions or flexibility exists.
Only mark it as failing if the dominant, default, or most frequent order goes against
the prediction. If the language has a dominant order that matches the consequent, it
SUPPORTS the universal — even if a minority pattern differs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVESTIGATION STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Check the ANTECEDENT: "{antecedent}"
  Start with the target sections and diagnostic tags listed above.
  Determine the DOMINANT pattern. Minor exceptions do not negate the antecedent.
  Conclude: does the antecedent hold in {language}? (Yes / No / Unclear)

STEP 2 — Check the CONSEQUENT: "{consequent}"
  Search for evidence of the DOMINANT order. Be explicit about what percentage
  or which contexts are dominant vs. marginal.
  Conclude: does the consequent hold in {language}? (Yes / No / Unclear)

STEP 3 — VERDICT
  Based on logic type ({logic}):
  - ABSOLUTE    : violated only if the consequent clearly fails as the dominant pattern.
  - IMPLICATION : violated ONLY IF antecedent holds AND consequent dominant order fails.
                  If the antecedent does not hold, state NOT APPLICABLE.
  - CORRELATION : does the dominant pattern conform to the predicted tendency?
  State SUPPORTS or VIOLATES (or NOT APPLICABLE) and explain which order is dominant.
"""


_VERDICT_EXTRACTION_PROMPT = """\
You are a typology expert extracting a structured verdict.

UNIVERSAL {uid}: "{statement}"
ANTECEDENT: {antecedent}
CONSEQUENT: {consequent}
LOGIC: {logic}
LANGUAGE: {language}

PROSE INVESTIGATION RESULT:
{answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"antecedent_holds" : true (confirmed) / false (disconfirmed) / null (unclear)
"consequent_holds" : true  — the DOMINANT order matches the prediction
                   false — the DOMINANT order clearly contradicts the prediction
                   null  — insufficient evidence to determine the dominant order

IMPORTANT: "flexible" or "variable" order does NOT mean false.
  false requires clear evidence that the dominant/default order contradicts the prediction.
  If the language's dominant order matches the consequent, mark true — even if
  minority exceptions exist.

"violates"         :
  ABSOLUTE    → true if consequent_holds = false (dominant order fails)
  IMPLICATION → true ONLY if antecedent_holds = true AND consequent_holds = false
  CORRELATION → true if the dominant pattern clearly contradicts the tendency
  Otherwise   → false
"confidence"           : 0.0–1.0. Lower confidence when evidence is sparse or ambiguous.
                         Do NOT assign high confidence (>0.7) when the answer says
                         "unclear", "flexible", or "insufficient evidence".
"antecedent_evidence"  : ≤150-char quote / paraphrase of key antecedent evidence
"consequent_evidence"  : ≤150-char quote / paraphrase of key consequent evidence
"notes"                : one sentence summarising this language's relation to the universal

Output ONLY valid JSON:
{{
  "antecedent_holds": true|false|null,
  "consequent_holds": true|false|null,
  "violates": true|false,
  "confidence": <0.0–1.0>,
  "antecedent_evidence": "...",
  "consequent_evidence": "...",
  "notes": "..."
}}"""


_SYNTHESIS_PROMPT = """\
You are a typological linguist adjudicating a universal rule
against evidence gathered from multiple languages.

UNIVERSAL {uid}: "{statement}"
INTERPRETED AS:
  Antecedent : {antecedent}
  Consequent : {consequent}
  Logic      : {logic}
DOMAIN: {domain}

RAW FINDINGS FROM {n_languages} LANGUAGES:
{verdicts_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Review each language's finding.

For EVERY language listed above:

  (a) Read the FULL ANSWER TEXT carefully — not just the Phase-B labels.
      The Phase-B labels (antecedent_holds, consequent_holds, violates) are
      automated extractions that can be wrong. If the raw answer text says
      something DIFFERENT from the Phase-B label, trust the text.

  (b) Check for internal contradictions: e.g. raw answer says "the antecedent
      does not hold" but Phase-B says antecedent_holds=true. In this case
      the text wins — mark ANTECEDENT_NA or INSUFFICIENT accordingly.

  (c) Check whether the evidence actually addresses the antecedent AND
      consequent of THIS universal specifically, not some related feature.

Then assign one of these labels:

  VALID_SUPPORT    — The antecedent genuinely holds AND the consequent holds.
                     Evidence directly tests this universal. Counts FOR it.
  VALID_VIOLATION  — The antecedent genuinely holds AND the consequent fails.
                     Evidence directly tests this universal. Counts AGAINST it.
  ANTECEDENT_NA    — The antecedent condition does not apply to this language.
                     Neither for nor against. Use this when the raw answer
                     indicates the antecedent fails, EVEN IF Phase-B says otherwise.
  IRRELEVANT       — The agent searched the wrong thing; evidence does not
                     test this universal.
  INSUFFICIENT     — Right direction but too little evidence to decide.

STEP 2 — Derive the verdict from VALID findings only.

  Count only VALID_SUPPORT and VALID_VIOLATION.
  Ignore ANTECEDENT_NA, IRRELEVANT, and INSUFFICIENT for the final decision.

  TRUE  — no VALID_VIOLATION exists AND at least one VALID_SUPPORT exists
           (or all applicable languages are ANTECEDENT_NA)
  FALSE — at least one VALID_VIOLATION exists
           OR no VALID_SUPPORT exists and the universal is not vacuously satisfied

STEP 3 — Write the conclusion and cross-linguistic notes.

  conclusion         : 2–3 sentences, NO individual language names,
                       state TRUE or FALSE and the typological reasoning.
  cross_linguistic_notes : 2–3 sentences naming specific languages,
                           explaining why each was assessed as it was,
                           and what the pattern means cross-linguistically.

Output ONLY valid JSON:
{{
  "language_assessments": {{
    "<language name>": {{
      "assessment": "VALID_SUPPORT|VALID_VIOLATION|ANTECEDENT_NA|IRRELEVANT|INSUFFICIENT",
      "reason": "one sentence explaining this assessment"
    }}
  }},
  "verdict": "TRUE|FALSE",
  "confidence": <0.0–1.0>,
  "conclusion": "2–3 sentence language-neutral typological claim.",
  "cross_linguistic_notes": "2–3 sentence cross-language pattern description."
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Core verifier class
# ═══════════════════════════════════════════════════════════════════════════════

class GreenbergVerifier:
    """
    Orchestrates the full Greenberg verification pipeline.

    Parameters
    ----------
    llm : QwenLLM (or any object with .generate(prompt, max_new_tokens, json_mode))
        Shared LLM — the same instance used by the agents.
    """

    def __init__(self, llm):
        self.llm = llm

    # ── Public entry point ────────────────────────────────────────

    def run(
        self,
        universals:     list[GreenbergUniversal],
        agents:         dict,            # {language_name: agent}
        output_dir:     Path,
        max_iterations: int  = 10,
        skip_existing:  bool = False,
    ) -> list[UniversalVerdict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_verdicts: list[UniversalVerdict] = []

        for u_idx, universal in enumerate(universals, 1):
            print(f"\n{'═'*64}")
            print(f"  Universal {u_idx}/{len(universals)}: {universal.uid}")
            print(f"  \"{universal.statement[:80]}{'...' if len(universal.statement)>80 else ''}\"")
            print(f"{'═'*64}")

            u_dir = output_dir / universal.uid
            u_dir.mkdir(parents=True, exist_ok=True)
            verdict_file = u_dir / "universal_verdict.json"

            # ── Skip if already fully done ────────────────────────
            if skip_existing and verdict_file.exists():
                uv = self._load_verdict(verdict_file, universal)
                if uv:
                    print(f"  [SKIP] Loaded existing verdict from {verdict_file.name}")
                    all_verdicts.append(uv)
                    continue

            # ════════════════════════════════════════════════════
            # PHASE 0 — Planning
            # ════════════════════════════════════════════════════
            plan_file = u_dir / "plan.json"
            if skip_existing and plan_file.exists():
                try:
                    plan = self._load_plan(plan_file)
                    print(f"  [Phase 0] Loaded existing plan.")
                except Exception:
                    plan = None
            else:
                plan = None

            if plan is None:
                print(f"\n  [Phase 0] Planning — reading {len(agents)} language digest(s)...")
                plan = self._plan(universal, agents)
                plan_file.write_text(
                    json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            print(f"  Antecedent : {plan.antecedent[:80]}")
            print(f"  Consequent : {plan.consequent[:80]}")
            print(f"  Logic      : {plan.logic}")

            # ════════════════════════════════════════════════════
            # PHASE A — Per-language investigation
            # ════════════════════════════════════════════════════
            language_verdicts: list[LanguageVerdict] = []

            for l_idx, (lang_name, agent) in enumerate(agents.items(), 1):
                print(f"\n  [Phase A — {l_idx}/{len(agents)}] {lang_name}...")

                lang_file = u_dir / f"{lang_name.replace(' ', '_')}.json"

                if skip_existing and lang_file.exists():
                    try:
                        qr = json.loads(lang_file.read_text(encoding="utf-8"))
                        print(f"    [SKIP] Loaded cached QueryResult.")
                    except Exception:
                        qr = None
                else:
                    qr = None

                if qr is None:
                    query = self._formulate_query(universal, plan, lang_name)
                    try:
                        qr = agent.answer_query(query, max_iterations=max_iterations)
                        if hasattr(qr, "to_dict"):
                            qr = qr.to_dict()
                        for _k in ("grambank_label", "grambank_reasoning"):
                            qr.pop(_k, None)
                        lang_file.write_text(
                            json.dumps(qr, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        logger.error(f"Agent query failed for {lang_name}: {exc}", exc_info=True)
                        qr = {"answer": f"ERROR: {exc}", "confidence": 0.0, "key_evidence": []}

                # Phase B inline
                lv = self._extract_verdict(plan, lang_name, qr)
                language_verdicts.append(lv)

                icon = "❌ VIOLATES" if lv.violates else (
                    "✅ supports" if lv.antecedent_holds is not False
                    else "–  N/A"
                )
                print(f"    {icon}  ant={_tri(lv.antecedent_holds)} "
                      f"con={_tri(lv.consequent_holds)} conf={lv.confidence:.2f}")
                if lv.notes:
                    print(f"    Notes: {lv.notes}")

            # ════════════════════════════════════════════════════
            # PHASE C — Aggregation
            # ════════════════════════════════════════════════════
            uv = self._aggregate(universal, plan, language_verdicts)
            verdict_file.write_text(
                json.dumps(uv.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n  ── VERDICT: {uv.verdict} "
                  f"(sup={uv.n_valid_support} vio={uv.n_valid_violation} "
                  f"conf={uv.confidence:.2f}) ──")
            print(f"  {uv.conclusion[:200]}")
            all_verdicts.append(uv)

        # Phase D
        self._build_report(all_verdicts, output_dir)
        return all_verdicts

    # ── Phase 0: planning ─────────────────────────────────────────

    def _plan(
        self,
        u:      GreenbergUniversal,
        agents: dict,
    ) -> UniversalPlan:
        """
        Read TOC + IGT digests from all agents, then call the LLM to:
          1. Parse the statement into antecedent / consequent / logic
          2. Design a per-language search strategy based on the digests
        """
        # Build digest block (reads from already-loaded agent toolkits)
        digest_parts = []
        for lang_name, agent in agents.items():
            toc, igt = self._get_digest(agent)
            block = f"── {lang_name.upper()} ──\n"
            if toc:
                block += f"TABLE OF CONTENTS (excerpt):\n{toc[:2000]}\n"
            if igt:
                block += f"IGT TAG SUMMARY (excerpt):\n{igt[:1000]}\n"
            if not toc and not igt:
                block += "(no digest available)\n"
            digest_parts.append(block)

        digests_block = "\n\n".join(digest_parts)
        example_lang  = next(iter(agents.keys()), "Language1")

        prompt = _PLANNING_PROMPT.format(
            uid           = u.uid,
            source        = u.source,
            statement     = u.statement,
            antecedent    = u.antecedent or u.statement,
            consequent    = u.consequent or u.statement,
            logic         = u.logic,
            digests_block = digests_block,
            example_lang  = example_lang,
        )

        raw    = self.llm.generate(prompt, max_new_tokens=2048, json_mode=True)
        parsed = _parse_json(raw)

        # The LLM only returns strategies now (no ant/con/logic inference).
        # Strategies may be at top level (new format) or nested under
        # "language_strategies" (old format) — handle both.
        if parsed and next(iter(parsed), None) in agents:
            strategies_raw = parsed          # top-level {lang: {...}}
        else:
            strategies_raw = parsed.get("language_strategies", {}) if parsed else {}

        if not strategies_raw:
            logger.warning("Planning LLM returned no strategies; using fallback.")
            return self._fallback_plan(u, agents)

        strategies: dict[str, LanguageStrategy] = {}
        for lang_name in agents:
            s_data = strategies_raw.get(lang_name, {})
            strategies[lang_name] = LanguageStrategy(
                target_sections = s_data.get("target_sections", []),
                diagnostic_tags = s_data.get("diagnostic_tags", []),
                search_focus    = s_data.get("search_focus", ""),
            )

        # antecedent/consequent/logic come from the CSV, not from the LLM
        return UniversalPlan(
            antecedent          = u.antecedent,
            consequent          = u.consequent,
            logic               = u.logic,
            parsing_rationale   = "from CSV",
            language_strategies = strategies,
        )

    @staticmethod
    def _fallback_plan(u: GreenbergUniversal, agents: dict) -> UniversalPlan:
        """Minimal plan when LLM planning fails — agent will still search."""
        strategies = {
            lang: LanguageStrategy(search_focus=f"Search for evidence relating to: {u.statement[:100]}")
            for lang in agents
        }
        return UniversalPlan(
            antecedent          = u.antecedent or u.statement,
            consequent          = u.consequent or u.statement,
            logic               = u.logic,
            parsing_rationale   = "Fallback — LLM planning failed.",
            language_strategies = strategies,
        )

    @staticmethod
    def _get_digest(agent) -> tuple[str, str]:
        """
        Extract TOC and IGT summary from an already-loaded agent.

        DeepLanguageResearchAgent  → toolkit.get_toc_with_summaries()
                                     toolkit.get_igt_summary()
        IGTOnlyResearchAgent       → agent._igt_summary  (pre-built at init)
                                     toolkit.igt_analyser.get_stats().summary_text
        """
        toc, igt = "", ""
        try:
            tk = getattr(agent, "toolkit", None)

            # ── TOC (grammar only) ────────────────────────────────
            if tk and hasattr(tk, "get_toc_with_summaries"):
                result = tk.get_toc_with_summaries()
                if isinstance(result, str):
                    toc = result[:3000]

            # ── IGT summary — three fallback paths ────────────────
            # Path 1: DeepGrammarToolkit with IGT loaded
            if not igt and tk and hasattr(tk, "get_igt_summary"):
                result = tk.get_igt_summary()
                if isinstance(result, str):
                    igt = result[:1500]

            # Path 2: IGTOnlyResearchAgent caches the summary directly
            if not igt:
                cached = getattr(agent, "_igt_summary", None)
                if isinstance(cached, str) and cached.strip():
                    igt = cached[:1500]

            # Path 3: raw access via igt_analyser
            if not igt and tk:
                analyser = getattr(tk, "igt_analyser", None)
                if analyser and hasattr(analyser, "get_stats"):
                    text = getattr(analyser.get_stats(), "summary_text", "")
                    if isinstance(text, str):
                        igt = text[:1500]

        except Exception as exc:
            logger.debug(f"Could not extract digest from agent: {exc}")
        return toc, igt

    @staticmethod
    def _load_plan(path: Path) -> UniversalPlan:
        data = json.loads(path.read_text(encoding="utf-8"))
        strategies = {}
        for lang, s in data.get("language_strategies", {}).items():
            strategies[lang] = LanguageStrategy(
                target_sections = s.get("target_sections", []),
                diagnostic_tags = s.get("diagnostic_tags", []),
                search_focus    = s.get("search_focus", ""),
            )
        return UniversalPlan(
            antecedent          = data.get("antecedent", ""),
            consequent          = data.get("consequent", ""),
            logic               = data.get("logic", "implication"),
            parsing_rationale   = data.get("parsing_rationale", ""),
            language_strategies = strategies,
        )

    # ── Phase A: query formulation ────────────────────────────────

    @staticmethod
    def _formulate_query(
        u:    GreenbergUniversal,
        plan: UniversalPlan,
        lang: str,
    ) -> str:
        strat = plan.language_strategies.get(lang, LanguageStrategy())
        return _QUERY_TEMPLATE.format(
            uid             = u.uid,
            source          = u.source,
            statement       = u.statement,
            antecedent      = plan.antecedent,
            consequent      = plan.consequent,
            logic           = plan.logic,
            language        = lang,
            target_sections = ", ".join(strat.target_sections) or "(use your judgment)",
            diagnostic_tags = ", ".join(strat.diagnostic_tags)  or "(use your judgment)",
            search_focus    = strat.search_focus or u.statement[:100],
        )

    # ── Phase B: verdict extraction ───────────────────────────────

    def _extract_verdict(
        self,
        plan:     UniversalPlan,
        language: str,
        qr:       dict,
    ) -> LanguageVerdict:
        if hasattr(qr, "to_dict"):
            qr = qr.to_dict()

        answer = (qr.get("answer") or "").strip()
        key_ev = qr.get("key_evidence") or []
        if key_ev:
            answer += "\n\nKEY EVIDENCE:\n" + "\n".join(f"• {e}" for e in key_ev[:5])

        prompt = _VERDICT_EXTRACTION_PROMPT.format(
            uid        = "(universal)",
            statement  = "(see plan)",
            antecedent = plan.antecedent,
            consequent = plan.consequent,
            logic      = plan.logic,
            language   = language,
            answer     = answer[:3000],
        )

        raw    = self.llm.generate(prompt, max_new_tokens=512, json_mode=True)
        parsed = _parse_json(raw)

        ant  = _to_tribool(parsed.get("antecedent_holds"))
        con  = _to_tribool(parsed.get("consequent_holds"))
        conf = float(parsed.get("confidence", qr.get("confidence", 0.5) or 0.5))

        if "violates" in parsed:
            violates = bool(parsed["violates"])
        else:
            violates = _compute_violates(plan.logic, ant, con)

        return LanguageVerdict(
            language             = language,
            antecedent_holds     = ant,
            consequent_holds     = con,
            violates             = violates,
            confidence           = conf,
            antecedent_evidence  = str(parsed.get("antecedent_evidence", ""))[:300],
            consequent_evidence  = str(parsed.get("consequent_evidence", ""))[:300],
            notes                = str(parsed.get("notes", ""))[:200],
            raw_answer           = (qr.get("answer") or "")[:400],
        )

    # ── Phase C: aggregation ──────────────────────────────────────

    def _aggregate(
        self,
        u:                 GreenbergUniversal,
        plan:              UniversalPlan,
        language_verdicts: list[LanguageVerdict],
    ) -> UniversalVerdict:
        avg_conf = sum(lv.confidence for lv in language_verdicts) / max(len(language_verdicts), 1)

        # ── Pre-screen: catch antecedent failures the agent stated in prose ──
        # Phase B sometimes extracts antecedent_holds=True even when the
        # agent's own answer text says "the antecedent does not hold".
        # We catch these before the Phase C call to prevent false violations.
        prescreened_na: set[str] = set()
        if plan.logic == "implication":
            for lv in language_verdicts:
                if lv.violates and _antecedent_fails_in_text(lv.raw_answer):
                    prescreened_na.add(lv.language)
                    logger.info(
                        f"Pre-screen: {lv.language} → ANTECEDENT_NA "
                        f"(raw answer says antecedent fails despite Phase-B VIOLATES)"
                    )

        prompt = _SYNTHESIS_PROMPT.format(
            uid            = u.uid,
            statement      = u.statement,
            antecedent     = plan.antecedent,
            consequent     = plan.consequent,
            logic          = plan.logic,
            domain         = u.domain,
            n_languages    = len(language_verdicts),
            verdicts_block = self._verdicts_block(language_verdicts),
        )

        raw    = self.llm.generate(prompt, max_new_tokens=1536, json_mode=True)
        parsed = _parse_json(raw)

        # ── Extract per-language assessments from Phase C LLM ───────
        # Phase C can only EXCLUDE (IRRELEVANT / ANTECEDENT_NA) a language.
        # It CANNOT downgrade a Phase-B violation to non-violation —
        # that would require overriding concrete antecedent+consequent evidence.
        # Allowed Phase-C overrides:
        #   VALID_SUPPORT    → kept as-is if Phase B also says supports
        #   VALID_VIOLATION  → kept as-is if Phase B also says violates
        #   ANTECEDENT_NA    → kept regardless (Phase C may spot that the
        #                      antecedent doesn't truly hold)
        #   IRRELEVANT       → kept regardless (Phase C spotted off-target search)
        #   INSUFFICIENT     → treated as: defer to Phase B

        raw_assessments: dict = parsed.get("language_assessments", {})
        language_assessments: dict = {}
        valid_support:   list[str] = []
        valid_violation: list[str] = []
        antecedent_na:   list[str] = []
        irrelevant:      list[str] = []
        insufficient:    list[str] = []

        lv_by_lang = {lv.language: lv for lv in language_verdicts}

        for lang, lv in lv_by_lang.items():
            a_data  = raw_assessments.get(lang, {})
            c_label = str(a_data.get("assessment", "INSUFFICIENT")).upper()
            reason  = str(a_data.get("reason", ""))
            if c_label not in ("VALID_SUPPORT", "VALID_VIOLATION",
                               "ANTECEDENT_NA", "IRRELEVANT", "INSUFFICIENT"):
                c_label = "INSUFFICIENT"

            # ── Pre-screen override ──────────────────────────────
            # Agent's own answer text says antecedent fails — trust the prose.
            if lang in prescreened_na:
                final_label = "ANTECEDENT_NA"
                reason = ("Pre-screen: agent's answer states antecedent does not hold, "
                          "overriding Phase-B VIOLATES extraction.")
                language_assessments[lang] = {"assessment": final_label, "reason": reason}
                antecedent_na.append(lang)
                continue

            # ── Confidence gate ─────────────────────────────────
            # Phase B results with confidence below the threshold cannot be
            # treated as reliable enough for VALID_SUPPORT or VALID_VIOLATION.
            # Phase C can still override to ANTECEDENT_NA or IRRELEVANT.
            MIN_CONF = 0.65
            conf_ok  = lv.confidence >= MIN_CONF

            # Phase C can exclude (ANTECEDENT_NA / IRRELEVANT) freely.
            if c_label in ("ANTECEDENT_NA", "IRRELEVANT"):
                final_label = c_label

            elif not conf_ok:
                # Too uncertain — do not count as a valid finding.
                final_label = "INSUFFICIENT"
                reason = (f"Phase-B confidence too low ({lv.confidence:.2f} < {MIN_CONF}) "
                          f"to treat as a reliable verdict.")

            elif lv.violates:
                # Phase B found a clear violation AND confidence is adequate.
                final_label = "VALID_VIOLATION"
                if c_label != "VALID_VIOLATION":
                    reason = (f"Phase-B override: antecedent=Yes, consequent=No "
                              f"(conf={lv.confidence:.2f}); Phase-C said {c_label}")

            elif lv.antecedent_holds is False and plan.logic == "implication":
                final_label = "ANTECEDENT_NA"
                if not reason:
                    reason = "phase-B: antecedent does not hold"

            elif lv.antecedent_holds is True and lv.consequent_holds is True:
                final_label = "VALID_SUPPORT"
                if not reason:
                    reason = "phase-B: antecedent and consequent both hold"

            else:
                final_label = c_label if c_label != "INSUFFICIENT" else "INSUFFICIENT"

            language_assessments[lang] = {"assessment": final_label, "reason": reason}
            if final_label == "VALID_SUPPORT":    valid_support.append(lang)
            elif final_label == "VALID_VIOLATION": valid_violation.append(lang)
            elif final_label == "ANTECEDENT_NA":  antecedent_na.append(lang)
            elif final_label == "IRRELEVANT":     irrelevant.append(lang)
            else:                                 insufficient.append(lang)

        # ── Derive verdict ────────────────────────────────────────
        # Primary: LLM verdict from reading all evidence.
        # Fallback (if LLM returns nothing usable): mechanical count.
        llm_verdict = (parsed.get("verdict") or "").upper()

        if llm_verdict in ("TRUE", "FALSE"):
            verdict = llm_verdict
        else:
            # Mechanical fallback from Phase-B-anchored counts
            if valid_violation:
                verdict = "FALSE"
            elif valid_support or len(antecedent_na) == len(lv_by_lang):
                verdict = "TRUE"
            else:
                verdict = "FALSE"

        conclusion             = parsed.get("conclusion", "(synthesis not available)")
        cross_linguistic_notes = parsed.get("cross_linguistic_notes", "")
        llm_conf               = float(parsed.get("confidence", avg_conf) or avg_conf)

        return UniversalVerdict(
            uid                    = u.uid,
            statement              = u.statement,
            domain                 = u.domain,
            plan                   = plan,
            verdict                = verdict,
            conclusion             = conclusion,
            cross_linguistic_notes = cross_linguistic_notes,
            confidence             = llm_conf,
            language_assessments   = language_assessments,
            n_valid_support        = len(valid_support),
            n_valid_violation      = len(valid_violation),
            n_antecedent_na        = len(antecedent_na),
            n_irrelevant           = len(irrelevant),
            n_insufficient         = len(insufficient),
            language_verdicts      = language_verdicts,
        )

    @staticmethod
    def _verdicts_block(language_verdicts: list[LanguageVerdict]) -> str:
        """
        Build the evidence block passed to the synthesis LLM.
        Passes the full raw answer and all key evidence so the LLM can make
        a holistic judgment rather than just accepting Phase-B label counts.
        """
        lines = []
        for lv in language_verdicts:
            phase_b = "VIOLATES" if lv.violates else (
                "SUPPORTS" if lv.antecedent_holds is not False else "ANTECEDENT_NA"
            )
            block = (
                f"── {lv.language.upper()} ──\n"
                f"Phase-B classification : {phase_b}\n"
                f"Antecedent holds       : {_tri(lv.antecedent_holds)}\n"
                f"Consequent holds       : {_tri(lv.consequent_holds)}\n"
                f"Confidence             : {lv.confidence:.2f}\n"
                f"Notes                 : {lv.notes or '—'}\n"
                f"Antecedent evidence   : {lv.antecedent_evidence or '—'}\n"
                f"Consequent evidence   : {lv.consequent_evidence or '—'}\n"
                f"Full answer:\n{lv.raw_answer or '—'}"
            )
            lines.append(block)
        return "\n\n".join(lines)

    # ── Phase D: report ───────────────────────────────────────────

    def _build_report(self, verdicts: list[UniversalVerdict], output_dir: Path) -> None:
        all_langs = sorted({lv.language for uv in verdicts for lv in uv.language_verdicts})
        md_lines = [
            "# Greenberg Universals Verification Report",
            "",
            f"**Universals tested:** {len(verdicts)}",
            f"**Languages:** " + ", ".join(all_langs),
            "",
            "---",
            "",
            "## Summary Table",
            "",
            "| # | Universal | Domain | Verdict | Support | Violation | N/A | Irrel | Insuf |",
            "|---|-----------|--------|---------|---------|-----------|-----|-------|-------|",
        ]
        for i, uv in enumerate(verdicts, 1):
            emoji = {"TRUE": "✅", "FALSE": "❌"}.get(uv.verdict, "❓")
            stmt_short = uv.statement[:55] + ("…" if len(uv.statement) > 55 else "")
            md_lines.append(
                f"| {i} | **{uv.uid}** {stmt_short} | {uv.domain} "
                f"| {emoji} {uv.verdict} "
                f"| {uv.n_valid_support} | {uv.n_valid_violation} "
                f"| {uv.n_antecedent_na} | {uv.n_irrelevant} | {uv.n_insufficient} |"
            )

        md_lines += ["", "---", ""]
        for i, uv in enumerate(verdicts, 1):
            md_lines.append(uv.to_markdown(index=i))
            md_lines.append("")

        md_lines += ["---", "_Generated by GreenbergVerifier_"]

        md_file   = output_dir / "universals_report.md"
        json_file = output_dir / "universals_report.json"
        md_file.write_text("\n".join(md_lines), encoding="utf-8")
        json_file.write_text(
            json.dumps({"n_universals": len(verdicts),
                        "verdicts": [uv.to_dict() for uv in verdicts]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Report : {md_file}")
        print(f"  Data   : {json_file}")

    # ── Resume helper ─────────────────────────────────────────────

    @staticmethod
    def _load_verdict(path: Path, universal: GreenbergUniversal) -> Optional[UniversalVerdict]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Re-hydrate plan
            p_data = data.get("plan", {})
            strategies = {}
            for lang, s in p_data.get("language_strategies", {}).items():
                strategies[lang] = LanguageStrategy(
                    target_sections = s.get("target_sections", []),
                    diagnostic_tags = s.get("diagnostic_tags", []),
                    search_focus    = s.get("search_focus", ""),
                )
            plan = UniversalPlan(
                antecedent          = p_data.get("antecedent", ""),
                consequent          = p_data.get("consequent", ""),
                logic               = p_data.get("logic", "implication"),
                parsing_rationale   = p_data.get("parsing_rationale", ""),
                language_strategies = strategies,
            )
            lvs = [
                LanguageVerdict(
                    language            = lv["language"],
                    antecedent_holds    = lv.get("antecedent_holds"),
                    consequent_holds    = lv.get("consequent_holds"),
                    violates            = lv.get("violates", False),
                    confidence          = lv.get("confidence", 0.0),
                    antecedent_evidence = lv.get("antecedent_evidence", ""),
                    consequent_evidence = lv.get("consequent_evidence", ""),
                    notes               = lv.get("notes", ""),
                    raw_answer          = lv.get("raw_answer_preview", ""),
                )
                for lv in data.get("language_verdicts", [])
            ]
            la = data.get("language_assessments", {})
            return UniversalVerdict(
                uid                    = data.get("uid", universal.uid),
                statement              = data.get("statement", universal.statement),
                domain                 = data.get("domain", universal.domain),
                plan                   = plan,
                verdict                = data.get("verdict", "FALSE"),
                conclusion             = data.get("conclusion", data.get("summary", "")),
                cross_linguistic_notes = data.get("cross_linguistic_notes", ""),
                confidence             = data.get("confidence", 0.0),
                language_assessments   = la,
                n_valid_support        = data.get("n_valid_support",
                                             sum(1 for a in la.values() if a.get("assessment") == "VALID_SUPPORT")),
                n_valid_violation      = data.get("n_valid_violation",
                                             sum(1 for a in la.values() if a.get("assessment") == "VALID_VIOLATION")),
                n_antecedent_na        = data.get("n_antecedent_na", 0),
                n_irrelevant           = data.get("n_irrelevant", 0),
                n_insufficient         = data.get("n_insufficient", 0),
                language_verdicts      = lvs,
            )
        except Exception as exc:
            logger.warning(f"Could not reload verdict from {path}: {exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Phrases the agent writes when the antecedent clearly does not hold.
# If any appear in the raw answer we trust the agent's prose over Phase B's
# structured extraction.
_ANT_FAIL_PHRASES: tuple[str, ...] = (
    "the antecedent does not hold",
    "the antecedent doesn't hold",
    "antecedent does not hold",
    "antecedent doesn't hold",
    "antecedent fails",
    "antecedent not clearly established",
    "does not meet the antecedent",
    "antecedent condition is not met",
    "antecedent condition does not hold",
    "state not applicable",
    "verdict: not applicable",
    "not applicable",
)


def _antecedent_fails_in_text(raw_answer: str) -> bool:
    """
    Return True if the agent's own prose indicates that the antecedent does
    not hold for this language — regardless of what Phase B extracted.
    Matches are case-insensitive.
    """
    text = raw_answer.lower()
    return any(phrase in text for phrase in _ANT_FAIL_PHRASES)


def _to_tribool(val) -> Optional[bool]:
    if val is True  or val in ("true",  "Yes"): return True
    if val is False or val in ("false", "No"):  return False
    return None


def _compute_violates(logic: str, ant: Optional[bool], con: Optional[bool]) -> bool:
    if logic == "absolute":    return con is False
    if logic == "implication": return ant is True and con is False
    if logic == "correlation": return ant is True and con is False
    return False


def _parse_json(text: str) -> dict:
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
                    return json.loads(text[:i+1])
                except json.JSONDecodeError:
                    continue
        return {}