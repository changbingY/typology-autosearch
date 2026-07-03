"""
global_planner.py — Cross-Language Global Research Planner
============================================================
Phase 0 of the multi-language pipeline.

Reads the Table of Contents and/or IGT statistics for EVERY language
before any per-language investigation begins, then asks the LLM to
design a unified set of typological research questions that is
investigatable across the whole language sample.

The output question list is passed to each language's query pipeline
(run_query_pipeline / run_igt_query_pipeline), ensuring all languages
investigate exactly the same phenomena — making cross-linguistic
comparison clean and direct.

Call sequence
-------------
  planner = GlobalPlanner(llm, n_questions=15)
  questions = planner.run(language_configs)
  # → ["What is the basic word order?", "Does the language mark tense?", ...]

Design notes
------------
- Toolkits are loaded ONE AT A TIME and discarded after digest extraction
  to keep peak memory low (large grammar JSONs can be several hundred MB).
- For IGT-only languages we replicate the IGTOnlyAgent trick of passing a
  temporary empty grammar so DeepGrammarToolkit can still index the IGT.
- The planner is aware of what data each language has (grammar / IGT / both)
  and biases questions toward what is investigatable across the whole sample.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class LanguageDigest:
    """
    Lightweight summary of a language's available data.
    Extracted from the toolkit and discarded after planning.
    """
    name:          str
    mode:          str           # "grammar_igt" | "grammar_only" | "igt_only"
    toc_text:      str           # from get_toc_with_summaries(); empty for igt_only
    igt_summary:   str           # from get_igt_summary(); empty if no IGT loaded
    n_toc_entries: int = 0
    n_igt_examples: int = 0


@dataclass
class PlannedQuestion:
    """One question in the unified research plan."""
    question:        str
    domain:          str
    rationale:       str
    answerable_from: list[str] = field(default_factory=list)   # language names


@dataclass
class GlobalPlan:
    """Full output of the global planning phase."""
    languages:          list[str]
    n_questions:        int
    planning_rationale: str
    questions:          list[PlannedQuestion]
    raw_llm_output:     dict = field(default_factory=dict)

    def question_strings(self) -> list[str]:
        """Return just the question text, for passing to query pipelines."""
        return [q.question for q in self.questions]

    def to_dict(self) -> dict:
        return {
            "languages":          self.languages,
            "n_questions":        self.n_questions,
            "planning_rationale": self.planning_rationale,
            "questions": [
                {
                    "question":        q.question,
                    "domain":          q.domain,
                    "rationale":       q.rationale,
                    "answerable_from": q.answerable_from,
                }
                for q in self.questions
            ],
        }


# ════════════════════════════════════════════════════════════════
# Prompt
# ════════════════════════════════════════════════════════════════

GLOBAL_PLAN_PROMPT = """\
You are a linguistic typologist designing a cross-linguistic study.

You are about to investigate {n_languages} languages:
  {language_list}

For each language, you have been provided a DATA DIGEST below — either:
  (a) a TABLE OF CONTENTS from the reference grammar (with section summaries), or
  (b) a QUANTITATIVE IGT SUMMARY (tag frequencies, positional profiles, constructions), or
  (c) both.

════════════════════════════════════════════════════════════════
DATA DIGESTS
════════════════════════════════════════════════════════════════

{digests_block}

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════

Design exactly {n_questions} research questions that will be asked of EVERY language.

These questions will later be answered by a separate deep-search agent that reads each
language's grammar and/or corpus. You are only planning the questions here.

REQUIREMENTS FOR EACH QUESTION:

1. CROSS-LINGUISTIC SCOPE: The question must be about a typological feature that is
   *in principle* investigatable in every language in the sample. Do not ask about
   features that only one language's grammar discusses.

2. ANSWERABLE FROM AVAILABLE DATA: Each question must be answerable from what is
   actually available (grammar prose for grammar languages, IGT patterns for igt-only
   languages). Use the digests to judge what data exists.
   - If a language has only IGT: questions about morphological tags, ordering patterns,
     and construction types are answerable. Questions that require prose descriptions
     (e.g., "What does the author say about...") are NOT answerable.
   - If a language has only grammar: questions about tag frequencies and positional
     profiles may not be answerable. Focus on prose-accessible features.
   - If a language has both: all question types are available.

3. TYPOLOGICAL COVERAGE: Together, the {n_questions} questions must span at least
   FIVE distinct typological domains from this list:
     WORD_ORDER, TENSE_ASPECT_MODALITY, ARGUMENT_MARKING, MORPHOLOGICAL_COMPLEXITY,
     NEGATION, EVIDENTIALITY, INFORMATION_STRUCTURE, NOUN_PHRASE_STRUCTURE,
     AGREEMENT, CASE, SWITCH_REFERENCE, PHONOLOGICAL_PROPERTIES

4. SPECIFICITY: Each question must be specific and produce a clear, classifiable answer.
   GOOD: "What is the basic transitive clause order (SOV / SVO / VSO / VOS / OVS / OSV)?"
   BAD:  "How does syntax work?"
   GOOD: "Does the language morphologically mark tense on the verb (Yes / No / Partial)?"
   BAD:  "What is tense like?"

5. UNIQUENESS: No two questions may investigate the same underlying phenomenon, even
   if worded differently. Scan all {n_questions} questions before finalising.

6. PRIORITY: Prefer questions where the digests give clear signals that the feature
   EXISTS in at least some languages (e.g., if PST appears frequently in IGT summaries,
   include a tense question). Features that appear absent in ALL digests are less useful
   for cross-linguistic comparison, though confirming absence is still typologically valuable.

QUESTION FORMAT RULES:
  - Phrased as a research question (ends with "?")
  - Language-neutral (no language name in the question)
  - Where appropriate, include the expected answer categories in parentheses,
    e.g. "(SOV / SVO / VSO / ...)" or "(Yes / No / Partial)"

Output ONLY valid JSON — no markdown, no extra text:
{{
  "planning_rationale": "2-3 sentences explaining the overall strategy for this language sample",
  "questions": [
    {{
      "question": "What is the basic transitive clause word order (SOV / SVO / VSO / VOS / OVS / OSV / flexible)?",
      "domain": "WORD_ORDER",
      "rationale": "All grammar TOCs have syntax chapters; IGT shows V and N tags in all languages, enabling positional analysis.",
      "answerable_from": ["Aguaruna", "Raramuri", "Yagua"]
    }},
    ...
  ]
}}"""


# ════════════════════════════════════════════════════════════════
# Digest loader
# ════════════════════════════════════════════════════════════════

def _load_grammar_digest(lang_cfg: dict) -> LanguageDigest:
    """
    Load TOC + IGT summary for a grammar (+ optional IGT) language.
    Instantiates DeepGrammarToolkit, extracts text, then discards the toolkit.
    """
    from deep_tools import DeepGrammarToolkit

    name         = lang_cfg["name"]
    grammar_path = lang_cfg["grammar"]
    igt_path     = lang_cfg.get("igt")
    abbrev_path  = lang_cfg.get("abbreviations")
    mode         = "grammar_igt" if igt_path else "grammar_only"

    logger.info(f"  [{name}] Loading toolkit ({mode})...")
    t0 = time.time()

    toolkit = DeepGrammarToolkit(
        grammar_path=grammar_path,
        igt_path=igt_path,
        abbreviations_path=abbrev_path,
    )

    toc_text    = toolkit.get_toc_with_summaries(max_entries=250, max_summary_chars=150)
    igt_summary = toolkit.get_igt_summary() if igt_path else ""

    n_toc   = len(toolkit.chunks)
    n_igt   = len(toolkit.igt_examples)

    # Discard toolkit immediately to free memory
    del toolkit

    logger.info(f"  [{name}] Loaded in {time.time()-t0:.1f}s "
                f"({n_toc} chunks, {n_igt} IGT examples)")

    return LanguageDigest(
        name=name,
        mode=mode,
        toc_text=toc_text,
        igt_summary=igt_summary,
        n_toc_entries=n_toc,
        n_igt_examples=n_igt,
    )


def _load_igt_only_digest(lang_cfg: dict) -> LanguageDigest:
    """
    Load IGT summary for an igt-only language.
    Replicates the IGTOnlyAgent._build_toolkit() trick:
    creates a temporary empty grammar file so DeepGrammarToolkit
    can still load and index the IGT corpus.
    """
    from deep_tools import DeepGrammarToolkit

    name        = lang_cfg["name"]
    igt_path    = lang_cfg["igt"]
    abbrev_path = lang_cfg.get("abbreviations")

    logger.info(f"  [{name}] Loading IGT-only toolkit...")
    t0 = time.time()

    # Temporary empty grammar file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump([], f)
        tmp_grammar = f.name

    try:
        toolkit = DeepGrammarToolkit(
            grammar_path=tmp_grammar,
            igt_path=igt_path,
            abbreviations_path=abbrev_path,
        )
        toolkit.chunks        = []
        toolkit._chunk_by_id  = {}
        toolkit.section_reader = None

        igt_summary = toolkit.get_igt_summary()
        n_igt       = len(toolkit.igt_examples)

        del toolkit
    finally:
        Path(tmp_grammar).unlink(missing_ok=True)

    logger.info(f"  [{name}] IGT loaded in {time.time()-t0:.1f}s ({n_igt} examples)")

    return LanguageDigest(
        name=name,
        mode="igt_only",
        toc_text="",
        igt_summary=igt_summary,
        n_toc_entries=0,
        n_igt_examples=n_igt,
    )


def load_digest(lang_cfg: dict) -> LanguageDigest:
    """Load the appropriate digest for one language config."""
    if lang_cfg.get("igt_only"):
        return _load_igt_only_digest(lang_cfg)
    else:
        return _load_grammar_digest(lang_cfg)


# ════════════════════════════════════════════════════════════════
# Global planner
# ════════════════════════════════════════════════════════════════

class GlobalPlanner:
    """
    Reads all language digests and generates a unified set of
    typological research questions for the full language sample.
    """

    def __init__(self, llm, n_questions: int = 15):
        """
        llm         : QwenLLM instance (from llm.py)
        n_questions : How many unified questions to generate.
                      More = better coverage, more compute per language.
                      Recommended: 10-20.
        """
        self.llm         = llm
        self.n_questions = n_questions

    # ── Digest loading ────────────────────────────────────────────

    def load_all_digests(self, language_configs: list[dict]) -> list[LanguageDigest]:
        """
        Load digests for all languages sequentially.
        Each toolkit is loaded, mined, and discarded before the next is loaded
        to keep peak memory usage low.
        """
        digests = []
        n = len(language_configs)
        for i, lang_cfg in enumerate(language_configs, 1):
            name = lang_cfg["name"]
            print(f"  [{i}/{n}] Reading {name}...", flush=True)
            try:
                digest = load_digest(lang_cfg)
                digests.append(digest)
                mode_label = {
                    "grammar_igt":  "grammar + IGT",
                    "grammar_only": "grammar only",
                    "igt_only":     "IGT only",
                }.get(digest.mode, digest.mode)
                print(f"         ✓ {name} ({mode_label}, "
                      f"{digest.n_toc_entries} sections, "
                      f"{digest.n_igt_examples} IGT examples)")
            except Exception as exc:
                logger.error(f"Failed to load digest for '{name}': {exc}", exc_info=True)
                print(f"         ✗ {name} — FAILED: {exc}")

        return digests

    # ── Prompt building ───────────────────────────────────────────

    def _build_digests_block(self, digests: list[LanguageDigest]) -> str:
        """Format all digests into the prompt block."""
        blocks = []
        for d in digests:
            header = (
                f"┌─ {d.name.upper()} "
                f"[{d.mode.replace('_', ' ').upper()}] "
                f"{'─'*max(0, 55-len(d.name)-len(d.mode))}\n"
            )

            body_parts = []

            if d.toc_text:
                # Truncate TOC to fit in context; keep the most informative parts
                toc_lines = d.toc_text.split("\n")
                # Prefer lines with summaries (they contain the most info)
                summary_lines = [l for l in toc_lines if "Summary:" in l]
                header_lines  = [l for l in toc_lines if "Summary:" not in l]
                # Interleave: keep all headers, keep summaries up to budget
                budget_chars = 3000
                selected = []
                chars = 0
                for h_line, s_line in zip(header_lines, summary_lines + [""]*len(header_lines)):
                    if chars + len(h_line) + len(s_line) > budget_chars:
                        selected.append(f"  ... [{len(toc_lines)-len(selected)} more sections]")
                        break
                    selected.append(h_line)
                    if s_line:
                        selected.append(s_line)
                    chars += len(h_line) + len(s_line)

                body_parts.append("TABLE OF CONTENTS:\n" + "\n".join(selected))

            if d.igt_summary:
                # Truncate IGT summary
                igt_text = d.igt_summary[:2500]
                if len(d.igt_summary) > 2500:
                    igt_text += "\n  [... IGT summary truncated ...]"
                body_parts.append("IGT STATISTICS:\n" + igt_text)

            if not body_parts:
                body_parts.append("(no data available)")

            block = header + "\n\n".join(body_parts) + "\n└" + "─"*60
            blocks.append(block)

        return "\n\n".join(blocks)

    # ── Planning LLM call ─────────────────────────────────────────

    def plan(self, digests: list[LanguageDigest]) -> GlobalPlan:
        """
        Call the LLM with all digests and return a GlobalPlan.
        """
        language_list = ", ".join(d.name for d in digests)
        digests_block = self._build_digests_block(digests)

        prompt = GLOBAL_PLAN_PROMPT.format(
            n_languages=len(digests),
            language_list=language_list,
            n_questions=self.n_questions,
            digests_block=digests_block,
        )

        logger.info("Calling LLM for global planning...")
        t0  = time.time()
        raw = self.llm.generate(prompt, max_new_tokens=2048, json_mode=True)
        logger.info(f"LLM planning call took {time.time()-t0:.1f}s")

        parsed = self._parse_plan(raw)

        if not parsed or "questions" not in parsed:
            logger.warning("LLM output could not be parsed; falling back to generic questions.")
            return self._fallback_plan(digests)

        questions = []
        for q_data in parsed.get("questions", []):
            q_text = q_data.get("question", "").strip()
            if not q_text:
                continue
            questions.append(PlannedQuestion(
                question=q_text,
                domain=q_data.get("domain", "UNKNOWN"),
                rationale=q_data.get("rationale", ""),
                answerable_from=q_data.get("answerable_from", [d.name for d in digests]),
            ))

        return GlobalPlan(
            languages=[d.name for d in digests],
            n_questions=len(questions),
            planning_rationale=parsed.get("planning_rationale", ""),
            questions=questions,
            raw_llm_output=parsed,
        )

    def _parse_plan(self, raw: str) -> Optional[dict]:
        """Strip markdown fences and parse JSON."""
        text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _fallback_plan(self, digests: list[LanguageDigest]) -> GlobalPlan:
        """
        Generic typological questions used when LLM planning fails.
        These are classic Greenbergian / WALS features that are almost
        always investigatable from grammar prose or IGT.
        """
        lang_names = [d.name for d in digests]
        fallback_questions = [
            PlannedQuestion(
                "What is the basic transitive clause word order (SOV / SVO / VSO / VOS / OVS / OSV / flexible)?",
                "WORD_ORDER", "Classic Greenberg Universal 1 — investigatable from syntax chapters and V/N IGT positions.",
                lang_names,
            ),
            PlannedQuestion(
                "What is the order of adjective and noun within the noun phrase (Adj-N / N-Adj / flexible)?",
                "NOUN_PHRASE_STRUCTURE", "Greenberg Universal 18 — investigatable from NP/DP chapters.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language use postpositions, prepositions, or both?",
                "WORD_ORDER", "Greenberg Universal 2 — correlated with verb-final order.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language morphologically encode tense distinctions on the verb (Yes / No / Partial)?",
                "TENSE_ASPECT_MODALITY", "Core TAM feature — detectable from PST/FUT/PRES tags and grammar chapters.",
                lang_names,
            ),
            PlannedQuestion(
                "What aspectual distinctions does the language grammaticalize (perfective/imperfective, completive/progressive, other)?",
                "TENSE_ASPECT_MODALITY", "Aspect is often marked even when tense is not.",
                lang_names,
            ),
            PlannedQuestion(
                "How is negation expressed in declarative clauses (morphological / syntactic / both)?",
                "NEGATION", "NEG tag is common in IGT; negation chapters appear in most grammars.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language have morphological case marking on nouns or pronouns (Yes / No / Partial)?",
                "CASE", "Case tags (NOM, ACC, ERG, ABS, etc.) are detectable in IGT.",
                lang_names,
            ),
            PlannedQuestion(
                "What is the alignment type of the language (nominative-accusative / ergative-absolutive / split / neutral)?",
                "ARGUMENT_MARKING", "Alignment type is a core typological classification.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language have subject-verb agreement marking (Yes / No / Partial)?",
                "AGREEMENT", "AGR tags and agreement chapters are detectable in both grammar and IGT.",
                lang_names,
            ),
            PlannedQuestion(
                "What is the morphological complexity of the language (analytic / agglutinative / fusional / polysynthetic)?",
                "MORPHOLOGICAL_COMPLEXITY", "Morpheme-per-word ratios are visible in IGT; grammars typically classify this.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language have evidentiality markers (grammaticalized encoding of information source)?",
                "EVIDENTIALITY", "EVID/REP/VIS tags in IGT signal evidential systems.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language have switch-reference marking (tracking of same vs. different subject across clauses)?",
                "SWITCH_REFERENCE", "SS/DS tags in IGT are diagnostic.",
                lang_names,
            ),
            PlannedQuestion(
                "How are relative clauses formed (prenominal / postnominal / internally headed / other)?",
                "NOUN_PHRASE_STRUCTURE", "Relative clause formation is a major Greenbergian typological variable.",
                lang_names,
            ),
            PlannedQuestion(
                "Does the language use classifiers or noun class / gender agreement?",
                "NOUN_PHRASE_STRUCTURE", "CL/GEN/M/F tags in IGT; noun class chapters in grammar.",
                lang_names,
            ),
            PlannedQuestion(
                "What subordination strategies does the language use (nominalization / converb / finite embedding / other)?",
                "MORPHOLOGICAL_COMPLEXITY", "Nominalization is common in SOV languages; detectable from NR/NML tags.",
                lang_names,
            ),
        ]
        return GlobalPlan(
            languages=lang_names,
            n_questions=len(fallback_questions),
            planning_rationale="LLM planning failed; using standard Greenbergian feature set as fallback.",
            questions=fallback_questions[:self.n_questions],
        )

    # ── Orchestrator ──────────────────────────────────────────────

    def run(
        self,
        language_configs: list[dict],
        output_dir: Optional[Path] = None,
    ) -> GlobalPlan:
        """
        Full planning pipeline:
          1. Load digests for all languages (sequential, memory-efficient)
          2. LLM call to generate unified question set
          3. Optionally save plan to output_dir/global_plan.json

        Returns a GlobalPlan whose .question_strings() is the list of
        questions to pass to each language's query pipeline.
        """
        print("\n" + "─"*60)
        print("  PHASE 0: GLOBAL RESEARCH PLANNING")
        print("─"*60)
        print(f"  Reading {len(language_configs)} languages before planning...\n")

        # 1. Load digests (one at a time to limit memory)
        digests = self.load_all_digests(language_configs)

        if not digests:
            raise ValueError("No language digests could be loaded. Check your config paths.")

        print(f"\n  All digests loaded. Calling LLM to plan {self.n_questions} questions...\n")

        # 2. LLM planning call
        plan = self.plan(digests)

        # 3. Print plan
        print(f"\n  Planning rationale: {plan.planning_rationale}\n")
        print(f"  Unified research questions ({plan.n_questions} total):")
        for i, q in enumerate(plan.questions, 1):
            print(f"    {i:02d}. [{q.domain}] {q.question}")
        print()

        # 4. Optionally save
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_file = output_dir / "global_plan.json"
            plan_file.write_text(
                json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Plan saved: {plan_file}")

        print("─"*60 + "\n")
        return plan
