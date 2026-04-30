"""
igt_agent.py — IGTOnlyResearchAgent
=====================================
A bottom-up typological research agent that works purely from IGT data,
with no reference grammar required.

Workflow (mirrors a field linguist's approach):
  1. IGT Inventory  — compute full statistics over the IGT corpus
  2. Domain Discovery — LLM infers typological domains from tag distributions
  3. Feature Planning — plan investigation using only IGT tools
  4. ReAct Loop       — analyse_tag / analyse_construction / analyse_absence /
                        compare_tags / get_section_igt / get_tag_inventory /
                        get_construction_inventory / find_tag_cluster
  5. Conclusion + Audit — same rigour as deep_agent.py, but evidence
                          requirements are adjusted: no grammar prose needed,
                          but IGT_PATTERN + ABSENCE_EVIDENCE are mandatory

Key difference from DeepLanguageResearchAgent:
  - Domain discovery starts from IGT statistics, not grammar TOC
  - No grammar-reading tools (read_full_section, extract_author_claims, etc.)
  - EvidenceGraph gap analysis adapted: GRAMMAR_STATEMENT not required
  - Conclusions explicitly labelled as "data-inferred, no grammar consulted"
  - AbbreviationRegistry: gloss tag meanings are injected into every prompt
    that lists IGT tags, improving LLM reasoning over unfamiliar abbreviations.
"""

import json
import re
from collections import Counter   
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from deep_tools import DeepGrammarToolkit
from evidence_graph import EvidenceGraph, ClaimType
from state import EpistemicState, Feature

logger = logging.getLogger(__name__)


def _print(msg: str, indent: int = 0) -> None:
    import sys
    print(" " * indent + msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

IGT_DOMAIN_EXTRACTION_PROMPT = """You are a linguistic typologist. Below is quantitative IGT data for the language "{language}". There is NO reference grammar — your analysis must be grounded entirely in the IGT corpus.

{abbrev_legend}

IGT CORPUS STATISTICS:
{igt_summary}

TOP CONSTRUCTION PATTERNS (most frequent ordered tag sequences):
{construction_patterns}

Your task: infer typological domains and feature questions purely from the data.

Reading the statistics:
- Tags with >5% corpus coverage are likely GRAMMATICALIZED categories
- Tags with <0.5% coverage may be marginal, borrowed, or absent
- Consistent positional labels ("preverbal/initial", "postverbal/final") reveal morphosyntactic slot
- Complementary tags (PST vs FUT, SG vs PL) suggest paradigm systems
- Near-zero categories in "LIKELY ABSENT" section are candidates for confirmed absence
- Frequent bigrams/trigrams reveal how morphemes cluster into constructions

Procedure:
1. Scan all tags and group by typological domain (TMA, Agreement, Case, Negation, etc.)
2. For each domain, identify what IS present (high frequency), what is PARTIAL (low frequency),
   and what is ABSENT (zero/near-zero)
3. Generate 3-5 answerable binary/categorical feature questions per domain
4. Mark your prior based on the data signal:
   - "likely_yes"  — tag frequency >5% or construction pattern is frequent
   - "likely_no"   — absent category confirmed by near-zero counts
   - "uncertain"   — low but non-zero frequency, positional instability

Each question must be fully answerable with the available IGT tools:
  analyse_tag, analyse_construction, analyse_absence, compare_tags,
  get_section_igt, get_tag_inventory, get_construction_inventory, find_tag_cluster

Output ONLY valid JSON:
{{
  "domains": [
    {{
      "domain_id": "D001",
      "domain_name": "TENSE_ASPECT_MODALITY",
      "igt_signals": ["PST: 140 (8.4%)", "IPFV: 428 (25.7%)", "FUT: 0 (0.0%)"],
      "candidate_features": [
        {{
          "feature_id": "F001",
          "question": "Does {language} grammaticalize tense distinctions?",
          "type": "binary",
          "value_space": ["Yes", "No", "Partial", "Unclear"],
          "igt_tags_to_check": ["PST", "FUT", "PRF"],
          "prior": "likely_yes"
        }}
      ]
    }}
  ]
}}"""


IGT_FEATURE_PLAN_PROMPT = """You are a linguistic typologist planning a data-driven investigation.

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}
IGT PRIOR SIGNALS: {igt_signals}

{abbrev_legend}

STEP 1 — CANDIDATE TAGS FROM ABBREVIATION REGISTRY (matched to this phenomenon):
{abbrev_candidates}

Use these as your primary starting point for igt_tags_to_check:
- Tags listed as IN CORPUS are confirmed present — analyse them first.
- Tags listed as NOT IN CORPUS are still worth checking via analyse_absence.
- Only add tags NOT in the list above if they appear prominently in the IGT statistics
  AND are clearly diagnostic for the phenomenon being investigated.

CRITICAL: IGT PRIOR SIGNALS may contain generic high-frequency tags (PST, FUT, IMPF, etc.)
that are NOT relevant to this phenomenon. DO NOT copy those into igt_tags_to_check.
Use ONLY tags from the STEP 1 list, or tags you can explicitly justify as diagnostic for "{question}".

FULL IGT STATISTICS (for cross-referencing candidate tags above):
{igt_summary}

TOP CONSTRUCTION PATTERNS:
{construction_patterns}

There is NO reference grammar. Plan the investigation using ONLY the following tools:
  1. get_tag_inventory          — sorted list of all tags with frequencies (start here)
  2. get_construction_inventory — top bigrams and trigrams across the corpus
  3. find_tag_cluster(seed_tag) — find tags that co-occur most with a given tag
  4. analyse_tag(tag)           — deep profile: frequency, position, co-occurrents
  5. analyse_construction(tags) — examples of a specific ordered tag sequence
  6. analyse_absence(category)  — quantify absence of a typological category
  7. compare_tags(tag_a, tag_b) — PMI + positional comparison
  8.  get_section_igt(query)          — IGT from sections whose label matches the query
  9.  analyse_semantic_context(tag)   — for examples with this tag, analyse translation-line
                                        semantics to cross-validate the tag's interpretation
  10. search_translations(query)      — keyword search across translation lines; returns
                                        examples and co-occurring tags

REQUIRED evidence before concluding:
  1. At least one IGT_PATTERN claim (from analyse_tag, analyse_construction, or get_construction_inventory)
  2. At least one ABSENCE_EVIDENCE check (from analyse_absence)
  3. At least one COUNTER-EVIDENCE check (compare_tags or absence check with alternative hypothesis)
  4. If contradictions detected → must be resolved

Planning strategy:
  - Start with the tags flagged in igt_tags_to_check
  - Use find_tag_cluster to discover related tags the inventory might not reveal
  - Use analyse_construction to test morpheme order hypotheses
  - Use compare_tags to distinguish competing analyses (e.g. tense vs aspect)
  - Formulate counter_evidence_framing as a serious alternative hypothesis

Output ONLY valid JSON:
{{
  "igt_tags_to_check": ["PST", "PRF", "FUT"],
  "constructions_to_check": [["SBJ", "PST", "V"], ["NEG", "PST"]],
  "cluster_seeds": ["PST", "IPFV"],
  "category_absence_check": "TENSE",
  "tags_to_compare": [["PST", "PFV"], ["PRF", "PFV"]],
  "counter_evidence_framing": "aspect-only language with no tense, temporal reference via adverbs",
  "min_queries_before_conclude": 5
}}"""


IGT_SEARCH_DECISION_PROMPT = """You are deep-searching an IGT corpus for a typological feature. There is NO reference grammar.

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}

CURRENT EVIDENCE GRAPH:
{evidence_summary}

EVIDENCE GAPS (what is still missing):
{gap_analysis}

Progress: iteration {iteration}/{max_iter}
Constraints met: {constraints_met}

AVAILABLE TOOLS (IGT-only — no grammar reading tools):
1.  get_tag_inventory               — full sorted list of all tags with frequencies
2.  get_construction_inventory      — top bigrams and trigrams in the corpus
3.  find_tag_cluster(seed_tag)      — tags that co-occur most with seed_tag
4.  analyse_tag(tag)                — frequency, position, co-occurrence profile
5.  analyse_construction(tags)      — find ordered tag sequence [TAG1, TAG2, ...]
6.  analyse_absence(category)       — quantify absence (real negative evidence)
7.  compare_tags(tag_a, tag_b)      — PMI + positional comparison
8.  get_section_igt(query)          — all IGT from sections matching the query string
9.  analyse_semantic_context(tag)   — for examples with this tag, analyse translation-line
                                      semantics: what temporal/modal/negation words appear?
                                      Cross-validates tag interpretation against sentence meaning.
10. search_translations(query)      — keyword search across all translation lines; returns
                                      matching examples AND which tags co-occur with the query.
                                      Use to find constructions by meaning rather than tag.
11. get_triline_examples(query)     — show aligned MORPHEME / GLOSS / TRANSLATION trilines
                                      for examples matching a tag, translation keyword, or
                                      section name. Use before parse_example_structure.
12. analyse_morpheme_position(tag)  — determine whether a morpheme is a PREFIX, SUFFIX, or
                                      STEM by analysing its position within words in the
                                      morpheme line. Essential for morphological typology.
13. parse_example_structure(query)  — LLM-powered clause analysis: fetches trilines for
                                      the query, then identifies subject/predicate/object,
                                      clause type, word order, and morphological locus.
                                      Use when statistical tools alone cannot resolve structure.
14. get_morpheme_forms(tag)         — show the ACTUAL SURFACE FORMS of a morpheme in this
                                      language (e.g. FUT → "'-ma' (22x), '=ma' (6x)").
                                      Use when the query asks what a marker "looks like" or
                                      when the answer should describe the language's forms.
15. analyse_tag_usage(tag)          — LLM reads all examples with this tag and infers its
                                      grammatical function: primary function, morpheme form,
                                      syntactic position, co-occurrence patterns, confidence.
                                      Use when statistics alone leave the function unclear.
16. conclude                        — only when all required evidence types are present

Decision guidance:
- If gap says "NO IGT QUANTITATIVE EVIDENCE" → use analyse_tag on the primary diagnostic tag
- If gap says "NO ABSENCE CHECK" → use analyse_absence
- If gap says "NO COUNTER-EVIDENCE SEARCH" → use compare_tags or analyse_absence with
  a competing category (e.g., check ASPECT when investigating TENSE)
- If gap says "UNRESOLVED CONTRADICTIONS" → use find_tag_cluster or analyse_construction
  to gather more context that resolves the conflict
- To validate what a tag MEANS semantically → use analyse_semantic_context(tag):
  check if "past" tags co-occur with "yesterday/ago/before" in translations
- To find constructions BY MEANING rather than tag → use search_translations(keyword)
- To see actual sentence data for structural reasoning → use get_triline_examples(tag_or_keyword)
  THEN parse_example_structure(same_query) to get LLM clause analysis
- To determine if a morpheme is a prefix or suffix → use analyse_morpheme_position(tag)
- To show what a morpheme actually looks like in this language → use get_morpheme_forms(tag)
- When statistics are available but function is unclear → use analyse_tag_usage(tag)
- Do NOT repeat a tool call with the same arguments

Output ONLY valid JSON:
{{
  "thought": "what the current evidence shows, what gap this action addresses, and why",
  "action": "get_tag_inventory|get_construction_inventory|find_tag_cluster|analyse_tag|analyse_construction|analyse_absence|compare_tags|get_section_igt|analyse_semantic_context|search_translations|get_triline_examples|analyse_morpheme_position|parse_example_structure|get_morpheme_forms|analyse_tag_usage|conclude",
  "args": {{}},
  "evidence_type": "igt_quantitative|absence|construction|counter_evidence",
  "claim_to_add": "one sentence claim this observation would support or refute",
  "supports_hypothesis": true
}}"""


IGT_CONCLUDE_PROMPT = """You are writing a typological feature entry inferred from IGT data (no reference grammar available).

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}

COMPLETE EVIDENCE GRAPH:
{evidence_graph}

SEARCH TRACE SUMMARY:
{trace_summary}

═══ WRITING PHILOSOPHY ═══════════════════════════════════════════
The subject of every sentence is the LINGUISTIC PHENOMENON, not the tag.

WRONG: "The IRR tag appears in 8 examples (3.5%)."
RIGHT: "Irrealis mood in {language} is marked by the enclitic '=ri', which appears
        in 8 attested examples — typically in adverbial clauses and conditional
        constructions, often alongside dubitative particles."

The tag (IRR, AFF, FUT…) is merely a label for the phenomenon. Always:
  1. Name the PHENOMENON as the subject
  2. Name the ACTUAL MORPHEME FORM (e.g. '=ri', '-ma', 'ká') as its realization
  3. Describe the CONTEXTS it appears in (clause types, co-occurring elements)
  4. State what function the form serves in those contexts
  5. Give the corpus count as supporting parenthetical, not as the main point
═══════════════════════════════════════════════════════════════════

Evidence weighting:
- Semantic usage analysis (SEMANTIC USAGE ANALYSIS sections above) is PRIMARY — read it first
- IGT_PATTERN claims from tag analysis are supporting evidence
- ABSENCE_EVIDENCE with zero counts is strong negative evidence
- Isolated low-frequency forms → "Partial" or "Unclear", not "Yes"
- Confidence ceiling 0.85 (no grammar prose available)

SELF-CONSISTENCY RULES:
1. If structural_description says a form was found → value must be "Yes" or "Partial"
2. If value is "No" → every key_evidence item must confirm absence
3. If evidence is contradictory → value="Partial" or "Unclear", explain the tension
4. Read structural_description and value together before finalising — they must agree

For each key_evidence item:
  - Open with the PHENOMENON (not the tag): "Irrealis mood is expressed by..."
  - Name the morpheme form: "...the suffix '-ri'..."
  - Describe the usage context: "...in conditional and subjunctive clauses,
    where it follows the verb root and precedes any agreement suffixes"
  - Cite the source tool concisely: "Source: analyse_tag_usage(IRR)"
  - Add corpus count as parenthetical: "(8/226 examples, 3.5%)"
  NEVER open a key_evidence item with "The IRR tag..." or "The tag..."

Output ONLY valid JSON:
{{
  "linguistic_definition": "language-agnostic definition of the phenomenon (1-2 sentences, no tag names)",
  "structural_description": "How {language} expresses this phenomenon: name the morpheme form, where it attaches, what constructions it appears in. Subject = the phenomenon, not the tag.",
  "value": "Yes|No|Partial|Unclear|?",
  "value_detail": "One sentence: '{language} marks [phenomenon] with [morpheme form], attested N times in [construction type]'",
  "confidence": 0.0-0.85,
  "key_evidence": [
    "Irrealis mood in {language} is realized by the enclitic '=ri' (surface form from get_morpheme_forms), which consistently appears after the first constituent of conditional and adverbial clauses (8/226 examples, 3.5%). Source: analyse_tag_usage(IRR). This fixed constructional environment suggests grammaticalization as a mood marker rather than a free adverb.",
    "In translation, sentences containing '=ri' express hypothetical or counter-factual situations ('if...', 'in case...', 'were it to...') in 7 of 8 attested cases. Source: analyse_semantic_context(IRR). The near-perfect alignment between form and hypothetical meaning confirms its irrealis function.",
    "Irrealis and future-marking co-occur in only 1 of 8 irrealis examples, suggesting they occupy different semantic/structural slots rather than being in free variation. Source: compare_tags(IRR, FUT).",
    "No canonical tense/aspect tags (PST, PFV) appear alongside irrealis marking, consistent with cross-linguistic patterns where irrealis operates independently of tense. Source: find_tag_cluster(IRR)."
  ],
  "typological_notes": "3 sentences situating the phenomenon cross-linguistically. End with: 'Conclusion is data-inferred from IGT corpus only; no grammar prose consulted. Human verification recommended.'",
  "needs_human_review": true,
  "review_reason": "what a grammar consultation would add or clarify"
}}
CRITICAL: claim_id values (C001, C005…) are INTERNAL — never cite them. Cite tools instead.
CRITICAL: Maximum 4 items in key_evidence. Subject of each = the phenomenon, not the tag."""


IGT_QUERY_PLAN_PROMPT = """You are planning a free-form query investigation using only IGT data.

LANGUAGE: {language}
QUERY: {query}

{abbrev_legend}

STEP 1 — CANDIDATE TAGS FROM ABBREVIATION REGISTRY (matched to this query):
{abbrev_candidates}

Use these as your primary starting point for igt_tags_to_check:
- Tags listed as IN CORPUS should be your first analyse_tag targets.
- Tags listed as NOT IN CORPUS are useful for analyse_absence (confirming absence).
- Only add tags outside this list if the IGT statistics strongly suggest them.

IGT STATISTICS (for cross-referencing):
{igt_summary}

TOP CONSTRUCTION PATTERNS:
{construction_patterns}

Identify the specific linguistic phenomena the query asks about, then plan how to
investigate using ONLY the 8 IGT tools (no grammar reading).

Output ONLY valid JSON:
{{
  "phenomena": ["phenomenon 1", "phenomenon 2"],
  "rationale": "one sentence: why these tags and tools are relevant",
  "igt_tags_to_check": ["TAG1", "TAG2"],
  "constructions_to_check": [["TAG1", "TAG2"]],
  "cluster_seeds": ["TAG1"],
  "category_absence_check": "CATEGORY",
  "tags_to_compare": [["TAG1", "TAG2"]]
}}"""


IGT_QUERY_DECISION_PROMPT = """You are investigating a linguistic query using only IGT data. No grammar book is available.

LANGUAGE: {language}
QUERY: {query}

EVIDENCE GATHERED SO FAR:
{evidence_summary}

Progress: iteration {iteration}/{max_iter}

AVAILABLE TOOLS:
1.  get_tag_inventory               — full tag list with frequencies
2.  get_construction_inventory      — top bigrams and trigrams
3.  find_tag_cluster(seed_tag)      — tags co-occurring most with seed
4.  analyse_tag(tag)                — deep frequency/position/cooccurrence profile
5.  analyse_construction(tags)      — find ordered sequence in corpus
6.  analyse_absence(category)       — quantify absence
7.  compare_tags(tag_a, tag_b)      — PMI and positional comparison
8.  get_section_igt(query)          — IGT from sections matching query
9.  analyse_semantic_context(tag)   — semantic analysis of translation lines for tag's examples
10. search_translations(query)      — find examples by keyword in translation line
11. get_triline_examples(query)     — aligned MORPHEME/GLOSS/TRANSLATION trilines for a tag,
                                      translation keyword, or section name
12. analyse_morpheme_position(tag)  — is this morpheme a PREFIX, SUFFIX, or STEM?
13. parse_example_structure(query)  — LLM clause analysis: word order, argument structure,
                                      morphological locus from actual sentence trilines
14. get_morpheme_forms(tag)         — show actual surface forms of a morpheme (e.g. '-ma', '=ri')
15. conclude                        — when evidence is sufficient

Output ONLY valid JSON:
{{
  "thought": "what evidence is gathered and what still needs clarification",
  "action": "tool_name or conclude",
  "args": {{}},
  "finding": "one sentence: what this action is expected to reveal"
}}"""


IGT_QUERY_CONCLUDE_PROMPT = """You are answering a typological research query using only IGT data (no grammar available).

LANGUAGE: {language}
QUERY: {query}

COMPLETE EVIDENCE GRAPH:
{evidence_graph}

SEARCH TRACE: {trace_summary}

═══ WRITING PHILOSOPHY ══════════════════════════════════════════
The subject of every sentence is the LINGUISTIC PHENOMENON, not the tag.

WRONG: "The IRR tag appears in 8 examples and co-occurs with DUB."
RIGHT: "Irrealis mood in {language} is realized by the enclitic '=ri', which
        appears in conditional and hypothetical clauses, often alongside
        the dubitative particle."

Tags and morpheme labels are just names for the things you are describing.
Always lead with what the LANGUAGE DOES, then cite the evidence.
═══════════════════════════════════════════════════════════════════

CRITICAL: claim_id values (C001, C005…) are INTERNAL — never write them.

For each key_evidence item:
  - Open with the PHENOMENON: "Future tense in {language} is marked by..."
  - Name the MORPHEME FORM: "...the suffix '-ma'..."
  - Describe the CONTEXT: "...which appears after the verb root in transitive
    and intransitive clauses alike, immediately before any agreement suffixes"
  - Cite the SOURCE: "Source: analyse_tag_usage(FUT)"
  - Add NUMBERS parenthetically: "(49/226 examples, 21.7%)"
  NEVER open with "The FUT tag..." or "The tag..."
  NEVER write "various structures" or "several patterns" — name them.

Output ONLY valid JSON with answer as the FIRST key:
{{
  "answer": "2-4 paragraph prose answer. Subject = the phenomenon. Name morpheme forms (e.g. '-ma', '=ri'). Explain usage contexts and what they reveal about the grammar. Note limits of IGT-only analysis.",
  "linguistic_definition": "language-agnostic definition (no tag names)",
  "structural_description": "How {language} expresses this: morpheme form, attachment site, constructions it appears in. Subject = the phenomenon.",
  "key_evidence": [
    "Future tense in {language} is expressed by the suffix '-ma' (or its clitic variant '=ma'), which attaches to the verb after the root and before agreement morphology. It appears in 49 of 226 corpus examples (21.7%). Source: analyse_tag_usage(FUT). The fixed post-root position suggests grammaticalization as an inflectional suffix rather than a free particle.",
    "Translations of sentences with the future suffix consistently use will/going to/tomorrow in 78% of cases, confirming the morpheme encodes genuine futurity rather than modality or aspect. Source: analyse_semantic_context(FUT).",
    "Future and imperfective aspect almost never co-occur (2/49 future examples contain IMPF), suggesting these categories occupy the same structural slot and may be mutually exclusive. Source: compare_tags(FUT, IMPF).",
    "Past tense is entirely absent from the corpus (0/226 examples), while future is frequent — a typologically common asymmetry in aspect-prominent languages where past is inferred from context. Source: analyse_absence(TENSE)."
  ],
  "confidence": 0.0-0.85,
  "needs_human_review": true,
  "review_reason": "what a grammar consultation would add or clarify"
}}"""


PARSE_STRUCTURE_PROMPT = """You are a linguistic field linguist analyzing sentence structure from IGT (Interlinear Glossed Text) data for {language}.

The IGT examples below show three lines:
  MORPHEME:    segmented morphemes (hyphens = morpheme boundaries within a word)
  GLOSS:       functional label for each morpheme
  TRANSLATION: free translation into English

{trilines}

For each example, identify:
  1. Main predicate: which morpheme(s), what position (clause-initial / medial / final)
  2. Arguments: subject and object — overt NPs or zero-marked via verbal agreement?
  3. Modifiers: adverbial, temporal, adjectival expressions
  4. Clause type: declarative / interrogative / imperative / subordinate

After the per-example analysis, give:
  BASIC WORD ORDER: dominant order of arguments and predicate (SOV / SVO / VSO / flexible / verb-complex)
  MORPHOLOGICAL LOCUS: are grammatical relations encoded on the verb (polysynthetic), on case-marked NPs, or both?
  KEY STRUCTURAL PATTERNS: 2-3 specific recurring patterns with cited morpheme examples

Rules:
  - Be specific: cite actual morphemes (e.g. "the suffix -ma marks subject agreement") not vague labels
  - If the evidence is insufficient to determine something, say so explicitly
  - Do NOT produce JSON — write plain structured prose"""


SEMANTIC_USAGE_PROMPT = """You are a field linguist analysing IGT (Interlinear Glossed Text) data for {language}.

Below are all attested examples containing the morpheme/tag '{tag}' (total: {count} examples).
{trilines}

Your task: infer the GRAMMATICAL FUNCTION and USAGE CONTEXT of '{tag}' from the examples above.

For each example, note:
- What type of clause it appears in (declarative, interrogative, imperative, conditional…)
- What other elements co-occur with '{tag}' (negation, tense, aspect, person…)
- What the translation suggests about its meaning or function

Then provide a SUMMARY covering:
  1. PRIMARY FUNCTION: What does '{tag}' most consistently express? (e.g. evidentiality,
     mood, affirmation, emphasis, topic-marking, etc.)
  2. SECONDARY FUNCTIONS or contexts where the function seems different
  3. MORPHEME FORM: What is the actual surface form? (e.g. "the enclitic =á")
  4. SYNTACTIC POSITION: Where does it appear (clause-initial, preverbal, enclitic to NP…)?
  5. CONFIDENCE: How clear is the pattern across the {count} examples?
     Use: HIGH (consistent function), MEDIUM (mostly consistent), LOW (mixed/unclear)
  6. WHAT IT IS NOT: explicitly state if it does NOT seem to be negation / tense / aspect /
     agreement if those might be confused with it.

Write plain prose. Be specific about morpheme forms and cite example IDs where helpful.
Do NOT produce JSON."""


CLAIM_EXTRACTION_PROMPT = """You are extracting typed claims from an IGT observation.

LANGUAGE: {language}
FEATURE QUESTION: {question}
OBSERVATION:
{observation}

Extract 1-2 specific, falsifiable claims. Use EXACTLY ONE of these type strings:
  igt_pattern       = pattern from tag frequency, position, or co-occurrence data
  absence_evidence  = confirms a category is absent or ungrammaticalized
  inference         = derived by combining multiple evidence pieces
  counter_evidence  = contradicts the working hypothesis
  author_caveat     = data is ambiguous or contradictory

Output ONLY valid JSON:
{{
  "claims": [
    {{
      "text": "one sentence stating the claim",
      "type": "igt_pattern",
      "confidence": 0.85,
      "supports_hypothesis": true,
      "igt_examples": []
    }}
  ]
}}"""


# ═══════════════════════════════════════════════════════════════
# IGT GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════

def _igt_gap_analysis(graph: EvidenceGraph) -> str:
    """
    Gap analysis adapted for IGT-only mode.
    GRAMMAR_STATEMENT is not required; IGT_PATTERN and ABSENCE_EVIDENCE are.
    """
    has_igt_pattern = any(
        c.claim_type == ClaimType.IGT_PATTERN
        for c in graph.claims.values()
    )
    has_absence = any(
        c.claim_type == ClaimType.ABSENCE_EVIDENCE
        for c in graph.claims.values()
    )
    has_counter = any(
        c.supports_hypothesis is False
        for c in graph.claims.values()
    )
    unresolved = [c for c in graph.contradictions if not c.resolved]

    gaps = []
    if not has_igt_pattern:
        gaps.append(
            "NO IGT QUANTITATIVE EVIDENCE: Use analyse_tag or get_construction_inventory "
            "to ground claims in data"
        )
    if not has_absence:
        gaps.append(
            "NO ABSENCE CHECK: Use analyse_absence to verify whether the category "
            "is confirmed absent in the corpus"
        )
    if not has_counter:
        gaps.append(
            "NO COUNTER-EVIDENCE SEARCH: Use compare_tags or analyse_absence with "
            "a competing category to test the alternative hypothesis"
        )
    if unresolved:
        gaps.append(
            f"UNRESOLVED CONTRADICTIONS ({len(unresolved)}): "
            + "; ".join(c.description[:80] for c in unresolved[:2])
        )

    if not gaps:
        return "Evidence is comprehensive: IGT patterns, absence check, and counter-evidence all present."
    return "EVIDENCE GAPS:\n" + "\n".join(f"  - {g}" for g in gaps)


def _constraints_met(graph: EvidenceGraph) -> str:
    has_igt  = any(c.claim_type == ClaimType.IGT_PATTERN for c in graph.claims.values())
    has_abs  = any(c.claim_type == ClaimType.ABSENCE_EVIDENCE for c in graph.claims.values())
    has_ctr  = any(c.supports_hypothesis is False for c in graph.claims.values())
    n_claims = len(graph.claims)
    return (
        f"IGT_pattern={'✓' if has_igt else '✗'}  "
        f"absence={'✓' if has_abs else '✗'}  "
        f"counter={'✓' if has_ctr else '✗'}  "
        f"claims={n_claims}"
    )


def _all_constraints_met(graph: EvidenceGraph, min_queries: int = 4) -> bool:
    has_igt = any(c.claim_type == ClaimType.IGT_PATTERN for c in graph.claims.values())
    has_abs = any(c.claim_type == ClaimType.ABSENCE_EVIDENCE for c in graph.claims.values())
    has_ctr = any(c.supports_hypothesis is False for c in graph.claims.values())
    n_claims = len(graph.claims)
    return has_igt and has_abs and has_ctr and n_claims >= min_queries


# ═══════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════

class IGTOnlyResearchAgent:
    """
    Typological research agent that works entirely from IGT data.
    No reference grammar is required or used.
    """

    def __init__(
        self,
        language: str,
        igt_path: str,
        llm,
        max_iterations_per_feature: int = 12,
        confidence_threshold: float = 0.65,
        min_queries_per_feature: int = 4,
        abbreviations_path: Optional[str] = None,    # ← NEW
    ):
        self.language    = language
        self.llm         = llm
        self.max_iter    = max_iterations_per_feature
        self.conf_thresh = confidence_threshold
        self.min_queries = min_queries_per_feature

        self.toolkit = self._build_toolkit(igt_path, abbreviations_path)  # ← NEW arg

        if not self.toolkit.igt_analyser:
            raise ValueError(f"IGT file could not be loaded from: {igt_path}")

        self._abbrev            = self.toolkit.abbrev   # ← NEW: convenience shortcut
        self._igt_summary       = self.toolkit.igt_analyser.get_stats().summary_text
        # Enrich the cached summary with abbreviation annotations
        self._igt_summary       = self._abbrev.enrich_igt_summary(self._igt_summary)  # ← NEW
        self._construction_text = self._get_construction_summary()

    # ── Toolkit construction ──────────────────────────────────────

    def _build_toolkit(
        self,
        igt_path: str,
        abbreviations_path: Optional[str] = None,    # ← NEW
    ) -> DeepGrammarToolkit:
        """
        Build a DeepGrammarToolkit with IGT only.
        """
        import tempfile, json
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([], f)
            tmp_path = f.name

        toolkit = DeepGrammarToolkit(
            grammar_path=tmp_path,
            igt_path=igt_path,
            abbreviations_path=abbreviations_path,   # ← NEW
        )
        toolkit.chunks = []
        toolkit._chunk_by_id  = {}
        toolkit.section_reader = None
        Path(tmp_path).unlink(missing_ok=True)
        return toolkit

    def _get_construction_summary(self) -> str:
        stats = self.toolkit.igt_analyser.get_stats()
        lines = []
        for cp in stats.constructions[:20]:
            if len(cp.tags) == 2:
                trans = cp.typical_translations[0][:60] if cp.typical_translations else ""
                # ← NEW: annotate tags with meanings
                a = self._abbrev.expand(cp.tags[0])
                b = self._abbrev.expand(cp.tags[1])
                lines.append(f"  {a}→{b}: {cp.count}x  e.g. '{trans}'")
        return "\n".join(lines) if lines else "(no construction patterns)"

    # ── Main pipeline ─────────────────────────────────────────────

    def run(self) -> EpistemicState:
        state = EpistemicState(language=self.language)

        _print(f"\n[IGT-only] Discovering domains for {self.language}...")
        domains = self._discover_domains()
        if not domains:
            _print("  [WARN] No domains discovered from IGT — check corpus quality")
            return state
        state.domains = domains
        _print(f"  Found {len(domains)} domains", indent=2)

        for domain in domains:
            _print(f"\n  Domain: {domain['domain_name']}", indent=2)
            for feat in domain.get("candidate_features", []):
                _print(f"    Feature: {feat['question'][:80]}", indent=4)
                result = self._investigate_feature(feat, domain)
                if result:
                    if result.confidence >= self.conf_thresh:
                        state.confirmed_features.append(result)
                    else:
                        state.uncertain_features.append(result)

        return state

    # ── Domain discovery ─────────────────────────────────────────

    def _discover_domains(self) -> list:
        # ← NEW: inject full abbreviation legend
        abbrev_legend = self._abbrev.prompt_legend()
        prompt = IGT_DOMAIN_EXTRACTION_PROMPT.format(
            language=self.language,
            abbrev_legend=abbrev_legend,
            igt_summary=self._igt_summary,
            construction_patterns=self._construction_text,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=2048)
        data = self._parse_json(raw)
        return data.get("domains", [])

    # ── Feature investigation ─────────────────────────────────────

    def _investigate_feature(self, feat: dict, domain: dict) -> Optional[Feature]:
        question   = feat["question"]
        feature_id = feat.get("feature_id", "F???")
        igt_signals = domain.get("igt_signals", [])

        self.llm.reset_token_counter()   # start fresh for this feature
        graph = EvidenceGraph(question)
        trace = []

        plan = self._plan_feature(feat, domain)
        min_q = plan.get("min_queries_before_conclude", self.min_queries)
        _print(f"      Plan: check tags {plan.get('igt_tags_to_check', [])}", indent=6)

        self._execute_plan(plan, graph, trace, question)

        for iteration in range(1, self.max_iter + 1):
            if _all_constraints_met(graph, min_q) and iteration >= min_q:
                _print(f"      → All constraints met at iteration {iteration}", indent=6)
                break

            decision = self._decide_next_action(
                question=question,
                domain=domain["domain_name"],
                graph=graph,
                trace=trace,
                iteration=iteration,
            )
            if not decision:
                break
            action = decision.get("action", "")
            if action == "conclude":
                break

            res = self._execute_tool(action, decision.get("args", {}), question)
            if res.text:
                _print(f"      [iter {iteration}] {action}({list(decision.get('args',{}).values())[:1]})")
                _print(f"        thought: {decision.get('thought','')[:120]}")
                _print(f"        obs:")
                for line in res.text.splitlines()[:20]:
                    _print(f"          {line}")
                trace.append({"action": action, "args": decision.get("args", {}), "observation": res.text[:400]})
                self._add_to_graph(graph, res.text, question, decision, igt_ids=res.igt_ids)
                self._resolve_contradictions(graph, question)

        return self._conclude_feature(feat, domain, graph, trace)

    # ── Abbreviation-based tag candidate lookup ───────────────────

    _KW_STOP = frozenset({
        'the','and','that','this','with','for','from','have','his','her',
        'him','they','their','its','was','are','but','not','has','had',
        'into','what','who','where','when','why','how','which','does',
        'did','will','would','could','should','may','might','must','can',
        'been','being','different','element','expressing','language',
        'feature','marker','form','used','use','also','only','such',
        'does','grammar','case','tense','aspect','verb','noun','clause',
        'sentence','construction','marking','marked','grammatical',
    })

    def _find_tags_via_translations(self, query_text: str, top_n: int = 6) -> tuple:
        """
        Fallback tag discovery when abbreviation lookup finds no candidates.
        Extracts keywords from query_text, searches translation lines,
        and returns diagnostic tags from matching examples.
        Returns (diagnostic_tags, terms_hit, total_hits).
        """
        if not self.toolkit.igt_analyser:
            return [], [], 0

        stats    = self.toolkit.igt_analyser.get_stats()
        universal = {
            t for t, p in stats.tag_profiles.items()
            if p.example_coverage > 0.30
        }
        lang_tokens = {
            tok.lower()
            for tok in re.split(r'\W+', self.language)
            if len(tok) >= 3
        }
        stop = self._KW_STOP | lang_tokens

        phrase_pats = [
            r'\bthere (?:is|are|was|were)\b',
            r'\bis a\b', r'\bare a\b', r'\bwas a\b', r'\bwere a\b',
            r'\bwill be\b', r'\bcan be\b', r'\bbecame\b',
        ]
        phrases = []
        for pat in phrase_pats:
            for m in re.finditer(pat, query_text.lower()):
                phrases.append(m.group())

        words        = re.findall(r"[a-z]{3,}", query_text.lower())
        kws          = list(dict.fromkeys(w for w in words if w not in stop))
        search_terms = list(dict.fromkeys(phrases + kws))[:8]
        if not search_terms:
            return [], [], 0

        all_tags   = Counter()
        total_hits = 0
        terms_hit  = []
        examples   = self.toolkit.igt_examples

        for term in search_terms:
            matches = [ex for ex in examples if term in ex.translation.lower()]
            if not matches:
                continue
            total_hits += len(matches)
            terms_hit.append(f"'{term}':{len(matches)}")
            for ex in matches:
                for t in ex.gloss_tags:
                    all_tags[t] += 1

        if not all_tags:
            return [], search_terms, 0

        diagnostic = [
            (t, c) for t, c in all_tags.most_common(20)
            if t not in universal
            and c >= 2
            and c / max(total_hits, 1) < 0.65
            and len(t) >= 2
        ][:top_n]

        return diagnostic, terms_hit, total_hits

    def _format_abbrev_candidates(self, query_text: str) -> str:
        """
        Reverse-lookup abbreviation registry for tags relevant to query_text.
        When no abbreviation matches, falls back to translation-line search:
        finds examples via keywords, surfaces their tags, and tells the LLM
        to start with search_translations rather than analyse_tag.
        """
        candidates = self._abbrev.find_tags_for_phenomenon(query_text, top_n=10)
        stats = self.toolkit.igt_analyser.get_stats() if self.toolkit.igt_analyser else None

        # ── Primary path: abbreviation lookup succeeded ──────────
        if candidates:
            lines = []
            for tag, meaning in candidates:
                if stats and tag in stats.tag_profiles:
                    p = stats.tag_profiles[tag]
                    corpus_note = f"IN CORPUS: {p.count}x ({p.example_coverage*100:.1f}%)"
                elif stats:
                    corpus_note = "NOT IN CORPUS (0 occurrences) — useful for absence check"
                else:
                    corpus_note = ""
                lines.append(f"  {tag:<12} {meaning:<30}  {corpus_note}")
            return "\n".join(lines)

        # ── Fallback: translation-line search ────────────────────
        diag_tags, terms_hit, total_hits = self._find_tags_via_translations(query_text)

        if not diag_tags:
            return (
                "(no matches found in abbreviation registry or translation lines — "
                "use get_tag_inventory to explore the full tag set)"
            )

        lines = [
            "NOTE: No direct abbreviation matches found for this phenomenon.",
            f"TRANSLATION SEARCH found {total_hits} relevant examples "
            f"via: {', '.join(terms_hit[:5])}",
            "Tags appearing in those examples:",
        ]
        if stats:
            for tag, count in diag_tags:
                p       = stats.tag_profiles.get(tag)
                pct     = f"{p.example_coverage*100:.1f}%" if p else "?"
                meaning = self._abbrev.label(tag)
                label   = f"{tag} ({meaning})" if meaning != tag else tag
                lines.append(
                    f"  {label:<28}  found in {count} matching examples ({pct} overall)"
                )
        else:
            for tag, count in diag_tags:
                lines.append(f"  {tag:<12}  {count} matching examples")

        lines.append(
            "IMPORTANT: Start with search_translations(keyword) to retrieve "
            "the relevant sentences, then use get_triline_examples + "
            "parse_example_structure to analyse their structure."
        )
        return "\n".join(lines)

    # ── Plan execution ────────────────────────────────────────────

    def _plan_feature(self, feat: dict, domain: dict) -> dict:
        phenomenon_text   = feat["question"] + " " + " ".join(domain.get("igt_signals", []))
        abbrev_candidates = self._format_abbrev_candidates(phenomenon_text)

        # Check whether we're in translation-fallback mode
        # (abbrev found nothing; candidates come from translation search)
        trans_fallback = abbrev_candidates.startswith("NOTE: No direct abbreviation")

        candidate_tags = [
            tag for tag, _ in self._abbrev.find_tags_for_phenomenon(phenomenon_text, top_n=8)
        ]
        legend_tags   = candidate_tags if candidate_tags else feat.get("igt_tags_to_check", [])
        abbrev_legend = self._abbrev.prompt_legend(legend_tags) if legend_tags else self._abbrev.prompt_legend()

        prompt = IGT_FEATURE_PLAN_PROMPT.format(
            language=self.language,
            question=feat["question"],
            domain=domain["domain_name"],
            igt_signals=", ".join(domain.get("igt_signals", [])),
            abbrev_legend=abbrev_legend,
            abbrev_candidates=abbrev_candidates,
            igt_summary=self._igt_summary,
            construction_patterns=self._construction_text,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=512)
        plan = self._parse_json(raw)

        if trans_fallback:
            # In translation-fallback mode: tag-based tools are secondary.
            # Extract search keywords and store them so _execute_plan can
            # call search_translations first.
            _, terms_hit, _ = self._find_tags_via_translations(phenomenon_text)
            # terms_hit are like "'there is':3", extract the keyword part
            plan["_translation_search_terms"] = [
                t.strip("'").split("'")[0] for t in terms_hit
            ][:4]
            plan.setdefault("igt_tags_to_check", [])
        else:
            # Post-process: if LLM ignored abbrev candidates, override
            if candidate_tags:
                plan_tags  = [t.upper() for t in plan.get("igt_tags_to_check", [])]
                cand_upper = [t.upper() for t in candidate_tags]
                overlap    = [t for t in plan_tags if t in cand_upper]
                if not overlap:
                    _print(
                        f"      ⚠ plan tags {plan_tags} don't match abbrev candidates "
                        f"{cand_upper[:5]} — overriding",
                        indent=6,
                    )
                    plan["igt_tags_to_check"] = candidate_tags[:5]
            plan.setdefault("igt_tags_to_check", candidate_tags[:5] if candidate_tags else [])

        plan.setdefault("_translation_search_terms", [])
        plan.setdefault("constructions_to_check", [])
        plan.setdefault("cluster_seeds", [])
        plan.setdefault("category_absence_check", "")
        plan.setdefault("tags_to_compare", [])
        plan.setdefault("min_queries_before_conclude", self.min_queries)
        return plan

    def _execute_plan(self, plan: dict, graph: EvidenceGraph, trace: list, question: str):
        _print("      [PLAN] Initial tool calls:")

        # ── Translation-fallback mode ─────────────────────────────
        # When abbreviation lookup found nothing, we go via translation
        # search → trilines → structural parsing instead of analyse_tag.
        trans_terms = plan.get("_translation_search_terms", [])
        if trans_terms:
            _print(f"        [translation-fallback mode: {trans_terms}]")
            for term in trans_terms[:3]:
                _print(f"        -> search_translations('{term}')")
                res = self._call_tool("search_translations", {"query": term})
                if res.text and "No translation" not in res.text:
                    for _l in res.text.splitlines()[:15]:
                        _print(f"           {_l}")
                    trace.append({"action": "search_translations",
                                  "args": {"query": term}, "observation": res.text[:500]})
                    self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN,
                                           f"search_translations('{term}')", igt_ids=res.igt_ids)

            # Follow up with trilines + structural parse on the first term
            if trans_terms:
                first = trans_terms[0]
                _print(f"        -> get_triline_examples('{first}')")
                res = self._call_tool("get_triline_examples", {"query": first})
                if res.text and "No examples" not in res.text:
                    for _l in res.text.splitlines()[:20]:
                        _print(f"           {_l}")
                    trace.append({"action": "get_triline_examples",
                                  "args": {"query": first}, "observation": res.text[:600]})
                    self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN,
                                           f"get_triline_examples('{first}')", igt_ids=res.igt_ids)

                    _print(f"        -> parse_example_structure('{first}')")
                    res2 = self._call_tool("parse_example_structure", {"query": first})
                    if res2.text:
                        for _l in res2.text.splitlines()[:25]:
                            _print(f"           {_l}")
                        trace.append({"action": "parse_example_structure",
                                      "args": {"query": first}, "observation": res2.text[:800]})
                        self._add_simple_claim(graph, res2.text, ClaimType.IGT_PATTERN,
                                               f"parse_example_structure('{first}')",
                                               igt_ids=res2.igt_ids)
            return   # Skip the tag-based tools below

        # ── Normal (abbreviation-match) mode ─────────────────────
        for tag in plan.get("igt_tags_to_check", [])[:3]:
            _print(f"        -> analyse_tag({tag})")
            res = self._call_tool("analyse_tag", {"tag": tag})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"           {_l}")
                trace.append({"action": "analyse_tag", "args": {"tag": tag}, "observation": res.text[:400]})
                self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"analyse_tag({tag})", igt_ids=res.igt_ids)

        for seed in plan.get("cluster_seeds", [])[:2]:
            _print(f"        -> find_tag_cluster({seed})")
            res = self._call_tool("find_tag_cluster", {"seed_tag": seed})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"           {_l}")
                trace.append({"action": "find_tag_cluster", "args": {"seed_tag": seed}, "observation": res.text[:400]})
                self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"find_tag_cluster({seed})", igt_ids=res.igt_ids)

        for tags in plan.get("constructions_to_check", [])[:2]:
            _print(f"        -> analyse_construction({tags})")
            res = self._call_tool("analyse_construction", {"tags": tags})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"           {_l}")
                trace.append({"action": "analyse_construction", "args": {"tags": tags}, "observation": res.text[:400]})
                self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"analyse_construction({tags})", igt_ids=res.igt_ids)

        if plan.get("category_absence_check"):
            cat = plan["category_absence_check"]
            _print(f"        -> analyse_absence({cat})")
            res = self._call_tool("analyse_absence", {"category": cat})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"           {_l}")
                trace.append({"action": "analyse_absence", "args": {"category": cat}, "observation": res.text[:400]})
                # Only add as ABSENCE_EVIDENCE if the category was actually checkable
                if "CANNOT ASSESS" in res.text:
                    _print(f"           → skipped (unknown cluster — not absence evidence)", indent=6)
                else:
                    self._add_simple_claim(graph, res.text, ClaimType.ABSENCE_EVIDENCE,
                                           f"analyse_absence({cat})", igt_ids=res.igt_ids)

        for pair in plan.get("tags_to_compare", [])[:2]:
            if len(pair) == 2:
                _print(f"        -> compare_tags({pair[0]}, {pair[1]})")
                res = self._call_tool("compare_tags", {"tag_a": pair[0], "tag_b": pair[1]})
                if res.text:
                    for _l in res.text.splitlines()[:15]: _print(f"           {_l}")
                    trace.append({"action": "compare_tags", "args": {"tag_a": pair[0], "tag_b": pair[1]}, "observation": res.text[:400]})
                    self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"compare_tags({pair[0]},{pair[1]})", igt_ids=res.igt_ids)

    # ── ReAct decision ────────────────────────────────────────────

    def _decide_next_action(
        self, question: str, domain: str, graph: EvidenceGraph,
        trace: list, iteration: int,
    ) -> dict:
        prompt = IGT_SEARCH_DECISION_PROMPT.format(
            language=self.language,
            question=question,
            domain=domain,
            evidence_summary=graph.summarize(),
            gap_analysis=_igt_gap_analysis(graph),
            iteration=iteration,
            max_iter=self.max_iter,
            constraints_met=_constraints_met(graph),
        )
        raw = self.llm.generate(prompt, max_new_tokens=512)
        return self._parse_json(raw)

    # ── Tool dispatch ─────────────────────────────────────────────

    IGT_TOOLS = {
        "get_tag_inventory", "get_construction_inventory", "find_tag_cluster",
        "analyse_tag", "analyse_construction", "analyse_absence",
        "compare_tags", "get_section_igt",
        "analyse_semantic_context", "search_translations",
        "get_triline_examples", "analyse_morpheme_position",
        "parse_example_structure", "get_morpheme_forms",
        "analyse_tag_usage",
    }

    def _execute_tool(self, action: str, args: dict, question: str):
        if action not in self.IGT_TOOLS:
            from deep_tools import _result
            return _result("")
        return self._call_tool(action, args)

    def _call_tool(self, action: str, args: dict) -> str:
        tk = self.toolkit
        try:
            if action == "get_tag_inventory":
                result = tk.get_tag_inventory()
            elif action == "get_construction_inventory":
                result = tk.get_construction_inventory()
            elif action == "find_tag_cluster":
                result = tk.find_tag_cluster(args.get("seed_tag", ""))
            elif action == "analyse_tag":
                result = tk.analyse_tag(args.get("tag", ""))
            elif action == "analyse_construction":
                result = tk.analyse_construction(args.get("tags", []))
            elif action == "analyse_absence":
                result = tk.analyse_absence(args.get("category", ""))
            elif action == "compare_tags":
                result = tk.compare_tags(args.get("tag_a", ""), args.get("tag_b", ""))
            elif action == "get_section_igt":
                result = tk.get_section_igt(args.get("section_query", ""))
            elif action == "analyse_semantic_context":
                result = tk.analyse_semantic_context(args.get("tag", ""))
            elif action == "search_translations":
                result = tk.search_translations(
                    args.get("query", ""),
                    args.get("max_results", 15),
                )
            elif action == "get_triline_examples":
                result = tk.get_triline_examples(
                    args.get("query", ""),
                    args.get("max_examples", 8),
                )
            elif action == "analyse_morpheme_position":
                result = tk.analyse_morpheme_position(args.get("tag", ""))
            elif action == "get_morpheme_forms":
                result = tk.get_morpheme_forms(args.get("tag", ""))
            elif action == "analyse_tag_usage":
                # LLM-powered — handled directly
                return self._analyse_tag_usage_llm(
                    args.get("tag", ""),
                    args.get("max_examples", 10),
                )
            elif action == "parse_example_structure":
                # LLM-powered — handled in _execute_tool, not here
                return self._parse_example_structure(
                    args.get("query", ""),
                    args.get("max_examples", 5),
                )
            else:
                return ""
            if hasattr(result, "text"):
                return result
            from deep_tools import _result
            return _result(str(result))
        except Exception as e:
            logger.warning(f"Tool {action} failed: {e}")
            from deep_tools import _result
            return _result("")

    # ── Evidence graph helpers ────────────────────────────────────

    def _add_simple_claim(
        self, graph: EvidenceGraph, observation: str,
        claim_type: ClaimType, source: str,
        supports: Optional[bool] = None,
        confidence: float = 0.7,
        igt_ids: list = None,
    ):
        graph.add_claim(
            text=observation[:200],
            claim_type=claim_type,
            source=source,
            confidence=confidence,
            supports_hypothesis=supports,
            raw_evidence=observation,
            igt_examples=igt_ids or [],
        )

    def _add_to_graph(
        self, graph: EvidenceGraph, observation: str,
        question: str, decision: dict, igt_ids: list = None,
    ):
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            language=self.language,
            question=question,
            observation=observation[:1000],
        )
        raw  = self.llm.generate(prompt, max_new_tokens=256)
        data = self._parse_json(raw)

        ev_type_map = {
            "igt_quantitative": ClaimType.IGT_PATTERN,
            "absence":          ClaimType.ABSENCE_EVIDENCE,
            "construction":     ClaimType.IGT_PATTERN,
            "counter_evidence": ClaimType.COUNTER_EVIDENCE,
        }
        default_type = ev_type_map.get(
            decision.get("evidence_type", ""), ClaimType.INFERENCE
        )

        for claim_data in data.get("claims", []):
            type_str = claim_data.get("type", "")
            ct_map = {
                "igt_pattern":      ClaimType.IGT_PATTERN,
                "absence_evidence": ClaimType.ABSENCE_EVIDENCE,
                "inference":        ClaimType.INFERENCE,
                "counter_evidence": ClaimType.COUNTER_EVIDENCE,
                "author_caveat":    ClaimType.AUTHOR_CAVEAT,
            }
            claim_type = ct_map.get(type_str, default_type)
            llm_ids  = [e for e in claim_data.get("igt_examples", []) if isinstance(e, str)]
            safe_ids = [e for e in (igt_ids or []) if isinstance(e, str)]
            combined = list(dict.fromkeys(safe_ids + llm_ids))
            graph.add_claim(
                text=claim_data.get("text", observation[:80]),
                claim_type=claim_type,
                source=f"{decision.get('action', '?')}({decision.get('args', {})})",
                confidence=float(claim_data.get("confidence", 0.65)),
                supports_hypothesis=claim_data.get("supports_hypothesis"),
                igt_examples=combined,
                raw_evidence=observation,
            )

    def _analyse_tag_usage_llm(self, tag: str, max_examples: int = 10):
        """
        LLM-powered semantic usage analysis for a specific tag.

        1. Fetches trilines for all examples containing the tag (up to max_examples).
        2. Asks the LLM to infer grammatical function from the actual sentences:
           primary function, morpheme form, syntactic position, co-occurrence patterns.
        3. Returns a ToolResult whose text is the LLM's functional analysis.

        This goes beyond statistical profiling (analyse_tag) by reading
        what the morpheme actually *does* in context.
        """
        from deep_tools import _result as mk_result

        # Get trilines for this tag
        triline_result = self._call_tool(
            "get_triline_examples",
            {"query": tag, "max_examples": max_examples},
        )
        if not triline_result.text or "No examples found" in triline_result.text:
            return mk_result(
                f"_analyse_tag_usage_llm: no triline examples found for '{tag}'."
            )

        # Count total examples in corpus
        tag_u = tag.upper()
        count = sum(1 for ex in self.toolkit.igt_examples if tag_u in ex.gloss_tags)

        prompt = SEMANTIC_USAGE_PROMPT.format(
            language=self.language,
            tag=tag,
            count=count,
            trilines=triline_result.text,
        )
        analysis = self.llm.generate(prompt, max_new_tokens=800, json_mode=False)

        combined = (
            f"SEMANTIC USAGE ANALYSIS  (tag='{tag}', {count} total in corpus, "
            f"showing {min(count, max_examples)})\n\n"
            f"{triline_result.text}\n"
            f"{'─'*64}\n"
            f"LLM FUNCTIONAL ANALYSIS\n\n"
            f"{analysis}"
        )
        return mk_result(combined, triline_result.igt_ids)

    def _parse_example_structure(self, query: str, max_examples: int = 5):
        """
        LLM-powered clause structure analysis.
        1. Fetch trilines for the query from the toolkit.
        2. Ask the LLM to identify subject/predicate/object, word order,
           morphological locus, and key structural patterns.
        Returns a ToolResult whose text is the LLM's structural analysis.
        """
        from deep_tools import _result as mk_result

        triline_result = self._call_tool(
            "get_triline_examples",
            {"query": query, "max_examples": max_examples},
        )
        if not triline_result.text or "No examples found" in triline_result.text:
            return mk_result(
                f"parse_example_structure: no examples found for '{query}'."
            )

        prompt = PARSE_STRUCTURE_PROMPT.format(
            language=self.language,
            trilines=triline_result.text,
        )
        analysis = self.llm.generate(prompt, max_new_tokens=900, json_mode=False)

        combined = (
            f"STRUCTURAL ANALYSIS  (query='{query}', "
            f"{max_examples} examples)\n\n"
            f"{triline_result.text}\n"
            f"{'─'*64}\n"
            f"LLM STRUCTURAL ANALYSIS\n\n"
            f"{analysis}"
        )
        return mk_result(combined, triline_result.igt_ids)

    def _resolve_contradictions(self, graph: EvidenceGraph, question: str):
        for contra in graph.contradictions:
            if contra.resolved:
                continue
            ca = graph.claims.get(contra.claim_a_id)
            cb = graph.claims.get(contra.claim_b_id)
            if not ca or not cb:
                continue
            if abs(ca.confidence - cb.confidence) > 0.25:
                weaker = ca if ca.confidence < cb.confidence else cb
                weaker.supports_hypothesis = None
                contra.resolved   = True
                contra.resolution = (
                    f"Auto-resolved: weaker claim ({weaker.claim_id}, "
                    f"conf={weaker.confidence:.2f}) demoted to neutral"
                )

    # ── Conclusion ────────────────────────────────────────────────

    def _conclude_feature(
        self, feat: dict, domain: dict,
        graph: EvidenceGraph, trace: list,
    ) -> Optional[Feature]:
        question   = feat["question"]
        feature_id = feat.get("feature_id", "F???")

        _, agg_conf, igt_note = graph.aggregate_confidence()

        trace_summary = "; ".join(
            f"{t['action']}({list(t['args'].values())[:1]})" for t in trace[-8:]
        )

        # Collect tags from the evidence graph that have actual IGT_PATTERN claims
        # and run semantic usage analysis for the primary ones (up to 2).
        from evidence_graph import ClaimType as _CT
        analysed_tags = []
        semantic_notes = []
        for claim in graph.claims.values():
            if claim.claim_type == _CT.IGT_PATTERN and claim.source:
                # Extract tag from source like "analyse_tag(AFF)" or "analyse_tag({'tag': 'AFF'})"
                m = re.search(r"analyse_tag\(?['\"]?([A-Z][A-Z0-9.]*)['\"]?\)?", claim.source)
                if m:
                    t = m.group(1)
                    if t not in analysed_tags:
                        analysed_tags.append(t)

        for tag in analysed_tags[:2]:
            tag_u = tag.upper()
            count = sum(1 for ex in self.toolkit.igt_examples if tag_u in ex.gloss_tags)
            if count == 0:
                continue
            _print(f"       running semantic usage analysis for '{tag}' ({count} examples)...", indent=4)
            usage_result = self._analyse_tag_usage_llm(tag, max_examples=min(count, 8))
            if usage_result.text and "no triline examples" not in usage_result.text:
                semantic_notes.append(
                    f"\n--- SEMANTIC USAGE ANALYSIS: {tag} ---\n{usage_result.text[:1200]}"
                )

        semantic_block = "\n".join(semantic_notes) if semantic_notes else ""

        prompt = IGT_CONCLUDE_PROMPT.format(
            language=self.language,
            question=question,
            domain=domain["domain_name"],
            evidence_graph=graph.summarize() + (
                f"\n\n{semantic_block}" if semantic_block else ""
            ),
            trace_summary=trace_summary,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=1024)
        data = self._parse_json(raw)
        if not data:
            return None

        data = self._sanitize_output(data)
        data = self._fix_value_consistency(data, graph)

        value      = data.get("value", "Unclear")
        confidence = float(data.get("confidence", agg_conf))

        audit_verdict, audit_obj, revised_val, revised_conf = self._audit(
            question, value, data.get("value_detail", ""), confidence,
            graph.summarize(),
        )
        if audit_verdict in ("weakened", "overturned"):
            value      = revised_val
            confidence = revised_conf

        igt_ids   = graph.get_igt_example_ids()
        igt_notes = graph.get_igt_example_notes()
        igt_used  = self.toolkit.lookup_igt_examples(igt_ids, notes=igt_notes)

        feat_result = Feature(
            feature_id=feature_id,
            question=question,
            domain=domain["domain_name"],
            linguistic_definition=data.get("linguistic_definition", ""),
            structural_description=data.get("structural_description", ""),
            value=value,
            value_detail=data.get("value_detail", ""),
            confidence=confidence,
            key_evidence=data.get("key_evidence", []),
            igt_examples_used=igt_used,
            igt_support=True,
            search_trace=trace,
            typological_notes=data.get("typological_notes", ""),
            needs_human_review=data.get("needs_human_review", True),
            review_reason=data.get("review_reason", "IGT-only conclusion; grammar verification recommended"),
            audit_verdict=audit_verdict,
            audit_objections=audit_obj,
            token_usage=self.llm.get_token_counts(),
        )

        _print("\n" + "="*60)
        _print(f"RESULT  [{feat_result.feature_id}]  {feat_result.question[:60]}")
        _print("="*60)
        _print(json.dumps(feat_result.to_dict(), ensure_ascii=False, indent=2))
        _print("="*60 + "\n")

        return feat_result

    def _audit(
        self, question: str, value: str, value_detail: str,
        confidence: float, evidence_summary: str,
    ) -> tuple:
        from deep_agent import AUDITOR_PROMPT
        prompt = AUDITOR_PROMPT.format(
            language=self.language,
            question=question,
            value=value,
            value_detail=value_detail,
            confidence=confidence,
            evidence_graph_summary=evidence_summary,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=512)
        data = self._parse_json(raw)
        if not data:
            return "upheld", [], value, confidence

        return (
            data.get("verdict", "upheld"),
            data.get("objections", []),
            data.get("revised_value", value),
            float(data.get("revised_confidence", confidence)),
        )

    # ── Free-form query mode ──────────────────────────────────────

    def answer_query(self, query: str, max_iterations: int = 10):
        from state import QueryResult

        self.llm.reset_token_counter()   # start fresh for this query
        graph = EvidenceGraph(query)
        trace = []

        plan = self._plan_query(query)
        self._execute_query_plan(plan, graph, trace, query)

        for iteration in range(1, max_iterations + 1):
            prompt = IGT_QUERY_DECISION_PROMPT.format(
                language=self.language,
                query=query,
                evidence_summary=graph.summarize(),
                iteration=iteration,
                max_iter=max_iterations,
            )
            raw      = self.llm.generate(prompt, max_new_tokens=512)
            decision = self._parse_json(raw)
            if not decision:
                break
            action = decision.get("action", "")
            if action == "conclude":
                break
            res = self._execute_tool(action, decision.get("args", {}), query)
            if res.text:
                _print(f"    [iter {iteration}] {action}({list(decision.get('args',{}).values())[:1]})")
                _print(f"      thought: {decision.get('thought','')[:120]}")
                _print(f"      obs:")
                for line in res.text.splitlines()[:20]:
                    _print(f"        {line}")
                trace.append({"action": action, "args": decision.get("args", {}), "observation": res.text[:400]})
                self._add_to_graph(graph, res.text, query, decision, igt_ids=res.igt_ids)

        _print("    [concluding...]")
        return self._conclude_query(query, graph, trace)

    def _plan_query(self, query: str) -> dict:
        abbrev_candidates = self._format_abbrev_candidates(query)
        trans_fallback    = abbrev_candidates.startswith("NOTE: No direct abbreviation")
        abbrev_legend     = self._abbrev.prompt_legend()
        prompt = IGT_QUERY_PLAN_PROMPT.format(
            language=self.language,
            query=query,
            abbrev_legend=abbrev_legend,
            abbrev_candidates=abbrev_candidates,
            igt_summary=self._igt_summary,
            construction_patterns=self._construction_text,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=512)
        plan = self._parse_json(raw)

        if trans_fallback:
            _, terms_hit, _ = self._find_tags_via_translations(query)
            plan["_translation_search_terms"] = [
                t.strip("'").split("'")[0] for t in terms_hit
            ][:4]
            plan["igt_tags_to_check"] = []
        else:
            plan.setdefault("_translation_search_terms", [])

        return plan

    def _execute_query_plan(self, plan: dict, graph: EvidenceGraph, trace: list, query: str):
        _print(f"    [PLAN] phenomena: {plan.get('phenomena', [])}")

        # ── Translation-fallback mode ─────────────────────────────
        trans_terms = plan.get("_translation_search_terms", [])
        if trans_terms:
            _print(f"    [PLAN] translation-fallback: {trans_terms}")
            for term in trans_terms[:3]:
                _print(f"      -> search_translations('{term}')")
                res = self._call_tool("search_translations", {"query": term})
                if res.text and "No translation" not in res.text:
                    for _l in res.text.splitlines()[:15]: _print(f"         {_l}")
                    trace.append({"action": "search_translations",
                                  "args": {"query": term}, "observation": res.text[:500]})
                    self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN,
                                           f"search_translations('{term}')", igt_ids=res.igt_ids)

            # Trilines + structural parse on the first term
            if trans_terms:
                first = trans_terms[0]
                _print(f"      -> get_triline_examples('{first}')")
                res = self._call_tool("get_triline_examples", {"query": first})
                if res.text and "No examples" not in res.text:
                    for _l in res.text.splitlines()[:20]: _print(f"         {_l}")
                    trace.append({"action": "get_triline_examples",
                                  "args": {"query": first}, "observation": res.text[:600]})
                    self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN,
                                           f"get_triline_examples('{first}')", igt_ids=res.igt_ids)

                    _print(f"      -> parse_example_structure('{first}')")
                    res2 = self._call_tool("parse_example_structure", {"query": first})
                    if res2.text:
                        for _l in res2.text.splitlines()[:25]: _print(f"         {_l}")
                        trace.append({"action": "parse_example_structure",
                                      "args": {"query": first}, "observation": res2.text[:800]})
                        self._add_simple_claim(graph, res2.text, ClaimType.IGT_PATTERN,
                                               f"parse_example_structure('{first}')",
                                               igt_ids=res2.igt_ids)
            return

        # ── Normal mode ───────────────────────────────────────────
        _print(f"    [PLAN] tags: {plan.get('igt_tags_to_check', [])}  absence: {plan.get('category_absence_check','')}")
        for tag in plan.get("igt_tags_to_check", [])[:3]:
            _print(f"      -> analyse_tag({tag})")
            res = self._call_tool("analyse_tag", {"tag": tag})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"         {_l}")
                trace.append({"action": "analyse_tag", "args": {"tag": tag}, "observation": res.text[:400]})
                self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"analyse_tag({tag})", igt_ids=res.igt_ids)
        for seed in plan.get("cluster_seeds", [])[:2]:
            _print(f"      -> find_tag_cluster({seed})")
            res = self._call_tool("find_tag_cluster", {"seed_tag": seed})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"         {_l}")
                trace.append({"action": "find_tag_cluster", "args": {"seed_tag": seed}, "observation": res.text[:400]})
                self._add_simple_claim(graph, res.text, ClaimType.IGT_PATTERN, f"find_tag_cluster({seed})", igt_ids=res.igt_ids)
        if plan.get("category_absence_check"):
            cat = plan["category_absence_check"]
            _print(f"      -> analyse_absence({cat})")
            res = self._call_tool("analyse_absence", {"category": cat})
            if res.text:
                for _l in res.text.splitlines()[:15]: _print(f"         {_l}")
                trace.append({"action": "analyse_absence", "args": {"category": cat}, "observation": res.text[:400]})
                if "CANNOT ASSESS" in res.text:
                    _print(f"         → skipped (unknown cluster — not absence evidence)")
                else:
                    self._add_simple_claim(graph, res.text, ClaimType.ABSENCE_EVIDENCE,
                                           f"analyse_absence({cat})", igt_ids=res.igt_ids)

    # ── Output sanitiser ─────────────────────────────────────────

    @staticmethod
    def _sanitize_output(data: dict) -> dict:
        """
        Strip internal claim_id references (e.g. '[claim_id=C005]', 'C005')
        from all human-readable output fields.  These are evidence-graph
        internals that should never surface in the final JSON result.
        """
        import re
        # Match "from/in/by [claim_id=C005], [claim_id=C006], and [C009]" as a unit,
        # plus standalone bare and bracketed claim_id references.
        _CLAIM_ID_RE = re.compile(
            r'(?:from|in|by|via|of)\s+'
            r'(?:(?:\[claim_id=C\d+\]|claim_id=C\d+)[\s,]*(?:and\s+)?)+'
            r'|'
            r'\[claim_id=C\d+\],?\s*(?:and\s+)?'
            r'|'
            r'\bclaim_id=C\d+,?\s*(?:and\s+)?',
            re.IGNORECASE,
        )

        def _clean(v):
            if isinstance(v, str):
                cleaned = _CLAIM_ID_RE.sub('', v)
                cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
                return cleaned
            if isinstance(v, list):
                return [_clean(i) for i in v]
            return v

        TEXT_FIELDS = (
            "answer", "linguistic_definition", "structural_description",
            "key_evidence", "value_detail", "typological_notes", "review_reason",
        )
        return {k: (_clean(v) if k in TEXT_FIELDS else v) for k, v in data.items()}

    def _conclude_query(self, query: str, graph: EvidenceGraph, trace: list):
        from state import QueryResult
        _print("    [concluding query — synthesizing evidence...]")

        trace_summary = "; ".join(
            f"{t['action']}({list(t['args'].values())[:1]})" for t in trace[-8:]
        )
        prompt = IGT_QUERY_CONCLUDE_PROMPT.format(
            language=self.language,
            query=query,
            evidence_graph=graph.summarize(),
            trace_summary=trace_summary,
        )
        raw  = self.llm.generate(prompt, max_new_tokens=1024, json_mode=False)
        data = self._parse_json(raw)
        if not data or not data.get("answer"):
            data = {
                "answer": raw,
                "linguistic_definition": "",
                "structural_description": "",
                "key_evidence": [],
                "confidence": 0.5,
                "needs_human_review": True,
                "review_reason": "Could not parse structured answer; grammar consultation recommended",
            }

        data = self._sanitize_output(data)

        igt_ids  = graph.get_igt_example_ids()
        igt_used = self.toolkit.lookup_igt_examples(igt_ids)

        import hashlib
        qid = "Q" + hashlib.md5(query.encode()).hexdigest()[:6].upper()

        result = QueryResult(
            query_id=qid,
            query=query,
            phenomena=data.get("phenomena", []),
            linguistic_definition=data.get("linguistic_definition", ""),
            structural_description=data.get("structural_description", ""),
            answer=data.get("answer", ""),
            key_evidence=data.get("key_evidence", []),
            igt_examples_used=igt_used,
            igt_support=True,
            search_trace=trace,
            confidence=float(data.get("confidence", 0.5)),
            needs_human_review=data.get("needs_human_review", True),
            review_reason=data.get("review_reason", ""),
            audit_verdict="upheld",
            audit_objections=[],
            token_usage=self.llm.get_token_counts(),
        )

        _print("\n" + "="*60)
        _print(f"RESULT  [{result.query_id}]")
        _print("="*60)
        _print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        _print("="*60 + "\n")

        return result

    # ── Value consistency checker ─────────────────────────────────

    def _fix_value_consistency(self, data: dict, graph=None) -> dict:
        """
        Detect and fix contradictions between value / value_detail / key_evidence
        and structural_description, e.g.:
          - value="No" but structural_description says the tag was found
          - key_evidence items contradict each other (some say found, some say absent)

        Correction rules:
          1. Count "found / attested / appears / present" signals vs
             "not found / absent / zero / does not" signals across all text fields.
          2. If the tag appears in the evidence graph as IGT_PATTERN claims,
             treat that as strong "found" evidence.
          3. If found_signals > 0 and value is "No" → upgrade to "Partial".
          4. Rewrite value_detail to reflect the correction.
          5. Flag needs_human_review = True with a reason.
        """
        value = data.get("value", "Unclear")
        if value not in ("No", "Yes", "Partial", "Unclear", "?"):
            return data

        # Signals from text fields
        FOUND_WORDS   = re.compile(
            r'\b(?:found|attested|appears?|present|occurs?|exists?|'
            r'marked|used|shown|observed|detected)\b', re.IGNORECASE
        )
        ABSENT_WORDS  = re.compile(
            r'\b(?:not found|not attested|absent|zero|0 occurrence|'
            r'does not (?:appear|occur|exist)|no (?:evidence|examples?|instance))\b',
            re.IGNORECASE
        )

        # Collect all human-readable text
        all_text_parts = [
            data.get("structural_description", ""),
            data.get("value_detail", ""),
        ] + (data.get("key_evidence", []) if isinstance(data.get("key_evidence"), list) else [])

        combined = " ".join(str(p) for p in all_text_parts)

        found_count  = len(FOUND_WORDS.findall(combined))
        absent_count = len(ABSENT_WORDS.findall(combined))

        # Also check evidence graph directly if provided
        igt_pattern_count = 0
        if graph is not None:
            from evidence_graph import ClaimType
            igt_pattern_count = sum(
                1 for c in graph.claims.values()
                if c.claim_type == ClaimType.IGT_PATTERN
                and c.supports_hypothesis is True
            )

        has_positive_evidence = found_count > 0 or igt_pattern_count > 0

        if value == "No" and has_positive_evidence and absent_count < found_count:
            _print(
                f"       ⚠ consistency fix: value='No' but found_signals={found_count} "
                f"(absent={absent_count}) → upgrading to 'Partial'",
                indent=4,
            )
            data["value"]              = "Partial"
            data["needs_human_review"] = True
            data["review_reason"]      = (
                f"Auto-corrected: LLM wrote value='No' but evidence contains "
                f"{found_count} attested/found signals. "
                "Human should verify whether the feature is present or absent."
            )
            old_detail = data.get("value_detail", "")
            if old_detail and "not found" in old_detail.lower():
                data["value_detail"] = (
                    "Evidence is contradictory: some sources indicate the feature "
                    "is present, others suggest absence. Human verification needed."
                )

        return data

    def _parse_json(self, text: str) -> dict:
        original = text
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
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
            for i in range(len(text) - 1, max(len(text) - 500, 0), -1):
                if text[i] == "}":
                    try:
                        return json.loads(text[:i+1])
                    except json.JSONDecodeError:
                        continue
            for suffix in ['"}', '"]}', '"]}}'  , ']}', ']}}'  , '}', '}}']:
                try:
                    return json.loads(text + suffix)
                except json.JSONDecodeError:
                    continue
            return {}


# ═══════════════════════════════════════════════════════════════
# Pipeline entry points
# ═══════════════════════════════════════════════════════════════

def run_igt_pipeline(
    language: str,
    igt_path: str,
    output_dir: str = "output",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_iterations_per_feature: int = 12,
    confidence_threshold: float = 0.65,
    use_vllm: bool = False,
    abbreviations_path: Optional[str] = None,    # ← NEW
) -> dict:
    from llm import QwenLLM
    from state import EpistemicState

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm   = QwenLLM(model_name, use_vllm=use_vllm)
    agent = IGTOnlyResearchAgent(
        language=language,
        igt_path=igt_path,
        llm=llm,
        max_iterations_per_feature=max_iterations_per_feature,
        confidence_threshold=confidence_threshold,
        abbreviations_path=abbreviations_path,               # ← NEW
    )

    _print(f"\n{'='*60}")
    _print(f"Language: {language}  [IGT-only mode]")
    _print(f"{'='*60}")

    state = agent.run()

    out_file = output_path / f"{language.lower()}_igt_features.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    _print(f"Saved: {out_file}")

    return state.to_dict()


def run_igt_query_pipeline(
    language: str,
    igt_path: str,
    queries: list,
    output_dir: str = "output",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_iterations: int = 10,
    use_vllm: bool = False,
    abbreviations_path: Optional[str] = None,    # ← NEW
) -> list:
    from llm import QwenLLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm   = QwenLLM(model_name, use_vllm=use_vllm)
    agent = IGTOnlyResearchAgent(
        language=language,
        igt_path=igt_path,
        llm=llm,
        max_iterations_per_feature=max_iterations,
        abbreviations_path=abbreviations_path,               # ← NEW
    )

    results = []
    for i, query in enumerate(queries, 1):
        _print(f"\n[IGT Query {i}/{len(queries)}] {query[:60]}")
        result = agent.answer_query(query, max_iterations=max_iterations)
        results.append(result)

        safe_name = re.sub(r"[^\w\s-]", "", query[:40]).strip().replace(" ", "_").lower()
        out_file  = output_path / f"igt_query_{i:02d}_{safe_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        _print(f"Saved: {out_file}")

    lang_slug     = language.lower().replace(" ", "_")
    combined_file = output_path / f"{lang_slug}_igt_queries.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(
            {"language": language, "mode": "igt_only", "queries": [r.to_dict() for r in results]},
            f, ensure_ascii=False, indent=2,
        )
    _print(f"Combined saved: {combined_file}")
    return results
