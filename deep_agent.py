"""
deep_agent.py — DeepLanguageResearchAgent
==========================================
A genuinely deep search agent for typological feature discovery.

Core improvements over agent.py:
  1. Full section reading — reads entire sections, not 700-char snippets
  2. Quantitative IGT grounding — every feature claim must be grounded in
     IGT statistics, not just prose retrieval
  3. Evidence graph — structured typed claim graph with automatic
     contradiction detection and log-odds confidence aggregation
  4. Multi-hop cross-reference following — automatically reads linked sections
  5. Absence quantification — real negative evidence from IGT corpus
  6. Construction pattern analysis — tests hypotheses against ordered tag sequences
  7. Author claim extraction — separates analytical statements from examples
  8. Contradiction resolution — agent must explicitly resolve detected contradictions
     before concluding
  9. AbbreviationRegistry — gloss tag meanings are injected into every prompt
     that lists IGT tags, so the LLM reasons over "PST (past)" rather than
     bare "PST".

Tool set (11 tools, each with a clear epistemic role):
  read_full_section(query)          → full section text, no truncation
  follow_cross_references(query)    → section + all linked sections
  extract_author_claims(query)      → only analytical statements
  search_text(query, top_k)         → discovery search across grammar
  analyse_tag(tag)                  → quantitative tag profile
  analyse_construction(tags)        → ordered tag sequence search
  analyse_absence(category)         → real negative evidence
  compare_tags(tag_a, tag_b)        → complementary distribution test
  get_section_igt(section_query)    → all IGT from matching sections
  search_translations(query)        → keyword search across translation lines
  get_triline_examples(query)       → aligned morpheme/gloss/translation examples
  conclude                          → only when all evidence types present
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from deep_tools import DeepGrammarToolkit
from evidence_graph import EvidenceGraph, ClaimType
from state import EpistemicState, Feature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _print(msg: str, indent: int = 0) -> None:
    import sys
    print(" " * indent + msg, flush=True)


def _wrap(text: str, width: int) -> list:
    words = text.split()
    lines, current, length = [], [], 0
    for w in words:
        if length + len(w) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [w], len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return lines or [""]


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

DOMAIN_EXTRACTION_PROMPT = """You are a linguistic typologist. Below is the COMPLETE table of contents of a reference grammar for "{language}", down to the finest level of detail (chapter > section > subsection > subsubsection). Each entry is followed by an LLM-generated summary of its linguistic content where available.

{abbrev_legend}

TABLE OF CONTENTS WITH SECTION SUMMARIES (full 4-level hierarchy):
{toc_with_summaries}

QUANTITATIVE IGT SUMMARY (empirical basis):
{igt_summary}

Your task:
1. Read the section summaries carefully — they reveal the ACTUAL content of each section, which often differs from what the title alone suggests. Use subsection and subsubsection summaries to identify phenomena that are only discussed in narrow sub-parts of a chapter.
2. Identify major typological domains covered (TENSE_ASPECT_MODALITY, MORPHOLOGICAL_COMPLEXITY, WORD_ORDER, ARGUMENT_MARKING, INFORMATION_STRUCTURE, EVIDENTIALITY, PHONOLOGY, etc.)
3. For each domain, generate 3-5 specific, answerable feature questions grounded in:
   (a) what the section SUMMARIES confirm is discussed in the grammar, AND
   (b) what the IGT tag frequencies suggest is present or absent.

IMPORTANT — reading the hierarchy:
- A subsubsection like "Verbs > Tense > Past tense > Remote past" is more specific than "Verbs > Tense". Prefer the most specific section name when formulating relevant_sections.
- If a subsection summary mentions a phenomenon explicitly (e.g. "discusses the preverbal past marker bin"), add that subsection to relevant_sections even if the chapter title is generic.
- If no summary mentions a phenomenon, do NOT assume it is covered — mark those features as needing discovery.

IMPORTANT — using the IGT summary:
- If PST/PFV/PRF appear frequently → generate TMA questions
- If EVID/REP appear at 0% → mark EVIDENTIALITY as likely absent, still worth confirming
- If no agreement tags appear → AGREEMENT questions may resolve to "No"

Feature question rules:
- Binary: "Does [LANGUAGE] have X?" → Yes / No / Partial / Unclear
- Categorical: "What is the basic word order?" → SOV / SVO / VSO / ...
- Scalar: "How morphologically complex is [LANGUAGE]?" → High / Medium / Low

UNIQUENESS RULE — before finalising, scan ALL candidate_features across ALL domains:
- Each feature question must be unique across the entire output, not just within its own domain.
- If the same phenomenon (e.g. negation, switch-reference, evidentiality) would appear in two
  different domains, keep it only in the domain where it is most central, and drop it from the other.
- Two questions that differ only in wording but investigate the same linguistic phenomenon count
  as duplicates (e.g. "Does X mark tense?" and "Does X grammaticalize tense distinctions?").
- Prefer more specific, falsifiable questions over broad ones when collapsing duplicates.

Output ONLY valid JSON:
{{
  "domains": [
    {{
      "domain_id": "D001",
      "domain_name": "TENSE_ASPECT_MODALITY",
      "relevant_sections": ["Verbs > Tense > Past tense", "Verbs > Aspect"],
      "igt_signals": ["PST: 140 examples (8.4%)", "IPFV: 428 (25.7%)"],
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


FEATURE_PLAN_PROMPT = """You are a linguistic typologist building a search plan.

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}
IGT PRIOR SIGNALS: {igt_signals}

{abbrev_legend}

STEP 1 — CANDIDATE TAGS FROM ABBREVIATION REGISTRY (matched to this phenomenon):
{abbrev_candidates}

Use these as your primary starting point for igt_tags_to_check before consulting raw statistics.
Tags marked IN CORPUS should be analysed first; those marked NOT IN CORPUS are useful for absence checks.

CRITICAL: The IGT PRIOR SIGNALS above may contain generic high-frequency tags (PST, FUT, IMPF, etc.)
that are irrelevant to this specific phenomenon. DO NOT copy those into igt_tags_to_check.
ONLY use tags from the STEP 1 candidate list above, or tags you can explicitly justify as
diagnostic for "{question}" based on the section summaries below.

AVAILABLE SECTIONS (with chapter/section/subsection hierarchy and content summaries):
{toc_with_summaries}

FULL IGT STATISTICS:
{igt_summary}

{igt_availability_note}
Read the section summaries carefully before planning:
- Identify sections whose summary explicitly mentions the phenomenon you are investigating,
  even if the section title does not.
- Note the author's exact terminology from the summaries and use it in search_queries.
- If a summary confirms a feature is discussed there, add that section to target_sections.
- If a summary rules out relevance, skip that section.

Plan a comprehensive investigation. You have 11 tools:
  read_full_section       → read complete text of a section
  follow_cross_references → read section + all linked sections
  extract_author_claims   → get only the author's analytical statements
  search_text             → keyword/semantic search across grammar
  analyse_tag             → quantitative profile of a gloss tag         [IGT required]
  analyse_construction    → find ordered tag sequence in IGT            [IGT required]
  analyse_absence         → quantify absence of a typological category  [IGT required]
  compare_tags            → test complementary distribution of two tags [IGT required]
  get_section_igt         → all IGT examples from a section             [IGT required]
  search_translations     → keyword search across translation lines     [IGT required]
  get_triline_examples    → aligned morpheme/gloss/translation data     [IGT required]

REQUIRED evidence types before concluding:
{required_evidence_block}
  4. If contradictions are detected in the evidence graph, they must be resolved

IMPORTANT — how to choose target_sections:
  Read each section's Summary carefully. Only add a section to target_sections if its
  Summary explicitly mentions the phenomenon you are investigating. If the Summary is about
  something unrelated (e.g. phonology, morphological derivation, syntax of NPs), skip it.
  For feature "{question}", look for summaries mentioning tense/aspect/mood markers,
  preverbal particles, TMA system, grammaticalized distinctions, or the relevant domain.

  Format: use the section name exactly as it appears in the TOC header (e.g. "The TMA system",
  "Tense", "Aspect"). Do NOT use chunk IDs. Do NOT copy the full TOC line.

Output ONLY valid JSON:
{{
  "target_sections": ["The TMA system", "Tense"],
  "search_queries": ["tense marking preverbal particle", "past anterior marker"],
  "igt_tags_to_check": ["PST", "PRF", "FUT"],
  "constructions_to_check": [["SBJ", "PST", "V"], ["PST", "NEG"]],
  "category_absence_check": "TENSE",
  "tags_to_compare": [["PST", "PFV"], ["PRF", "PFV"]],
  "counter_evidence_framing": "aspect only language, tense absent, temporal adverbs instead of markers",
  "min_queries_before_conclude": 5
}}"""


SEARCH_DECISION_PROMPT = """You are deep-searching a reference grammar for a typological feature.

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}

SEARCH PLAN:
{plan_summary}

CURRENT EVIDENCE GRAPH:
{evidence_summary}

EVIDENCE GAPS (what is still missing):
{gap_analysis}

Progress: iteration {iteration}/{max_iter}
Constraints met: {constraints_met}

{igt_availability_note}
AVAILABLE TOOLS:
1.  read_full_section(query)           — Read COMPLETE section text (preferred over search_text for known sections)
2.  follow_cross_references(query)     — Read section + all sections it references
3.  extract_author_claims(query)       — Get only the author's analytical statements (no examples)
4.  search_text(query, top_k)          — Keyword/semantic search for discovery
5.  analyse_tag(tag)                   — Quantitative profile: frequency, position, co-occurrence
6.  analyse_construction(tags)         — Find ordered tag sequence [TAG1, TAG2, ...] in IGT
7.  analyse_absence(category)          — Quantify absence of a category (real negative evidence)
8.  compare_tags(tag_a, tag_b)         — PMI + positional comparison of two tags
9.  get_section_igt(section_query)     — All IGT examples from a section
10. search_translations(query)         — Keyword search across translation lines; use when abbreviation lookup finds no candidates, to locate relevant examples by meaning and discover diagnostic tags
11. get_triline_examples(query)        — Aligned morpheme/gloss/translation trilines for a tag, keyword, or section; use to inspect actual surface forms and morpheme order after identifying candidates via search_translations or analyse_tag
12. conclude                           — Only when all constraints are met AND gaps are filled

Think carefully:
- If gap_analysis says "NO IGT QUANTITATIVE EVIDENCE" → use analyse_tag or analyse_absence
- If gap_analysis says "NO GRAMMAR PROSE EVIDENCE" → use read_full_section or extract_author_claims
- If gap_analysis says "UNRESOLVED CONTRADICTIONS" → use read_full_section or follow_cross_references to resolve
- If gap_analysis says "NO COUNTER-EVIDENCE SEARCH" → use analyse_absence or search_text with counter framing
- Prefer read_full_section over search_text when you know which section to read
- Use analyse_construction to test a specific grammatical pattern hypothesis
- If the abbreviation candidate block says "no matches found" → use search_translations(keyword) first to find relevant examples by meaning, then get_triline_examples to inspect their morpheme/gloss structure and identify the actual diagnostic tag

Output ONLY valid JSON:
{{
  "thought": "detailed reasoning about what evidence is missing and why this action addresses it",
  "action": "read_full_section|follow_cross_references|extract_author_claims|search_text|analyse_tag|analyse_construction|analyse_absence|compare_tags|get_section_igt|search_translations|get_triline_examples|conclude",
  "args": {{}},
  "evidence_type": "grammar_prose|igt_quantitative|absence|construction|counter_evidence",
  "claim_to_add": "one sentence claim this observation would support or refute",
  "supports_hypothesis": true | false | null
}}"""


CLAIM_EXTRACTION_PROMPT = """You are extracting typed claims from a grammar observation.

LANGUAGE: {language}
FEATURE QUESTION: {question}
OBSERVATION:
{observation}

Extract 1-2 specific, falsifiable claims from this observation.
For the "type" field, write EXACTLY ONE of these six strings (no pipes, no slashes):
  grammar_statement  = author explicitly describes a grammatical category or rule
  igt_pattern        = pattern derived from IGT frequency, position, or co-occurrence data
  absence_evidence   = confirms a category is absent or ungrammaticalized
  inference          = conclusion derived by combining multiple pieces of evidence
  counter_evidence   = evidence that contradicts the working hypothesis
  author_caveat      = author hedges or qualifies a claim

Output ONLY a valid JSON object. No prose before or after.
{{
  "claims": [
    {{
      "text": "one sentence stating the claim",
      "type": "grammar_statement",
      "confidence": 0.9,
      "supports_hypothesis": true,
      "igt_examples": []
    }}
  ]
}}"""


CONTRADICTION_RESOLUTION_PROMPT = """You are resolving a contradiction in the evidence for a typological feature.

LANGUAGE: {language}
FEATURE QUESTION: {question}

CONTRADICTION:
{contradiction_description}

CLAIM A: {claim_a_text}
  Source: {claim_a_source}
  Evidence: {claim_a_evidence}

CLAIM B: {claim_b_text}
  Source: {claim_b_source}
  Evidence: {claim_b_evidence}

Resolve this contradiction:
- Is one source more authoritative than the other?
- Could both be true (e.g., synchronic variation, different construction types)?
- Does one apply to a restricted context the other doesn't mention?
- Is one the author's analysis and the other a misinterpretation?

Output ONLY valid JSON:
{{
  "resolution": "brief explanation of how the contradiction is resolved",
  "resolved": true | false,
  "preferred_claim": "A" | "B" | "both_partial" | "neither",
  "revised_confidence_a": 0.0-1.0,
  "revised_confidence_b": 0.0-1.0,
  "synthesis": "one sentence that captures the nuanced truth"
}}"""


CONCLUSION_PROMPT = """You are writing a final typological feature entry.

LANGUAGE: {language}
FEATURE QUESTION: {question}
DOMAIN: {domain}

COMPLETE EVIDENCE GRAPH:
{evidence_graph}

CHUNK SUMMARIES (concise LLM-generated synopsis of each cited grammar section):
{chunk_summaries}

SEARCH TRACE SUMMARY:
{trace_summary}

Synthesize all evidence into a final determination. Structure your output as follows:

1. LINGUISTIC DEFINITION — define the feature as a linguistic category (language-agnostic,
   1-2 sentences). This should be a general typological definition, not a description of {language}.
   Example: "Grammatical tense marking is the obligatory encoding of temporal reference through
   bound morphology or fixed particles, distinct from aspect or modality."

2. STRUCTURAL DESCRIPTION — how the feature is realised in {language}: specific forms,
   markers, positions, and constructions found in the evidence (2-3 sentences).

3. VALUE + EVIDENCE — the Yes/No/Partial/Unclear verdict with supporting paragraphs.

Weighting rules:
- GRAMMAR_STATEMENT claims (author's explicit prose) outweigh IGT_PATTERN alone, because the IGT corpus may be sampled from the grammar itself and is likely incomplete
- ABSENCE_EVIDENCE with zero IGT counts is strong negative evidence
- Unresolved contradictions → lower confidence, flag for review
- Only claim "Yes" if you have BOTH grammar prose AND IGT quantitative support

NOTE: claim_id values like C001 are INTERNAL — do not use them.

For each key_evidence item write a SHORT PARAGRAPH (2-4 sentences):
  1. Finding: what the evidence shows
  2. Source: cite section and chunk IDs (e.g. "§Tense [chunk_0093_p0]") or IGT tool
  3. Justification: why this supports or complicates the conclusion
  4. (Optional) one short phrase quoted from the grammar text

Output ONLY valid JSON:
{{
  "linguistic_definition": "language-agnostic definition of the feature as a linguistic category",
  "structural_description": "how this feature is realised in {language}: specific forms, markers, positions",
  "value": "Yes" | "No" | "Partial" | "Unclear" | "?",
  "value_detail": "one sentence with specifics (key tags, section refs, marker names)",
  "confidence": 0.0-1.0,
  "key_evidence": [
    "The author explicitly marks past tense with the preverbal particle bin. Source: §The verbal system > Tense [chunk_0093_p0, chunk_0093_p1]. This directly confirms grammaticalized past marking in Pichi. The grammar states bin 'invariably precedes the verb stem'.",
    "PST tag appears in 140 of 1668 IGT examples (8.4%), consistently in preverbal position (mean=0.31). Source: IGT tag analysis: PST. The frequency and fixed position confirm bin is a productive, not marginal, past marker.",
    "FUT tense marking is absent in IGT (0 examples across all future-related tags). Source: IGT absence analysis: TENSE. This suggests asymmetric tense marking — past is grammaticalized but future is not, common in creole systems."
  ],
  "typological_notes": "2-3 sentences on cross-linguistic significance and typological context",
  "needs_human_review": true,
  "review_reason": "one sentence explaining what requires human verification, else empty string"
}}
IMPORTANT: Maximum 4 items in key_evidence. Each item should be 2-4 sentences."""


AUDITOR_PROMPT = """You are a critical auditor reviewing a typological conclusion.

LANGUAGE: {language}
FEATURE: {question}
CONCLUSION: {value} — "{value_detail}"
CONFIDENCE: {confidence}

EVIDENCE GRAPH SUMMARY:
{evidence_graph_summary}

Try to DISPROVE this conclusion using only the evidence listed:
- Are the IGT patterns actually diagnostic of this feature, or just correlated?
- Could the grammar prose be describing a marginal or restricted phenomenon?
- Are there unresolved contradictions the conclusion ignores?
- Is "Yes" warranted, or should it be "Partial" given the evidence?
- Is the confidence calibrated — not overconfident given the evidence quality?

Output ONLY valid JSON:
{{
  "verdict": "upheld" | "weakened" | "overturned",
  "objections": ["short objection, max 60 chars", "short objection, max 60 chars"],
  "revised_value": "same or corrected",
  "revised_confidence": 0.0-1.0,
  "revised_value_detail": "one sentence, max 120 chars",
  "audit_notes": "one sentence summary"
}}
IMPORTANT: Keep all string values short. Maximum 2 objections."""


QUERY_PLAN_PROMPT = """You are planning a deep search of a reference grammar to answer a research query.

LANGUAGE: {language}
QUERY: {query}

{abbrev_legend}

STEP 1 — CANDIDATE TAGS FROM ABBREVIATION REGISTRY (matched to this query):
{abbrev_candidates}

Use these as your primary starting point for igt_tags_to_check before consulting raw statistics.

TABLE OF CONTENTS WITH SECTION SUMMARIES:
{toc_with_summaries}

IGT STATISTICS (tag frequencies and positions):
{igt_summary}

{igt_availability_note}
Your task:
1. Identify the specific linguistic phenomena the query is asking about.
2. Read the section summaries carefully. Add a section to target_sections ONLY if its
   summary explicitly mentions those phenomena — not just because the title sounds related.
3. Identify IGT tags that are diagnostic for those phenomena (use the IGT statistics above).

Rules for target_sections:
- Use the section name exactly as it appears in the TOC (e.g. "Aspect/mood marking", "Tense")
- Do NOT use chunk IDs
- Maximum 4 sections; pick the most directly relevant ones

Output ONLY valid JSON:
{{
  "phenomena": ["short name of phenomenon 1", "phenomenon 2"],
  "rationale": "one sentence: why these sections and tags are relevant to the query",
  "target_sections": ["exact section name 1", "exact section name 2"],
  "igt_tags_to_check": ["TAG1", "TAG2"],
  "constructions_to_check": [["TAG1", "TAG2"]],
  "category_absence_check": "TENSE",
  "tags_to_compare": [["PST", "PFV"]],
  "search_queries": ["fallback query if sections not found"]
}}"""


QUERY_DECISION_PROMPT = """You are deep-searching a reference grammar to answer a research question.

LANGUAGE: {language}
QUERY: {query}

EVIDENCE GATHERED SO FAR:
{evidence_summary}

Progress: iteration {iteration}/{max_iter}  |  sections already read: {sections_read}

{igt_availability_note}
AVAILABLE TOOLS:
1.  read_full_section(query)           — Read COMPLETE section text
2.  follow_cross_references(query)     — Read section + all sections it links to
3.  extract_author_claims(query)       — Get only the author's analytical statements
4.  search_text(query, top_k)          — Keyword search across the whole grammar
5.  analyse_tag(tag)                   — Quantitative profile: frequency, position, co-occurrence
6.  analyse_construction(tags)         — Find ordered tag sequence in IGT
7.  analyse_absence(category)          — Quantify absence of a typological category
8.  compare_tags(tag_a, tag_b)         — PMI + positional comparison of two tags
9.  get_section_igt(section_query)     — All IGT examples from a section
10. search_translations(query)         — Keyword search across translation lines; use when abbreviation lookup finds no candidates, to locate relevant examples by meaning and discover diagnostic tags
11. get_triline_examples(query)        — Aligned morpheme/gloss/translation trilines for a tag, keyword, or section; use to inspect actual surface forms and morpheme order after identifying candidates via search_translations or analyse_tag
12. conclude                           — When you have enough evidence to answer the query well

Use this loop to follow up on cross-references, fill gaps the plan missed, or
dig deeper into a specific aspect of the query. The main sections have already
been read above — focus on what is still missing.

Output ONLY valid JSON:
{{
  "thought": "what the evidence shows so far and what specific gap remains",
  "action": "tool_name or conclude",
  "args": {{}},
  "finding": "one sentence: what this action is expected to add"
}}"""


QUERY_CONCLUSION_PROMPT = """You are writing a detailed, fully-cited answer to a grammar research query.

LANGUAGE: {language}
QUERY: {query}

COMPLETE EVIDENCE GRAPH:
{evidence_graph}

CHUNK SUMMARIES (LLM-generated synopsis of each cited grammar section):
{chunk_summaries}

SEARCH TRACE: {trace_summary}

Produce a structured answer. Write the fields IN THIS EXACT ORDER:

1. ANSWER (first and most important) — a direct, detailed prose answer (2-4 paragraphs).
   Cite sources inline: "§Section name [chunk_0093_p0]".
   If evidence is limited, write what IS known and note what remains unclear.
   This field must never be empty.

2. LINGUISTIC DEFINITION — define the queried phenomenon as a general linguistic category,
   language-agnostic (1-2 sentences).

3. STRUCTURAL DESCRIPTION — how the phenomenon is realised in {language}:
   specific forms, markers, positions (2-3 sentences).

4. KEY EVIDENCE — up to 4 items, each a SHORT PARAGRAPH (2-4 sentences):
   Finding / Source (§Section [chunk_id] or IGT tool) / Justification

CRITICAL RULES:
- Write answer FIRST — it is the most important field
- claim_id values C001, C002, C003 are INTERNAL — never cite them
- Cite sources as: §Section name [chunk_0028_p0]

Output ONLY valid JSON with answer as the FIRST key:
{{
  "answer": "2-4 paragraph prose answer — THIS MUST NOT BE EMPTY",
  "linguistic_definition": "language-agnostic definition",
  "structural_description": "how this works in {language}: specific forms and markers",
  "key_evidence": [
    "Finding: ... Source: §Section [chunk_0093_p0]. Justification: ...",
    "Finding: ... Source: IGT tag analysis: FUT (49 examples, 21.7%). Justification: ..."
  ],
  "confidence": 0.0-1.0,
  "needs_human_review": true or false,
  "review_reason": "one sentence or empty string"
}}"""


QUERY_AUDITOR_PROMPT = """You are a critical auditor reviewing an answer to a grammar research query.

LANGUAGE: {language}
QUERY: {query}
ANSWER SUMMARY: "{answer_summary}"
CONFIDENCE: {confidence}

EVIDENCE GRAPH SUMMARY:
{evidence_graph_summary}

Try to find weaknesses in this answer using only the evidence listed:
- Are the cited sections actually about the queried phenomenon, or tangentially related?
- Does the answer over-claim based on limited evidence?
- Are there unresolved contradictions the answer ignores?
- Is the confidence calibrated given the evidence quality and quantity?
- Is important evidence missing that would change the answer?

Output ONLY valid JSON:
{{
  "verdict": "upheld" | "weakened" | "overturned",
  "objections": ["short objection, max 60 chars", "short objection, max 60 chars"],
  "revised_confidence": 0.0-1.0,
  "audit_notes": "one sentence summary of audit outcome"
}}
IMPORTANT: Keep all string values short. Maximum 2 objections."""


# ═══════════════════════════════════════════════════════════════
# DeepLanguageResearchAgent
# ═══════════════════════════════════════════════════════════════

class DeepLanguageResearchAgent:

    def __init__(
        self,
        language: str,
        grammar_path: str,
        igt_path: Optional[str],
        llm,
        max_iterations_per_feature: int = 15,
        confidence_threshold: float = 0.75,
        min_queries_per_feature: int = 5,
        abbreviations_path: Optional[str] = None,    # ← NEW
    ):
        self.language    = language
        self.llm         = llm
        self.toolkit     = DeepGrammarToolkit(
            grammar_path,
            igt_path,
            abbreviations_path=abbreviations_path,   # ← NEW
        )
        self._abbrev     = self.toolkit.abbrev        # ← NEW: convenience shortcut
        self._has_igt    = self.toolkit.igt_analyser is not None
        self.state       = EpistemicState(language)
        self.max_iter    = max_iterations_per_feature
        self.conf_thresh = confidence_threshold
        self.min_queries = min_queries_per_feature

    # ── Abbreviation-based tag candidate lookup ───────────────────

    # Words to strip from queries before keyword extraction
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

    def _find_tags_via_translations(
        self, query_text: str, top_n: int = 6
    ) -> tuple:
        """
        Fallback tag discovery when abbreviation lookup finds no candidates.

        1. Extracts content keywords and short phrases from query_text
           (stripping the language name and generic linguistic terminology).
        2. Searches translation lines for each keyword.
        3. Aggregates tag frequencies from matching examples, filtering out
           universal high-frequency tags that appear in >30% of the corpus.
        4. Returns (diagnostic_tags, search_terms_used, total_hit_count) where
           diagnostic_tags = [(tag, count), ...] sorted by frequency.
        """
        if not self._has_igt or not self.toolkit.igt_analyser:
            return [], [], 0

        stats    = self.toolkit.igt_analyser.get_stats()
        n        = self.toolkit.igt_analyser.n
        # Tags that appear in >30% of examples are not diagnostic
        universal = {
            t for t, p in stats.tag_profiles.items()
            if p.example_coverage > 0.30
        }

        # Strip language name tokens so "Choguita Rarámuri" → nothing useful
        lang_tokens = {
            tok.lower()
            for tok in re.split(r'\W+', self.language)
            if len(tok) >= 3
        }
        stop = self._KW_STOP | lang_tokens

        # Short phrases (more specific than single words)
        phrase_pats = [
            r'\bthere (?:is|are|was|were)\b',
            r'\bis a\b', r'\bare a\b', r'\bwas a\b', r'\bwere a\b',
            r'\bwill be\b', r'\bcan be\b', r'\bbecame\b',
        ]
        phrases = []
        for pat in phrase_pats:
            for m in re.finditer(pat, query_text.lower()):
                phrases.append(m.group())

        # Content keywords (≥3 chars, not in stop list)
        words = re.findall(r"[a-z]{3,}", query_text.lower())
        kws   = list(dict.fromkeys(w for w in words if w not in stop))

        search_terms = list(dict.fromkeys(phrases + kws))[:8]
        if not search_terms:
            return [], [], 0

        from collections import Counter
        all_tags    = Counter()
        total_hits  = 0
        terms_hit   = []
        examples    = self.toolkit.igt_examples

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
        Reverse-lookup the abbreviation registry for tags relevant to
        query_text, annotated with corpus presence when IGT is loaded.

        If the abbreviation registry finds no matches, falls back to
        translation-line search: extracts keywords from query_text,
        searches translation lines, and returns the tags found in matching
        examples — along with a note telling the LLM to use
        search_translations as its first tool call.
        """
        candidates = self._abbrev.find_tags_for_phenomenon(query_text, top_n=10)

        stats = self.toolkit.igt_analyser.get_stats() if self._has_igt else None

        # ── Primary path: abbreviation lookup succeeded ───────────
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

        # ── Fallback: translation-line search ─────────────────────
        if not self._has_igt:
            return (
                "(no matches found in abbreviation registry; "
                "no IGT corpus loaded — rely on grammar prose only)"
            )

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
            "Tags appearing in those examples (use search_translations as first tool):",
        ]
        if stats:
            for tag, count in diag_tags:
                p = stats.tag_profiles.get(tag)
                pct = f"{p.example_coverage*100:.1f}%" if p else "?"
                meaning = self._abbrev.label(tag)
                label   = f"{tag} ({meaning})" if meaning != tag else tag
                lines.append(f"  {label:<28}  found in {count} matching examples ({pct} overall)")
        else:
            for tag, count in diag_tags:
                lines.append(f"  {tag:<12}  {count} matching examples")

        lines.append(
            "IMPORTANT: Start with search_translations(keyword) to retrieve "
            "the relevant sentences, then use get_triline_examples to inspect "
            "their morpheme/gloss structure and identify the actual diagnostic tag."
        )
        return "\n".join(lines)

    # ── Phase 0: Cross-domain deduplication ──────────────────────

    # Stop-words for question normalisation (generic linguistic terms that
    # appear in almost every question and carry no discriminative information).
    _DEDUP_STOP = frozenset({
        'does', 'the', 'a', 'an', 'in', 'of', 'is', 'are', 'have', 'has',
        'do', 'what', 'how', 'basic', 'language', 'grammatical', 'system',
        'mark', 'marks', 'marking', 'express', 'expresses', 'expressing',
        'encode', 'encodes', 'encoding', 'use', 'uses', 'using', 'show',
        'shows', 'exhibit', 'exhibits',
    })
    _DEDUP_JACCARD_THRESHOLD = 0.55   # questions sharing ≥55% of content words are duplicates

    def _dedup_candidates(self, domains: list) -> list:
        """
        Remove near-duplicate candidate feature questions across all domains.

        Algorithm:
          1. Iterate domains in order (earlier domains keep their features).
          2. For each candidate question, compute its normalised word set
             (content words only, language name stripped).
          3. Compare against all already-kept questions via Jaccard similarity.
          4. Drop the candidate if similarity >= _DEDUP_JACCARD_THRESHOLD.

        Returns the domains list with duplicates removed in-place.
        Logs every dropped question and a summary count.
        """
        lang_tokens = frozenset(
            tok.lower()
            for tok in re.split(r'\W+', self.language)
            if len(tok) >= 3
        )
        stop = self._DEDUP_STOP | lang_tokens

        def _norm(q: str) -> frozenset:
            words = re.findall(r'[a-z]+', q.lower())
            return frozenset(w for w in words if w not in stop and len(w) >= 3)

        def _jaccard(a: frozenset, b: frozenset) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        total_before = sum(len(d.get('candidate_features', [])) for d in domains)
        kept_norms: list[frozenset] = []   # norms of already-kept questions
        kept_questions: list[str]   = []   # text of already-kept questions (for logging)
        removed = 0

        for domain in domains:
            filtered = []
            for feature in domain.get('candidate_features', []):
                q     = feature['question']
                q_norm = _norm(q)
                # Find the highest-similarity already-kept question
                best_sim, best_match = 0.0, ''
                for kept_n, kept_q in zip(kept_norms, kept_questions):
                    sim = _jaccard(q_norm, kept_n)
                    if sim > best_sim:
                        best_sim, best_match = sim, kept_q
                if best_sim >= self._DEDUP_JACCARD_THRESHOLD:
                    removed += 1
                    _print(
                        f"  [dedup] dropped from {domain['domain_name']}: "
                        f"'{q[:70]}'\n"
                        f"           ↳ similar to: '{best_match[:70]}' "
                        f"(Jaccard={best_sim:.2f})",
                        indent=2,
                    )
                else:
                    kept_norms.append(q_norm)
                    kept_questions.append(q)
                    filtered.append(feature)
            domain['candidate_features'] = filtered

        total_after = sum(len(d.get('candidate_features', [])) for d in domains)
        if removed:
            _print(
                f"  [dedup] {removed} duplicate(s) removed  "
                f"({total_before} → {total_after} candidate features)",
                indent=0,
            )
        else:
            _print(f"  [dedup] no duplicates found ({total_before} features)", indent=0)
        return domains

    # ── Phase 1: Domain discovery ────────────────────────────────

    def discover_domains(self) -> list:
        mode_label = "TOC (full hierarchy)" + (" + IGT summary" if self._has_igt else " [grammar-only mode, no IGT]")
        _print(f"\n[{self.language}] Discovering domains from {mode_label}...")
        toc_with_summaries = self.toolkit.get_toc_with_summaries(
            skip_subsections=False, max_summary_chars=150
        )
        igt_summary = self.toolkit.get_igt_summary()

        # ← NEW: inject full abbreviation legend so the LLM understands every tag
        abbrev_legend = self._abbrev.prompt_legend()

        prompt = DOMAIN_EXTRACTION_PROMPT.format(
            language=self.language,
            abbrev_legend=abbrev_legend,
            toc_with_summaries=toc_with_summaries[:20000],
            igt_summary=igt_summary[:2000],
        )
        response = self.llm.generate(prompt, max_new_tokens=6144)
        parsed   = self._parse_json(response)
        domains  = parsed.get("domains", [])

        if not domains:
            logger.warning(
                f"  Domain extraction returned 0 domains. "
                f"Raw response (first 500 chars): {response[:500]}"
            )

        total = sum(len(d.get("candidate_features", [])) for d in domains)
        _print(f"  → {len(domains)} domain(s), {total} candidate feature(s) (before dedup)")
        domains = self._dedup_candidates(domains)
        return domains

    # ── Phase 2a: Feature search plan ────────────────────────────

    def build_feature_plan(self, feature: dict, domain: dict) -> dict:
        _print("  [1/5] Building search plan...", indent=4)
        toc_with_summaries = self.toolkit.get_toc_with_summaries(
            skip_subsections=False, max_summary_chars=80
        )
        igt_summary = self.toolkit.get_igt_summary()

        # Reverse-lookup abbreviation registry for phenomenon-relevant tags.
        # These are the ground-truth starting tags for this feature — they come
        # from matching the question text against the abbreviation table, which
        # is far more reliable than what the domain-extraction LLM may have guessed.
        phenomenon_text   = feature["question"] + " " + " ".join(feature.get("igt_signals", []))
        abbrev_candidates = self._format_abbrev_candidates(phenomenon_text)

        # Build the legend from abbrev-candidate tags, NOT from whatever the
        # domain-extraction LLM put in igt_tags_to_check (which is often wrong —
        # high-frequency tags like PST/FUT bleed into every feature).
        candidate_tags = [
            tag for tag, _ in self._abbrev.find_tags_for_phenomenon(phenomenon_text, top_n=8)
        ]
        legend_tags   = candidate_tags if candidate_tags else feature.get("igt_tags_to_check", [])
        abbrev_legend = self._abbrev.prompt_legend(legend_tags) if legend_tags else self._abbrev.prompt_legend()

        if self._has_igt:
            _plan_igt_note = ""
            _required_evidence = (
                "  1. At least one GRAMMAR PROSE source (read_full_section or extract_author_claims)\n"
                "  2. At least one IGT QUANTITATIVE source (analyse_tag or analyse_absence or get_section_igt)\n"
                "  3. At least one COUNTER-EVIDENCE check (analyse_absence, or search_text with negation framing)"
            )
        else:
            _plan_igt_note = (
                "NOTE: No IGT corpus is loaded. Tools marked [IGT required] above are UNAVAILABLE. "
                "Leave igt_tags_to_check, constructions_to_check, category_absence_check, and "
                "tags_to_compare as empty lists/strings. Plan using only grammar prose tools.\n"
            )
            _required_evidence = (
                "  1. At least one GRAMMAR PROSE source (read_full_section or extract_author_claims)\n"
                "  2. [IGT evidence NOT required — no corpus loaded]\n"
                "  3. At least one COUNTER-EVIDENCE check using search_text with negation/alternative framing"
            )

        prompt = FEATURE_PLAN_PROMPT.format(
            language=self.language,
            question=feature["question"],
            domain=domain["domain_name"],
            igt_signals=json.dumps(feature.get("igt_signals", []), ensure_ascii=False),
            abbrev_legend=abbrev_legend,
            abbrev_candidates=abbrev_candidates,
            toc_with_summaries=toc_with_summaries[:12000],
            igt_summary=igt_summary[:2000],
            igt_availability_note=_plan_igt_note,
            required_evidence_block=_required_evidence,
        )
        response = self.llm.generate(prompt, max_new_tokens=1536)
        plan     = self._parse_json(response)

        plan.setdefault("target_sections", [])
        plan.setdefault("search_queries", [feature["question"]])
        if not self._has_igt:
            plan["igt_tags_to_check"]      = []
            plan["constructions_to_check"] = []
            plan["category_absence_check"] = ""
            plan["tags_to_compare"]        = []
        else:
            plan.setdefault("constructions_to_check", [])
            plan.setdefault("category_absence_check", "")
            plan.setdefault("tags_to_compare", [])

            # Post-process: if the plan's igt_tags_to_check contains none of the
            # abbrev candidates (i.e. LLM ignored the guidance and picked generic
            # high-frequency tags), replace them with the candidate tags.
            if candidate_tags:
                plan_tags  = [t.upper() for t in plan.get("igt_tags_to_check", [])]
                cand_upper = [t.upper() for t in candidate_tags]
                overlap    = [t for t in plan_tags if t in cand_upper]
                if not overlap:
                    _print(
                        f"     ⚠ plan tags {plan_tags} don't match abbrev candidates "
                        f"{cand_upper[:5]} — overriding with candidates",
                        indent=4,
                    )
                    plan["igt_tags_to_check"] = candidate_tags[:5]

        plan.setdefault("igt_tags_to_check", candidate_tags[:5] if candidate_tags else [])
        plan.setdefault("counter_evidence_framing", "")
        plan.setdefault("min_queries_before_conclude", self.min_queries)

        _print(f"     target_sections : {plan['target_sections']}", indent=4)
        _print(f"     igt_tags        : {plan['igt_tags_to_check']}", indent=4)
        _print(f"     constructions   : {plan['constructions_to_check']}", indent=4)
        _print(f"     absence_check   : {plan['category_absence_check'] or '—'}", indent=4)
        _print(f"     min_queries     : {plan['min_queries_before_conclude']}", indent=4)

        if plan["target_sections"]:
            _print("     section summaries used by planner:", indent=4)
            for sec_ref in plan["target_sections"]:
                summary = self._get_section_summary(sec_ref)
                if summary:
                    _print(f"       [{sec_ref}] {summary[:200]}", indent=4)
                else:
                    _print(f"       [{sec_ref}] (no summary available)", indent=4)

        return plan

    # ── Phase 2b: Upfront reading of target sections ─────────────

    def read_target_sections(self, plan: dict, feature: dict, graph: EvidenceGraph) -> None:
        targets = plan.get("target_sections", [])[:3]
        _print(f"  [2/5] Reading {len(targets)} target section(s)...", indent=4)
        for section_ref in targets:
            _print(f"     → reading: {section_ref}", indent=4)
            sr = self.toolkit.read_full_section(section_ref)
            if "[NOT FOUND]" in sr.text:
                _print(f"       [NOT FOUND]", indent=4)
                continue

            if sr.chunk_ids:
                summary = self._get_section_summary(sr.chunk_ids[0])
                if summary:
                    _print(f"       Summary: {summary[:200]}", indent=4)

            chunk_ref = ", ".join(sr.chunk_ids[:4])
            if len(sr.chunk_ids) > 4:
                chunk_ref += f"... (+{len(sr.chunk_ids)-4} more)"
            source = f"§{sr.section_path} [chunks: {chunk_ref}]" if sr.section_path else f"section:{section_ref}"

            claims = self._extract_claims_from_observation(
                sr.text, feature["question"], "grammar_prose"
            )
            for claim_dict in claims:
                graph.add_claim(
                    text=claim_dict["text"],
                    claim_type=ClaimType.GRAMMAR_STATEMENT,
                    source=source,
                    confidence=claim_dict["confidence"],
                    supports_hypothesis=claim_dict.get("supports_hypothesis"),
                    raw_evidence=sr.text[:300],
                )
            _print(f"       → {len(claims)} claim(s) extracted  [{source[:60]}]", indent=4)

    def run_igt_initial_pass(self, plan: dict, feature: dict, graph: "EvidenceGraph") -> None:
        if not self._has_igt:
            _print(f"  [3/5] IGT initial pass skipped (no IGT corpus loaded).", indent=4)
            return

        tags        = plan.get("igt_tags_to_check", [])[:5]
        constructs  = plan.get("constructions_to_check", [])[:3]
        pairs       = plan.get("tags_to_compare", [])[:2]
        cat         = plan.get("category_absence_check", "")
        preferred   = plan.get("target_sections", [])
        total_ops   = len(tags) + len(constructs) + len(pairs) + (1 if cat else 0)
        _print(f"  [3/5] IGT initial pass ({total_ops} operation(s))...", indent=4)

        for tag in tags:
            _print(f"     analyse_tag({tag})", indent=4)
            r      = self.toolkit.analyse_tag(tag, preferred_sections=preferred)
            claims = self._extract_claims_from_observation(
                r.text, feature["question"], "igt_quantitative"
            )
            for claim_dict in claims:
                llm_ids = [e for e in claim_dict.get("igt_examples", []) if isinstance(e, str) and not e.startswith("chunk_")]
                graph.add_claim(
                    text=claim_dict["text"],
                    claim_type=ClaimType.IGT_PATTERN,
                    source=f"IGT tag analysis: {tag}",
                    confidence=claim_dict["confidence"],
                    supports_hypothesis=claim_dict.get("supports_hypothesis"),
                    igt_examples=list(dict.fromkeys(r.igt_ids + llm_ids)),
                    raw_evidence=r.text[:300],
                )
            _print(f"       → {len(claims)} claim(s) added", indent=4)

        if cat:
            _print(f"     analyse_absence({cat})", indent=4)
            r = self.toolkit.analyse_absence(cat, preferred_sections=preferred)
            verdict = next(
                (w for w in ["STRONG ABSENCE", "WEAK PRESENCE", "LIMITED", "PRESENT"]
                 if w in r.text), "?"
            )
            _print(f"       → verdict: {verdict}", indent=4)
            claim_type = (
                ClaimType.ABSENCE_EVIDENCE
                if "STRONG ABSENCE" in r.text or "WEAK PRESENCE" in r.text
                else ClaimType.IGT_PATTERN
            )
            supports = False if "STRONG ABSENCE" in r.text else (
                True if "PRESENT" in r.text else None
            )
            graph.add_claim(
                text=r.text[:200],
                claim_type=claim_type,
                source=f"IGT absence analysis: {cat}",
                confidence=0.85 if "STRONG ABSENCE" in r.text else 0.6,
                supports_hypothesis=supports,
                igt_examples=r.igt_ids,
                raw_evidence=r.text,
            )

        for construction in constructs:
            _print(f"     analyse_construction({construction})", indent=4)
            r = self.toolkit.analyse_construction(construction, preferred_sections=preferred)
            if "No examples found" not in r.text:
                claims = self._extract_claims_from_observation(
                    r.text, feature["question"], "igt_quantitative"
                )
                for claim_dict in claims:
                    llm_ids = [e for e in claim_dict.get("igt_examples", []) if isinstance(e, str) and not e.startswith("chunk_")]
                    graph.add_claim(
                        text=claim_dict["text"],
                        claim_type=ClaimType.IGT_PATTERN,
                        source=f"IGT construction: {construction}",
                        confidence=claim_dict["confidence"],
                        supports_hypothesis=claim_dict.get("supports_hypothesis"),
                        igt_examples=list(dict.fromkeys(r.igt_ids + llm_ids)),
                        raw_evidence=r.text[:300],
                    )
                _print(f"       → {len(claims)} claim(s) added", indent=4)
            else:
                _print(f"       → no examples found", indent=4)

        for pair in pairs:
            if len(pair) == 2:
                _print(f"     compare_tags({pair[0]}, {pair[1]})", indent=4)
                r = self.toolkit.compare_tags(pair[0], pair[1], preferred_sections=preferred)
                graph.add_claim(
                    text=r.text[:200],
                    claim_type=ClaimType.IGT_PATTERN,
                    source=f"IGT tag comparison: {pair[0]} vs {pair[1]}",
                    confidence=0.7,
                    supports_hypothesis=None,
                    igt_examples=r.igt_ids,
                    raw_evidence=r.text,
                )

    # ── Phase 2d: Main ReAct loop ─────────────────────────────────

    def investigate_feature(self, feature: dict, domain: dict) -> Optional[Feature]:
        question   = feature["question"]
        feature_id = feature["feature_id"]
        _print(f"  Feature: {question}", indent=2)

        self.llm.reset_token_counter()   # start fresh for this feature
        graph = EvidenceGraph(question)

        plan = self.build_feature_plan(feature, domain)
        self.read_target_sections(plan, feature, graph)
        self.run_igt_initial_pass(plan, feature, graph)

        _print(
            f"  [3/5 done] Upfront pass complete: "
            f"{len(graph.claims)} claim(s), "
            f"{len(graph.contradictions)} contradiction(s)",
            indent=4,
        )

        _print(f"  [4/5] ReAct search loop (max {self.max_iter} iterations)...", indent=4)
        search_trace   = []
        queries_fired  = []
        n_grammar_prose = sum(
            1 for c in graph.claims.values()
            if c.claim_type == ClaimType.GRAMMAR_STATEMENT
        )
        n_igt_quant = sum(
            1 for c in graph.claims.values()
            if c.claim_type in (ClaimType.IGT_PATTERN, ClaimType.ABSENCE_EVIDENCE)
        )
        n_counter   = sum(
            1 for c in graph.claims.values()
            if c.supports_hypothesis is False
        )

        for iteration in range(self.max_iter):
            self._attempt_contradiction_resolution(graph, question)

            unresolved = sum(1 for c in graph.contradictions if not c.resolved)
            can_conclude = (
                len(queries_fired) >= plan["min_queries_before_conclude"]
                and n_grammar_prose >= 1
                and (n_igt_quant >= 1 or not self._has_igt)
                and n_counter >= 1
                and unresolved == 0
            )

            hyp, conf, igt_note = graph.aggregate_confidence()

            if self._has_igt:
                igt_availability_note = ""
                constraints_met_str = (
                    f"grammar_prose={n_grammar_prose}≥1, "
                    f"igt_quant={n_igt_quant}≥1, "
                    f"counter={n_counter}≥1, "
                    f"unresolved_contradictions={unresolved}=0, "
                    f"queries={len(queries_fired)}≥{plan['min_queries_before_conclude']}"
                )
            else:
                igt_availability_note = (
                    "NOTE: No IGT corpus is loaded for this language. "
                    "Tools 5–11 (analyse_tag, analyse_construction, analyse_absence, "
                    "compare_tags, get_section_igt, search_translations, get_triline_examples) "
                    "are UNAVAILABLE and will return no data. "
                    "Focus exclusively on grammar prose tools (1–4). "
                    "IGT quantitative evidence is not required to conclude in this mode.\n"
                )
                constraints_met_str = (
                    f"grammar_prose={n_grammar_prose}≥1 [IGT not required — no corpus], "
                    f"counter={n_counter}≥1, "
                    f"unresolved_contradictions={unresolved}=0, "
                    f"queries={len(queries_fired)}≥{plan['min_queries_before_conclude']}"
                )

            decision_prompt = SEARCH_DECISION_PROMPT.format(
                language=self.language,
                question=question,
                domain=domain["domain_name"],
                plan_summary=json.dumps({
                    "remaining_sections": [
                        s for s in plan["target_sections"]
                        if s not in [q for q in queries_fired]
                    ],
                    "remaining_queries": [
                        q for q in plan["search_queries"]
                        if q not in queries_fired
                    ][:5],
                    "igt_tags": plan["igt_tags_to_check"] if self._has_igt else [],
                    "counter_framing": plan.get("counter_evidence_framing", ""),
                }, ensure_ascii=False),
                evidence_summary=graph.summarize()[:2500],
                gap_analysis=graph.get_gap_analysis(has_igt=self._has_igt)[:800],
                iteration=iteration + 1,
                max_iter=self.max_iter,
                constraints_met=constraints_met_str,
                igt_availability_note=igt_availability_note,
            )

            decision_raw = self.llm.generate(decision_prompt, max_new_tokens=512)
            decision     = self._parse_json(decision_raw)

            action  = decision.get("action", "")
            args    = decision.get("args", {})
            thought = decision.get("thought", "")
            ev_type = decision.get("evidence_type", "")
            claim_hint = decision.get("claim_to_add", "")
            supports   = decision.get("supports_hypothesis")

            hyp_str  = f"{hyp}({conf:.2f})"
            args_str = json.dumps(args, ensure_ascii=False)[:60]
            _print(
                f"  iter {iteration+1:02d}/{self.max_iter}  "
                f"action={action:<28}  hyp={hyp_str:<14}  "
                f"claims={len(graph.claims)}  args={args_str}",
                indent=4,
            )

            search_trace.append({
                "iteration": iteration + 1,
                "thought": thought,
                "action": action,
                "args": args,
                "evidence_type": ev_type,
                "hypothesis": hyp,
                "confidence": round(conf, 3),
            })

            if action == "conclude":
                if not can_conclude:
                    gap = graph.get_gap_analysis(has_igt=self._has_igt)
                    _print(f"       conclude blocked — {gap.splitlines()[1] if len(gap.splitlines())>1 else gap[:80]}", indent=4)
                    if self._has_igt and "NO IGT QUANTITATIVE" in gap:
                        action = "analyse_tag"
                        tag    = plan["igt_tags_to_check"][0] if plan["igt_tags_to_check"] else "PST"
                        args   = {"tag": tag}
                        ev_type = "igt_quantitative"
                    elif "NO GRAMMAR PROSE" in gap:
                        action = "read_full_section"
                        sec    = plan["target_sections"][0] if plan["target_sections"] else question
                        args   = {"query": sec}
                        ev_type = "grammar_prose"
                    elif "NO COUNTER-EVIDENCE" in gap:
                        if self._has_igt:
                            action  = "analyse_absence"
                            cat     = plan.get("category_absence_check") or domain["domain_name"]
                            args    = {"category": cat}
                            ev_type = "absence"
                        else:
                            remaining = [q for q in plan["search_queries"] if q not in queries_fired]
                            query = remaining[0] if remaining else plan.get("counter_evidence_framing", question)
                            action  = "search_text"
                            args    = {"query": query, "top_k": 5}
                            ev_type = "counter_evidence"
                    elif "UNRESOLVED CONTRADICTIONS" in gap:
                        action = "follow_cross_references"
                        args   = {"query": question}
                        ev_type = "grammar_prose"
                    else:
                        remaining = [q for q in plan["search_queries"] if q not in queries_fired]
                        query = remaining[0] if remaining else plan.get("counter_evidence_framing", question)
                        action  = "search_text"
                        args    = {"query": query, "top_k": 5}
                        ev_type = "counter_evidence"
                else:
                    _print(f"       conclude accepted  hyp={hyp}  conf={conf:.2f}", indent=4)
                    break

            observation, tool_igt_ids = self._execute_tool(
                action, args, preferred_sections=plan.get("target_sections", [])
            )
            if not observation:
                continue

            if action in ("search_text", "read_full_section",
                          "follow_cross_references", "extract_author_claims"):
                q = args.get("query", "")
                if q and q not in queries_fired:
                    queries_fired.append(q)

            raw_claim_type = self._map_evidence_type(ev_type)
            new_claims = self._extract_claims_from_observation(
                observation, question, ev_type
            )
            for claim_dict in new_claims:
                llm_ids  = [e for e in claim_dict.get("igt_examples", []) if isinstance(e, str) and not e.startswith("chunk_")]
                all_ids  = list(dict.fromkeys(tool_igt_ids + llm_ids))
                graph.add_claim(
                    text=claim_dict["text"],
                    claim_type=raw_claim_type,
                    source=f"{action}:{json.dumps(args)[:80]}",
                    confidence=claim_dict["confidence"],
                    supports_hypothesis=claim_dict.get("supports_hypothesis"),
                    igt_examples=all_ids,
                    raw_evidence=observation[:300],
                )

            if raw_claim_type == ClaimType.GRAMMAR_STATEMENT:
                n_grammar_prose += 1
            if raw_claim_type in (ClaimType.IGT_PATTERN, ClaimType.ABSENCE_EVIDENCE):
                n_igt_quant += 1
            if any(
                c.supports_hypothesis is False
                for c in graph.claims.values()
            ):
                n_counter += 1

            hyp, conf, igt_note = graph.aggregate_confidence()
            if can_conclude and conf >= self.conf_thresh:
                _print(
                    f"       early stop: constraints met, conf={conf:.2f} ≥ {self.conf_thresh}",
                    indent=4,
                )
                break

        # ── Step 4: Conclusion ──
        hyp, conf, igt_note = graph.aggregate_confidence()
        _print(
            f"  [5/5] Synthesizing conclusion  "
            f"(graph: {len(graph.claims)} claims, hyp={hyp}, conf={conf:.2f})"
            + (f"\n       ⚠ {igt_note}" if igt_note else "") + "...",
            indent=4,
        )

        cited_chunk_ids = []
        for claim in graph.claims.values():
            for eid in claim.igt_examples:
                if isinstance(eid, str) and eid.startswith("chunk_"):
                    if eid not in cited_chunk_ids:
                        cited_chunk_ids.append(eid)

        chunk_summaries = self.toolkit.get_chunk_summaries(cited_chunk_ids)

        conclusion_prompt = CONCLUSION_PROMPT.format(
            language=self.language,
            question=question,
            domain=domain["domain_name"],
            evidence_graph=graph.summarize()[:3000],
            chunk_summaries=chunk_summaries[:2000],
            trace_summary=(
                f"Iterations: {len(search_trace)}, "
                f"Queries: {len(queries_fired)}, "
                f"Claims: {len(graph.claims)}, "
                f"Contradictions: {len(graph.contradictions)}"
            ),
        )
        conclusion_raw = self.llm.generate(conclusion_prompt, max_new_tokens=3000)
        conclusion     = self._parse_json(conclusion_raw)

        # ── Step 5: Auditor ──
        _print("       running auditor...", indent=4)
        audit = self._run_auditor(question, conclusion, graph)

        final_value    = audit.get("revised_value",        conclusion.get("value", "?"))
        final_conf     = float(audit.get("revised_confidence",   conclusion.get("confidence", conf)))
        final_detail   = audit.get("revised_value_detail", conclusion.get("value_detail", ""))
        audit_changed  = audit.get("verdict") in ("overturned", "weakened")

        real_igt_ids = graph.get_igt_example_ids()
        igt_notes    = graph.get_igt_example_notes()
        igt_examples = self.toolkit.lookup_igt_examples(real_igt_ids, notes=igt_notes)

        feature = Feature(
            feature_id=f"{self.language[:3].upper()}_{feature_id}",
            question=question,
            domain=domain["domain_name"],
            linguistic_definition=conclusion.get("linguistic_definition", ""),
            structural_description=conclusion.get("structural_description", ""),
            value=final_value,
            value_detail=final_detail,
            confidence=final_conf,
            key_evidence=conclusion.get("key_evidence", []),
            igt_examples_used=igt_examples,
            igt_support=n_igt_quant > 0,
            search_trace=search_trace,
            typological_notes=conclusion.get("typological_notes", ""),
            needs_human_review=conclusion.get("needs_human_review", False) or audit_changed,
            review_reason=audit.get("audit_notes", conclusion.get("review_reason", "")),
            audit_verdict=audit.get("verdict", "upheld"),
            audit_objections=audit.get("objections", []),
            token_usage=self.llm.get_token_counts(),
        )

        W = 60
        _print("", indent=0)
        _print("  ┌─ FEATURE RESULT " + "─" * W, indent=2)
        _print(f"  │  Question : {feature.question}", indent=2)
        _print(f"  │  Value    : {feature.value}  (conf={feature.confidence:.2f})"
               f"  audit={feature.audit_verdict}", indent=2)
        tu = feature.token_usage
        _print(
            f"  │  Tokens   : in={tu['input_tokens']:,}  out={tu['output_tokens']:,}"
            f"  total={tu['total_tokens']:,}  calls={tu['llm_calls']}",
            indent=2,
        )
        if audit_changed:
            _print(f"  │           ← {feature.review_reason[:80]}", indent=2)
        if feature.audit_objections:
            for obj in feature.audit_objections:
                _print(f"  │  ! {obj}", indent=2)
        if feature.linguistic_definition:
            _print("  │", indent=2)
            _print("  │  DEFINITION", indent=2)
            for line in _wrap(feature.linguistic_definition, 95):
                _print(f"  │    {line}", indent=2)
        if feature.structural_description:
            _print("  │", indent=2)
            _print(f"  │  REALISATION IN {self.language.upper()}", indent=2)
            for line in _wrap(feature.structural_description, 95):
                _print(f"  │    {line}", indent=2)
        _print("  │", indent=2)
        _print(f"  │  VERDICT  : {feature.value_detail}", indent=2)
        _print("  │", indent=2)
        _print(f"  │  KEY EVIDENCE ({len(feature.key_evidence)} item(s))", indent=2)
        for i, ev in enumerate(feature.key_evidence, 1):
            _print(f"  │    {i}.", indent=2)
            for line in _wrap(str(ev), 93):
                _print(f"  │      {line}", indent=2)
        if feature.igt_examples_used:
            _print("  │", indent=2)
            _print(f"  │  IGT EXAMPLES ({len(feature.igt_examples_used)} cited)", indent=2)
            for ex in feature.igt_examples_used[:4]:
                _print(f"  │    [{ex['example_id']}]  "
                       f"'{ex.get('translation', '')[:65]}'", indent=2)
                if ex.get("morpheme"):
                    _print(f"  │      {ex['morpheme'][:80]}", indent=2)
                if ex.get("gloss"):
                    _print(f"  │      {ex['gloss'][:80]}", indent=2)
                if ex.get("note"):
                    _print(f"  │      → {ex['note'][:90]}", indent=2)
                _print(f"  │      [§ {ex.get('section','')[:60]}]", indent=2)
        if feature.typological_notes:
            _print("  │", indent=2)
            _print("  │  TYPOLOGICAL NOTES", indent=2)
            for line in _wrap(feature.typological_notes, 95):
                _print(f"  │    {line}", indent=2)
        if feature.needs_human_review:
            _print("  │", indent=2)
            _print(f"  │  ⚠ Review: {feature.review_reason[:100]}", indent=2)
        _print("  └─" + "─" * W, indent=2)
        _print("", indent=0)

        return feature

    # ── Auditor ───────────────────────────────────────────────────

    def _run_auditor(self, question: str, conclusion: dict, graph: EvidenceGraph) -> dict:
        prompt = AUDITOR_PROMPT.format(
            language=self.language,
            question=question,
            value=conclusion.get("value", "?"),
            value_detail=conclusion.get("value_detail", ""),
            confidence=conclusion.get("confidence", 0.0),
            evidence_graph_summary=graph.summarize()[:2000],
        )
        response = self.llm.generate(prompt, max_new_tokens=768)
        audit    = self._parse_json(response)
        audit.setdefault("verdict", "upheld")
        audit.setdefault("revised_value",        conclusion.get("value", "?"))
        audit.setdefault("revised_confidence",   conclusion.get("confidence", 0.0))
        audit.setdefault("revised_value_detail", conclusion.get("value_detail", ""))
        return audit

    # ── Free-form deep query ──────────────────────────────────────

    def build_query_plan(self, query: str) -> dict:
        mode_suffix = " + IGT" if self._has_igt else " [grammar-only, no IGT]"
        _print(f"  [1/4] Building search plan from TOC{mode_suffix}...", indent=2)

        toc_with_summaries = self.toolkit.get_toc_with_summaries(
            skip_subsections=False, max_summary_chars=80
        )
        igt_summary   = self.toolkit.get_igt_summary()
        abbrev_legend = self._abbrev.prompt_legend()

        abbrev_candidates = self._format_abbrev_candidates(query)

        _qplan_igt_note = (
            ""
            if self._has_igt
            else (
                "NOTE: No IGT corpus is loaded. Leave igt_tags_to_check, "
                "constructions_to_check, category_absence_check, and tags_to_compare "
                "as empty lists/strings — those tools are unavailable.\n"
            )
        )

        prompt = QUERY_PLAN_PROMPT.format(
            language=self.language,
            query=query,
            abbrev_legend=abbrev_legend,
            abbrev_candidates=abbrev_candidates,
            toc_with_summaries=toc_with_summaries[:12000],
            igt_summary=igt_summary[:2000],
            igt_availability_note=_qplan_igt_note,
        )
        response = self.llm.generate(prompt, max_new_tokens=1024)
        plan     = self._parse_json(response)

        plan.setdefault("phenomena",    [])
        plan.setdefault("rationale",    "")
        plan.setdefault("target_sections", [])
        plan.setdefault("search_queries",  [query])
        if not self._has_igt:
            plan["igt_tags_to_check"]      = []
            plan["constructions_to_check"] = []
            plan["category_absence_check"] = ""
            plan["tags_to_compare"]        = []
        else:
            plan.setdefault("igt_tags_to_check",      [])
            plan.setdefault("constructions_to_check", [])
            plan.setdefault("category_absence_check", "")
            plan.setdefault("tags_to_compare",        [])

        _print(f"     phenomena      : {plan['phenomena']}", indent=2)
        _print(f"     rationale      : {plan['rationale'][:80]}", indent=2)
        _print(f"     target_sections: {plan['target_sections']}", indent=2)
        _print(f"     igt_tags       : {plan['igt_tags_to_check']}", indent=2)
        _print(f"     absence_check  : {plan['category_absence_check'] or '—'}", indent=2)

        return plan

    def answer_query(self, query: str, max_iterations: int = 8) -> dict:
        W = 60
        _print(f"\n{'='*W}")
        _print(f"Deep Query: {query}")
        _print(f"{'='*W}")

        self.llm.reset_token_counter()   # start fresh for this query
        graph         = EvidenceGraph(query)
        search_trace  = []
        sections_read = []

        plan = self.build_query_plan(query)

        pseudo_feature = {
            "question":       query,
            "feature_id":     "Q001",
            "igt_signals":    [],
        }

        _print(f"  [2/4] Reading {len(plan['target_sections'])} target section(s)...", indent=2)
        self.read_target_sections(plan, pseudo_feature, graph)
        sections_read = list(plan["target_sections"])

        total_igt_ops = (
            len(plan["igt_tags_to_check"][:5])
            + len(plan["constructions_to_check"][:3])
            + len(plan["tags_to_compare"][:2])
            + (1 if plan["category_absence_check"] else 0)
        )
        _print(f"  [3/4] IGT pass ({total_igt_ops} operation(s))...", indent=2)
        self.run_igt_initial_pass(plan, pseudo_feature, graph)

        _print(
            f"  [3/4 done] Plan pass complete: "
            f"{len(graph.claims)} claim(s), "
            f"{len(graph.contradictions)} contradiction(s)",
            indent=2,
        )

        _print(f"  [4/4] ReAct follow-up (max {max_iterations} iterations)...", indent=2)

        _query_igt_note = (
            ""
            if self._has_igt
            else (
                "NOTE: No IGT corpus is loaded. "
                "Tools 5–11 (analyse_tag, analyse_construction, analyse_absence, "
                "compare_tags, get_section_igt, search_translations, get_triline_examples) "
                "are UNAVAILABLE. "
                "Use only grammar prose tools (1–4).\n"
            )
        )

        for iteration in range(max_iterations):
            can_conclude = (
                len(graph.claims) >= 3
                and len(sections_read) >= 1
            )

            decision_prompt = QUERY_DECISION_PROMPT.format(
                language=self.language,
                query=query,
                evidence_summary=graph.summarize()[:2500],
                iteration=iteration + 1,
                max_iter=max_iterations,
                sections_read=", ".join(sections_read[:5]) if sections_read else "none",
                igt_availability_note=_query_igt_note,
            )

            decision_raw = self.llm.generate(decision_prompt, max_new_tokens=512)
            decision     = self._parse_json(decision_raw)

            action  = decision.get("action", "conclude")
            args    = decision.get("args", {})
            thought = decision.get("thought", "")
            finding = decision.get("finding", "")

            args_str = json.dumps(args, ensure_ascii=False)[:55]
            _print(
                f"  iter {iteration+1:02d}/{max_iterations}"
                f"  action={action:<28}  claims={len(graph.claims)}"
                f"  args={args_str}",
                indent=2,
            )

            search_trace.append({
                "iteration": iteration + 1,
                "thought":   thought,
                "action":    action,
                "args":      args,
                "finding":   finding,
            })

            if action == "conclude":
                if can_conclude:
                    _print(f"       conclude accepted  claims={len(graph.claims)}", indent=2)
                    break
                else:
                    _print(f"       conclude blocked (need ≥3 claims + ≥1 section)", indent=2)
                    remaining = [q for q in plan["search_queries"] if q not in sections_read]
                    fallback  = remaining[0] if remaining else query
                    action = "search_text"
                    args   = {"query": fallback, "top_k": 4}

            observation, tool_igt_ids = self._execute_tool(
                action, args, preferred_sections=plan.get("target_sections", [])
            )
            if not observation:
                continue

            if action in ("read_full_section", "follow_cross_references", "extract_author_claims"):
                sec_name = args.get("query", "")
                if sec_name and sec_name not in sections_read:
                    sections_read.append(sec_name)

            ev_type_map = {
                "read_full_section":       "grammar_prose",
                "follow_cross_references": "grammar_prose",
                "extract_author_claims":   "grammar_prose",
                "search_text":             "grammar_prose",
                "analyse_tag":             "igt_quantitative",
                "analyse_construction":    "igt_quantitative",
                "analyse_absence":         "absence",
                "compare_tags":            "igt_quantitative",
                "get_section_igt":         "igt_quantitative",
                "search_translations":     "igt_quantitative",
                "get_triline_examples":    "igt_quantitative",
            }
            ev_type = ev_type_map.get(action, "grammar_prose")

            new_claims = self._extract_claims_from_observation(observation, query, ev_type)
            for claim_dict in new_claims:
                llm_ids = [
                    e for e in claim_dict.get("igt_examples", [])
                    if isinstance(e, str) and not e.startswith("chunk_")
                ]
                graph.add_claim(
                    text=claim_dict["text"],
                    claim_type=self._map_evidence_type(ev_type),
                    source=f"{action}:{json.dumps(args, ensure_ascii=False)[:60]}",
                    confidence=claim_dict["confidence"],
                    supports_hypothesis=claim_dict.get("supports_hypothesis"),
                    igt_examples=list(dict.fromkeys(tool_igt_ids + llm_ids)),
                    raw_evidence=observation[:300],
                )

        # ── Synthesis ────────────────────────────────────────────
        _print(f"\n  Synthesizing answer (claims={len(graph.claims)})...", indent=2)

        cited_chunk_ids = []
        for claim in graph.claims.values():
            for eid in claim.igt_examples:
                if isinstance(eid, str) and eid.startswith("chunk_") and eid not in cited_chunk_ids:
                    cited_chunk_ids.append(eid)

        chunk_summaries = self.toolkit.get_chunk_summaries(cited_chunk_ids)

        n_igt_quant = sum(
            1 for c in graph.claims.values()
            if c.claim_type in (ClaimType.IGT_PATTERN, ClaimType.ABSENCE_EVIDENCE)
        )

        conclusion_prompt = QUERY_CONCLUSION_PROMPT.format(
            language=self.language,
            query=query,
            evidence_graph=graph.summarize()[:3000],
            chunk_summaries=chunk_summaries[:2000],
            trace_summary=(
                f"Iterations: {len(search_trace)}, "
                f"Sections read: {len(sections_read)}, "
                f"Claims: {len(graph.claims)}, "
                f"IGT ops: {n_igt_quant}"
            ),
        )
        conclusion_raw = self.llm.generate(conclusion_prompt, max_new_tokens=3000)
        conclusion     = self._parse_json(conclusion_raw)

        if not conclusion.get("answer", "").strip():
            _print("       answer empty — retrying with simple prompt...", indent=2)
            evidence_text = "\n".join(
                f"- [{c.claim_type.value}] {c.text} (source: {c.source})"
                for c in graph.claims.values()
                if c.claim_type.value in ("grammar_statement", "igt_pattern", "absence_evidence")
            )[:2000]
            retry_prompt = (
                f"Based on the following evidence from the reference grammar of {self.language}, "
                f"write a 2-3 paragraph answer to this query: {query}\n\n"
                f"EVIDENCE:\n{evidence_text}\n\n"
                f"Write only the answer paragraphs, no JSON, no preamble."
            )
            answer_text = self.llm.generate(retry_prompt, max_new_tokens=1000, json_mode=False)
            conclusion["answer"] = answer_text.strip()
            _print(f"       retry produced {len(conclusion['answer'])} chars", indent=2)

        real_igt_ids = graph.get_igt_example_ids()
        igt_notes    = graph.get_igt_example_notes()
        igt_examples = self.toolkit.lookup_igt_examples(real_igt_ids, notes=igt_notes)

        _print("       running auditor...", indent=2)
        answer_summary = conclusion.get("answer", "")[:120]
        audit_prompt = QUERY_AUDITOR_PROMPT.format(
            language=self.language,
            query=query,
            answer_summary=answer_summary,
            confidence=conclusion.get("confidence", 0.5),
            evidence_graph_summary=graph.summarize()[:2000],
        )
        audit_raw = self.llm.generate(audit_prompt, max_new_tokens=512)
        audit     = self._parse_json(audit_raw)
        audit.setdefault("verdict",            "upheld")
        audit.setdefault("revised_confidence", conclusion.get("confidence", 0.5))
        audit.setdefault("audit_notes",        "")
        audit.setdefault("objections",         [])

        audit_changed = audit.get("verdict") in ("overturned", "weakened")
        final_conf    = float(audit.get("revised_confidence", conclusion.get("confidence", 0.5)))

        from state import QueryResult
        result = QueryResult(
            query_id=re.sub(r"[^\w]", "_", query[:30]).lower(),
            query=query,
            phenomena=plan["phenomena"],
            linguistic_definition=conclusion.get("linguistic_definition", ""),
            structural_description=conclusion.get("structural_description", ""),
            answer=conclusion.get("answer", ""),
            key_evidence=conclusion.get("key_evidence", []),
            igt_examples_used=igt_examples,
            igt_support=n_igt_quant > 0,
            search_trace=search_trace,
            confidence=final_conf,
            needs_human_review=conclusion.get("needs_human_review", False) or audit_changed,
            review_reason=audit.get("audit_notes", conclusion.get("review_reason", "")),
            audit_verdict=audit.get("verdict", "upheld"),
            audit_objections=audit.get("objections", []),
            token_usage=self.llm.get_token_counts(),
        )

        _print("", indent=0)
        _print("  ┌─ QUERY RESULT " + "─" * W, indent=2)
        _print(f"  │  Query      : {result.query}", indent=2)
        _print(f"  │  Phenomena  : {', '.join(result.phenomena)}", indent=2)
        _print(
            f"  │  Confidence : {result.confidence:.2f}"
            f"  audit={result.audit_verdict}",
            indent=2,
        )
        tu = result.token_usage
        _print(
            f"  │  Tokens   : in={tu['input_tokens']:,}  out={tu['output_tokens']:,}"
            f"  total={tu['total_tokens']:,}  calls={tu['llm_calls']}",
            indent=2,
        )
        if audit_changed:
            _print(f"  │           ← {result.review_reason[:80]}", indent=2)
        if result.audit_objections:
            for obj in result.audit_objections:
                _print(f"  │  ! {obj}", indent=2)
        if result.linguistic_definition:
            _print("  │", indent=2)
            _print("  │  DEFINITION", indent=2)
            for line in _wrap(result.linguistic_definition, 95):
                _print(f"  │    {line}", indent=2)
        if result.structural_description:
            _print("  │", indent=2)
            _print(f"  │  REALISATION IN {self.language.upper()}", indent=2)
            for line in _wrap(result.structural_description, 95):
                _print(f"  │    {line}", indent=2)
        _print("  │", indent=2)
        _print("  │  ANSWER", indent=2)
        for line in _wrap(result.answer[:600], 93):
            _print(f"  │    {line}", indent=2)
        if result.key_evidence:
            _print("  │", indent=2)
            _print(f"  │  KEY EVIDENCE ({len(result.key_evidence)} item(s))", indent=2)
            for i, ev in enumerate(result.key_evidence, 1):
                _print(f"  │    {i}.", indent=2)
                for line in _wrap(str(ev), 93):
                    _print(f"  │      {line}", indent=2)
        if result.igt_examples_used:
            _print("  │", indent=2)
            _print(f"  │  IGT EXAMPLES ({len(result.igt_examples_used)} cited)", indent=2)
            for ex in result.igt_examples_used[:4]:
                _print(
                    f"  │    [{ex['example_id']}]  "
                    f"'{ex.get('translation', '')[:65]}'",
                    indent=2,
                )
                if ex.get("morpheme"):
                    _print(f"  │      {ex['morpheme'][:80]}", indent=2)
                if ex.get("gloss"):
                    _print(f"  │      {ex['gloss'][:80]}", indent=2)
                if ex.get("note"):
                    _print(f"  │      → {ex['note'][:90]}", indent=2)
                _print(f"  │      [§ {ex.get('section','')[:60]}]", indent=2)
        if result.needs_human_review:
            _print("  │", indent=2)
            _print(f"  │  ⚠ Review: {result.review_reason[:100]}", indent=2)
        _print("  └─" + "─" * W, indent=2)
        _print("", indent=0)

        return result

    # ── Main run ──────────────────────────────────────────────────

    def run(self) -> EpistemicState:
        domains = self.discover_domains()
        self.state.domains = domains

        total_features = sum(len(d.get("candidate_features", [])) for d in domains)
        feature_num = 0
        # Safety net: track investigated questions so any duplicates that
        # somehow survived _dedup_candidates are not investigated twice.
        investigated_questions: set[str] = set()

        for domain in domains:
            _print(f"\n{'─'*60}", indent=0)
            _print(f"Domain: {domain['domain_name']}  ({len(domain.get('candidate_features', []))} features)", indent=0)
            _print(f"{'─'*60}", indent=0)
            for candidate in domain.get("candidate_features", []):
                feature_num += 1
                q_key = candidate["question"].strip().lower()
                if q_key in investigated_questions:
                    _print(
                        f"\n[Feature {feature_num}/{total_features}] SKIP (already investigated): "
                        f"{candidate['question'][:70]}",
                        indent=0,
                    )
                    continue
                investigated_questions.add(q_key)
                _print(f"\n[Feature {feature_num}/{total_features}]", indent=0)
                feature = self.investigate_feature(candidate, domain)
                if feature:
                    if feature.confidence >= self.conf_thresh:
                        self.state.confirmed_features.append(feature)
                    else:
                        self.state.uncertain_features.append(feature)

        _print(f"\n{'='*60}", indent=0)
        _print(
            f"[{self.language}] Done.  "
            f"Confirmed: {len(self.state.confirmed_features)}  "
            f"Uncertain: {len(self.state.uncertain_features)}",
            indent=0,
        )
        _print(f"{'='*60}", indent=0)
        return self.state

    # ── Helpers ───────────────────────────────────────────────────

    def _get_section_summary(self, sec_ref: str) -> str:
        node = self.toolkit.section_reader.get_full_section_node(sec_ref)
        if node:
            chunk = self.toolkit._chunk_by_id.get(node.section_id)
            if chunk and chunk.summary:
                return chunk.summary.strip()
            chunk = self.toolkit._chunk_by_id.get(node.section_id + "_p0")
            if chunk and chunk.summary:
                return chunk.summary.strip()

        chunk = self.toolkit._chunk_by_id.get(sec_ref)
        if chunk and chunk.summary:
            return chunk.summary.strip()

        chunk = self.toolkit._chunk_by_id.get(sec_ref + "_p0")
        if chunk and chunk.summary:
            return chunk.summary.strip()

        return ""

    _VALID_CLAIM_TYPES = {
        "grammar_statement", "igt_pattern", "absence_evidence",
        "inference", "counter_evidence", "author_caveat",
    }

    def _normalize_claim_type(self, raw_type: str) -> str:
        if not raw_type:
            return "inference"
        for token in re.split(r"[|/,; ]+", raw_type.strip()):
            token = token.strip().lower().replace(" ", "_")
            if token in self._VALID_CLAIM_TYPES:
                return token
        return "inference"

    def _extract_claims_from_observation(
        self,
        observation: str,
        question: str,
        evidence_type: str,
    ) -> list:
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            language=self.language,
            question=question,
            observation=observation[:1200],
        )
        response = self.llm.generate(prompt, max_new_tokens=1024)
        parsed   = self._parse_json(response)
        claims   = parsed.get("claims", [])
        for c in claims:
            c.setdefault("text", observation[:100])
            c["type"] = self._normalize_claim_type(c.get("type", ""))
            c.setdefault("confidence", 0.5)
            c.setdefault("supports_hypothesis", None)
            c.setdefault("igt_examples", [])
        return claims

    def _attempt_contradiction_resolution(self, graph: EvidenceGraph, question: str):
        for contra in graph.contradictions:
            if contra.resolved:
                continue
            claim_a = graph.claims.get(contra.claim_a_id)
            claim_b = graph.claims.get(contra.claim_b_id)
            if not claim_a or not claim_b:
                continue

            prompt = CONTRADICTION_RESOLUTION_PROMPT.format(
                language=self.language,
                question=question,
                contradiction_description=contra.description,
                claim_a_text=claim_a.text,
                claim_a_source=claim_a.source,
                claim_a_evidence=claim_a.raw_evidence[:300],
                claim_b_text=claim_b.text,
                claim_b_source=claim_b.source,
                claim_b_evidence=claim_b.raw_evidence[:300],
            )
            response   = self.llm.generate(prompt, max_new_tokens=512)
            resolution = self._parse_json(response)

            if resolution.get("resolved", False):
                contra.resolved   = True
                contra.resolution = resolution.get("resolution", "")
                rev_a = resolution.get("revised_confidence_a")
                rev_b = resolution.get("revised_confidence_b")
                if rev_a is not None:
                    claim_a.confidence = float(rev_a)
                if rev_b is not None:
                    claim_b.confidence = float(rev_b)
                _print(f"       contradiction resolved: {contra.description[:70]}", indent=4)

    def _execute_tool(self, action: str, args: dict, preferred_sections: list = None) -> tuple:
        ps = preferred_sections or []
        try:
            if action == "read_full_section":
                sr = self.toolkit.read_full_section(args.get("query", ""))
                return sr.text, sr.chunk_ids
            elif action == "follow_cross_references":
                sr = self.toolkit.follow_cross_references(args.get("query", ""))
                return sr.text, sr.chunk_ids
            elif action == "extract_author_claims":
                return self.toolkit.extract_author_claims(args.get("query", "")), []
            elif action == "search_text":
                return self.toolkit.search_text(
                    query=args.get("query", ""),
                    top_k=args.get("top_k", 5),
                ), []
            elif action == "analyse_tag":
                r = self.toolkit.analyse_tag(args.get("tag", ""), preferred_sections=ps)
                return r.text, r.igt_ids
            elif action == "analyse_construction":
                r = self.toolkit.analyse_construction(args.get("tags", []), preferred_sections=ps)
                return r.text, r.igt_ids
            elif action == "analyse_absence":
                r = self.toolkit.analyse_absence(args.get("category", ""), preferred_sections=ps)
                return r.text, r.igt_ids
            elif action == "compare_tags":
                r = self.toolkit.compare_tags(args.get("tag_a", ""), args.get("tag_b", ""), preferred_sections=ps)
                return r.text, r.igt_ids
            elif action == "get_section_igt":
                r = self.toolkit.get_section_igt(args.get("section_query", ""))
                return r.text, r.igt_ids
            elif action == "search_translations":
                r = self.toolkit.search_translations(
                    args.get("query", ""),
                    max_results=args.get("max_results", 15),
                )
                return r.text, r.igt_ids
            elif action == "get_triline_examples":
                r = self.toolkit.get_triline_examples(
                    args.get("query", ""),
                    max_examples=args.get("max_examples", 8),
                )
                return r.text, r.igt_ids
            elif action == "get_igt_candidates":
                r = self.toolkit.get_igt_candidates(
                    args.get("query_tags", []), args.get("query_text", "")
                )
                return r.text, r.igt_ids
        except Exception as e:
            logger.warning(f"Tool error ({action}): {e}")
        return "", []

    def _map_evidence_type(self, ev_type: str) -> ClaimType:
        return {
            "grammar_prose":    ClaimType.GRAMMAR_STATEMENT,
            "igt_quantitative": ClaimType.IGT_PATTERN,
            "absence":          ClaimType.ABSENCE_EVIDENCE,
            "construction":     ClaimType.IGT_PATTERN,
            "counter_evidence": ClaimType.COUNTER_EVIDENCE,
        }.get(ev_type, ClaimType.INFERENCE)

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
                logger.warning(f"_parse_json: no JSON object found. Raw: {original[:200]}")
                return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.debug(f"_parse_json: initial parse failed ({e}), trying truncation recovery")
            for i in range(len(text) - 1, max(len(text) - 500, 0), -1):
                if text[i] == "}":
                    try:
                        result = json.loads(text[:i+1])
                        logger.debug(f"_parse_json: recovered by truncating to pos {i}")
                        return result
                    except json.JSONDecodeError:
                        continue

            for suffix in ['"}', '"]}', '"]}}'  , ']}', ']}}'  , '}', '}}']:
                try:
                    result = json.loads(text + suffix)
                    logger.debug(f"_parse_json: recovered by appending {repr(suffix)}")
                    return result
                except json.JSONDecodeError:
                    continue

            logger.warning(
                f"_parse_json: all recovery attempts failed. "
                f"Raw response (first 300 chars): {original[:300]}"
            )
            return {}


# ═══════════════════════════════════════════════════════════════
# Pipeline entry points
# ═══════════════════════════════════════════════════════════════

@dataclass
class LanguageConfig:
    language: str
    grammar_path: str
    igt_path: Optional[str] = None


def run_deep_pipeline(
    language_configs: list,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "output",
    max_iterations_per_feature: int = 15,
    confidence_threshold: float = 0.75,
    min_queries_per_feature: int = 5,
    use_vllm: bool = False,
    abbreviations_path: Optional[str] = None,    # ← NEW
) -> dict:
    from llm import QwenLLM
    from state import EpistemicState

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm    = QwenLLM(model_name, use_vllm=use_vllm)
    states = []

    for config in language_configs:
        _print(f"\n{'='*60}")
        _print(f"Language: {config.language}")
        _print(f"{'='*60}")
        agent = DeepLanguageResearchAgent(
            language=config.language,
            grammar_path=config.grammar_path,
            igt_path=config.igt_path,
            llm=llm,
            max_iterations_per_feature=max_iterations_per_feature,
            confidence_threshold=confidence_threshold,
            min_queries_per_feature=min_queries_per_feature,
            abbreviations_path=abbreviations_path,             # ← NEW
        )
        state = agent.run()
        states.append(state)

        out_file = output_path / f"{config.language.lower()}_features.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        _print(f"Saved: {out_file}")

    result = {"per_language": {s.language: s.to_dict() for s in states}}

    if len(states) > 1:
        try:
            from agent import AlignmentAgent, _save_csv_matrix
        except ImportError:
            logger.warning(
                "agent.py not found — cross-language alignment skipped. "
                "To enable alignment, ensure agent.py is present in the same directory."
            )
            return result
        aligner   = AlignmentAgent(llm)
        alignment = aligner.align(states)
        if alignment:
            result["alignment"] = alignment
            with open(output_path / "aligned_feature_matrix.json", "w") as f:
                json.dump(alignment, f, ensure_ascii=False, indent=2)
            _save_csv_matrix(alignment, output_path / "feature_matrix.csv")

    return result


def run_query_pipeline(
    language: str,
    grammar_path: str,
    queries: list,
    igt_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "output",
    max_iterations: int = 10,
    use_vllm: bool = False,
    abbreviations_path: Optional[str] = None,    # ← NEW
) -> list:
    from llm import QwenLLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm   = QwenLLM(model_name, use_vllm=use_vllm)
    agent = DeepLanguageResearchAgent(
        language=language,
        grammar_path=grammar_path,
        igt_path=igt_path,
        llm=llm,
        max_iterations_per_feature=max_iterations,
        abbreviations_path=abbreviations_path,               # ← NEW
    )

    results = []
    for i, query in enumerate(queries, 1):
        _print(f"\n[Query {i}/{len(queries)}]")
        result = agent.answer_query(query, max_iterations=max_iterations)
        results.append(result)

        safe_name = re.sub(r"[^\w\s-]", "", query[:40]).strip().replace(" ", "_").lower()
        out_file  = output_path / f"query_{i:02d}_{safe_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        _print(f"Saved: {out_file}")

    lang_slug     = language.lower().replace(" ", "_")
    combined_file = output_path / f"{lang_slug}_queries.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(
            {"language": language, "queries": [r.to_dict() for r in results]},
            f, ensure_ascii=False, indent=2,
        )
    _print(f"Combined results saved: {combined_file}")

    return results
