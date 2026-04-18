"""
igt_analysis.py — Quantitative IGT Analytics
==============================================
Provides the statistical grounding that lets the agent reason from
*data* rather than only from the author's prose descriptions.

Key analyses:
  - Tag frequency and obligatoriness
  - Tag positional profiles (preverbal / postverbal / flexible)
  - Tag co-occurrence (construction fingerprints)
  - Per-section tag distribution (where does each category appear?)
  - Absence detection (tags that almost never appear → category absent)
  - Paradigm completeness (does every person/number slot appear?)
  - Negative evidence scoring (how confidently can we say X is absent?)
"""

import re
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class TagProfile:
    tag: str
    count: int                  # raw frequency
    example_coverage: float     # fraction of all IGT examples containing this tag
    mean_position: float        # 0=always first, 1=always last in gloss line
    position_std: float         # low=consistent position, high=flexible
    position_label: str         # "preverbal" | "postverbal" | "flexible" | "initial" | "final"
    top_cooccurrents: list      # [(tag, count), ...] most frequent co-occurring tags
    section_distribution: dict  # {section: count}
    top_sections: list          # [(section, count), ...] top 5


@dataclass
class ConstructionPattern:
    """A recurring sequence or co-occurrence of gloss tags."""
    tags: tuple                 # ordered tag n-gram or unordered pair
    count: int
    example_ids: list           # up to 5 example IDs
    typical_translations: list  # up to 3 translations
    sections: list              # sections where this pattern appears


@dataclass
class IGTStats:
    total_examples: int
    unique_tags: int
    tag_profiles: dict          # tag → TagProfile
    constructions: list         # ConstructionPattern list (top bigrams + trigrams)
    absent_categories: list     # tags/categories with near-zero evidence
    paradigm_slots: dict        # "person_number" → coverage analysis
    section_richness: dict      # section → {examples, unique_tags, top_tags}
    summary_text: str           # human-readable summary for LLM consumption


# ─────────────────────────────────────────────
# Core analyser
# ─────────────────────────────────────────────

class IGTAnalyser:

    # Known typological category clusters — used for absence detection
    TYPOLOGICAL_CLUSTERS = {
        "TENSE": ["PST", "PAST", "PRS", "FUT", "ANT", "REM", "REC", "HOD", "DIST"],
        "ASPECT": ["PFV", "IPFV", "PRF", "PROG", "HAB", "ITER", "INCEP", "CONT", "PROSP"],
        "MODALITY": ["POT", "SBJV", "COND", "IRR", "NECES", "OBLIG", "ABIL", "PERM", "DEONT", "EPIS"],
        "EVIDENTIALITY": ["EVID", "REP", "QUOT.EVID", "VIS", "INFER", "DIR", "INDIR", "HEARSAY"],
        "AGREEMENT": ["1SG.AGR", "2SG.AGR", "3SG.AGR", "1PL.AGR", "2PL.AGR", "3PL.AGR",
                      "SBJ.AGR", "OBJ.AGR", "M.AGR", "F.AGR"],
        "CASE": ["NOM", "ACC", "DAT", "GEN", "ERG", "ABS", "INSTR", "COMIT", "CAUS.CASE"],
        "GENDER": ["M", "F", "MASC", "FEM", "NEUT", "N", "CL1", "CL2", "CL3"],
        "NUMBER": ["SG", "PL", "DU", "TRIAL", "PAUCAL"],
        "NEGATION": ["NEG", "NEG.FOC", "NEG.EXIST", "NEG.PRED", "PROH"],
        "FOCUS": ["FOC", "TOP", "CONT.FOC", "ID.FOC", "NEG.FOC", "Q.FOC"],
    }

    def __init__(self, igt_examples: list):
        """
        igt_examples: list of IGTExample dataclass instances from tools.py
        """
        self.examples = igt_examples
        self.n = len(igt_examples)
        self._stats: Optional[IGTStats] = None

    def get_stats(self) -> IGTStats:
        if self._stats is None:
            self._stats = self._compute()
        return self._stats

    def _compute(self) -> IGTStats:
        tag_counts      = Counter()
        tag_positions   = defaultdict(list)
        tag_cooccur     = Counter()
        tag_sections    = defaultdict(Counter)
        bigram_counts   = Counter()
        trigram_counts  = Counter()
        bigram_examples = defaultdict(list)
        bigram_trans    = defaultdict(list)

        for ex in self.examples:
            tags = ex.gloss_tags
            n = len(tags)
            section = ex.source_section or ex.section

            for i, tag in enumerate(tags):
                tag_counts[tag] += 1
                tag_sections[tag][section] += 1
                if n > 1:
                    tag_positions[tag].append(i / (n - 1))
                else:
                    tag_positions[tag].append(0.5)

            # Pairwise co-occurrence (unordered)
            for i in range(n):
                for j in range(i + 1, n):
                    pair = tuple(sorted([tags[i], tags[j]]))
                    tag_cooccur[pair] += 1

            # Ordered bigrams (captures syntax/morphology order)
            for i in range(n - 1):
                bg = (tags[i], tags[i + 1])
                bigram_counts[bg] += 1
                if len(bigram_examples[bg]) < 5:
                    bigram_examples[bg].append(ex.example_id)
                if len(bigram_trans[bg]) < 3 and ex.translation:
                    bigram_trans[bg].append(ex.translation[:80])

            # Ordered trigrams
            for i in range(n - 2):
                tg = (tags[i], tags[i + 1], tags[i + 2])
                trigram_counts[tg] += 1

        # ── Tag profiles ──
        tag_profiles = {}
        for tag, count in tag_counts.items():
            positions = tag_positions[tag]
            mean_pos  = statistics.mean(positions)
            std_pos   = statistics.stdev(positions) if len(positions) > 1 else 0.0

            # Position label
            if std_pos < 0.15:
                if mean_pos < 0.25:
                    plabel = "preverbal/initial"
                elif mean_pos > 0.75:
                    plabel = "postverbal/final"
                else:
                    plabel = "medial-fixed"
            else:
                plabel = "flexible"

            # Top co-occurrents for this tag
            cooc = [
                (pair[1] if pair[0] == tag else pair[0], cnt)
                for pair, cnt in tag_cooccur.most_common()
                if tag in pair and (pair[0] == tag or pair[1] == tag)
            ][:8]

            sec_dist  = dict(tag_sections[tag])
            top_secs  = tag_sections[tag].most_common(5)

            tag_profiles[tag] = TagProfile(
                tag=tag,
                count=count,
                example_coverage=count / self.n,
                mean_position=round(mean_pos, 3),
                position_std=round(std_pos, 3),
                position_label=plabel,
                top_cooccurrents=cooc,
                section_distribution=sec_dist,
                top_sections=top_secs,
            )

        # ── Construction patterns (top bigrams + trigrams) ──
        constructions = []
        for bg, cnt in bigram_counts.most_common(40):
            if cnt < 5:
                break
            secs = list({
                ex.source_section or ex.section
                for ex in self.examples
                if len(ex.gloss_tags) > 1
                and any(
                    ex.gloss_tags[i] == bg[0] and ex.gloss_tags[i+1] == bg[1]
                    for i in range(len(ex.gloss_tags) - 1)
                )
            })[:5]
            constructions.append(ConstructionPattern(
                tags=bg,
                count=cnt,
                example_ids=bigram_examples[bg],
                typical_translations=bigram_trans[bg],
                sections=secs,
            ))
        for tg, cnt in trigram_counts.most_common(20):
            if cnt < 5:
                break
            constructions.append(ConstructionPattern(
                tags=tg,
                count=cnt,
                example_ids=[],
                typical_translations=[],
                sections=[],
            ))

        # ── Absent category detection ──
        absent_categories = []
        for category, candidates in self.TYPOLOGICAL_CLUSTERS.items():
            hits = sum(tag_counts.get(c, 0) for c in candidates)
            # None of the canonical tags for this category appear at all
            if hits == 0:
                absent_categories.append({
                    "category": category,
                    "evidence": "zero",
                    "note": f"None of {candidates[:4]} appear in {self.n} IGT examples",
                })
            # Extremely rare relative to corpus size
            elif hits / self.n < 0.005:
                present_tags = [(c, tag_counts[c]) for c in candidates if tag_counts.get(c, 0) > 0]
                absent_categories.append({
                    "category": category,
                    "evidence": "marginal",
                    "rate": round(hits / self.n, 4),
                    "present_tags": present_tags,
                    "note": f"Only {hits} hits across {self.n} examples ({100*hits/self.n:.2f}%)",
                })

        # ── Paradigm completeness ──
        person_number_tags = {
            "1SG": ["1SG", "1SG.SBJ", "1SG.OBJ", "1SG.INDP", "1SG.POSS"],
            "2SG": ["2SG", "2SG.SBJ", "2SG.OBJ", "2SG.INDP", "2SG.POSS"],
            "3SG": ["3SG", "3SG.SBJ", "3SG.OBJ", "3SG.INDP", "3SG.POSS"],
            "1PL": ["1PL", "1PL.SBJ", "1PL.OBJ", "1PL.INDP", "WE"],
            "2PL": ["2PL", "2PL.SBJ", "2PL.OBJ", "2PL.INDP", "UNA"],
            "3PL": ["3PL", "3PL.SBJ", "3PL.OBJ", "3PL.INDP", "DEN"],
        }
        paradigm_slots = {}
        for slot, tags_for_slot in person_number_tags.items():
            total_hits = sum(tag_counts.get(t, 0) for t in tags_for_slot)
            present    = [(t, tag_counts[t]) for t in tags_for_slot if tag_counts.get(t, 0) > 0]
            paradigm_slots[slot] = {
                "total_hits": total_hits,
                "present_forms": present,
                "attested": total_hits > 0,
            }

        # ── Section richness ──
        section_examples = defaultdict(list)
        for ex in self.examples:
            sec = ex.source_section or ex.section
            section_examples[sec].append(ex)

        section_richness = {}
        for sec, exs in section_examples.items():
            tags_in_sec = Counter(t for ex in exs for t in ex.gloss_tags)
            section_richness[sec] = {
                "n_examples": len(exs),
                "unique_tags": len(tags_in_sec),
                "top_tags": tags_in_sec.most_common(8),
            }

        stats = IGTStats(
            total_examples=self.n,
            unique_tags=len(tag_counts),
            tag_profiles=tag_profiles,
            constructions=constructions,
            absent_categories=absent_categories,
            paradigm_slots=paradigm_slots,
            section_richness=section_richness,
            summary_text="",   # filled below
        )
        stats.summary_text = self._make_summary(stats, tag_counts)
        return stats

    def _make_summary(self, stats: IGTStats, tag_counts: Counter) -> str:
        lines = [
            f"IGT CORPUS: {stats.total_examples} examples, {stats.unique_tags} unique tags",
            "",
            "TOP 30 TAGS (tag | count | % examples | position):",
        ]
        for tag, count in tag_counts.most_common(30):
            p = stats.tag_profiles[tag]
            lines.append(
                f"  {tag:<20} {count:>5}  {p.example_coverage*100:>5.1f}%  {p.position_label}"
            )

        lines += ["", "CONSTRUCTION PATTERNS (frequent ordered tag bigrams):"]
        for cp in stats.constructions[:20]:
            if len(cp.tags) == 2:
                ex_trans = cp.typical_translations[0][:60] if cp.typical_translations else ""
                lines.append(f"  {cp.tags[0]} → {cp.tags[1]}: {cp.count}x  e.g. '{ex_trans}'")

        lines += ["", "PARADIGM COVERAGE:"]
        for slot, info in stats.paradigm_slots.items():
            status = f"{info['total_hits']} hits" if info["attested"] else "NOT ATTESTED"
            forms  = ", ".join(f"{t}({c})" for t, c in info["present_forms"][:3])
            lines.append(f"  {slot:<6}: {status}  forms: {forms}")

        if stats.absent_categories:
            lines += ["", "LIKELY ABSENT CATEGORIES (near-zero IGT evidence):"]
            for ac in stats.absent_categories:
                lines.append(f"  {ac['category']}: {ac['note']}")

        lines += ["", "SECTION RICHNESS (top 15 by example count):"]
        sorted_secs = sorted(
            stats.section_richness.items(),
            key=lambda x: x[1]["n_examples"], reverse=True
        )
        for sec, info in sorted_secs[:15]:
            top = ", ".join(t for t, _ in info["top_tags"][:5])
            lines.append(f"  {sec:<45} {info['n_examples']:>4} examples  [{top}]")

        return "\n".join(lines)

    # ── Targeted query methods ────────────────────────────────────

    def query_tag(self, tag: str) -> str:
        """Detailed profile for a single tag — called by the agent during investigation."""
        stats = self.get_stats()
        tag_u = tag.upper()
        if tag_u not in stats.tag_profiles:
            # Try prefix match
            matches = [t for t in stats.tag_profiles if t.startswith(tag_u)]
            if not matches:
                return f"Tag '{tag}' not found in IGT corpus."
            results = [self._format_profile(stats.tag_profiles[m]) for m in matches[:5]]
            return "\n\n".join(results)
        return self._format_profile(stats.tag_profiles[tag_u])

    def get_morpheme_forms(self, tag: str) -> str:
        """
        Extract the actual surface morpheme forms that carry <tag> in the corpus.

        For each example containing the tag, aligns the morpheme and gloss lines
        and extracts the specific morpheme slot whose gloss matches the tag.

        Returns a frequency-ranked list of attested forms such as:
            Attested forms (38 instances): '-ma' (22x), '=ma' (6x), 'ma' (5x), '-m' (5x)

        This answers "what does the FUT/NEG/PST morpheme actually look like?" —
        essential for describing the language's surface morphology.
        """
        tag_u = tag.upper()
        from collections import Counter
        form_counts: Counter = Counter()

        for ex in self.examples:
            if tag_u not in ex.gloss_tags:
                continue
            if not ex.morpheme or not ex.gloss:
                continue

            morph_words = ex.morpheme.split()
            gloss_words = ex.gloss.split()

            for mw, gw in zip(morph_words, gloss_words):
                morphemes = re.split(r"[-=]", mw)
                glosses   = re.split(r"[-=]", gw)
                if len(morphemes) != len(glosses):
                    continue

                for m, g in zip(morphemes, glosses):
                    g_u = g.upper()
                    if (g_u == tag_u
                            or g_u.startswith(tag_u + ".")
                            or g_u.startswith(tag_u + "-")):
                        form_counts[m] += 1

        if not form_counts:
            return ""

        total = sum(form_counts.values())
        top   = form_counts.most_common(8)
        parts = [f"'{f}' ({n}x)" for f, n in top]
        return f"Attested forms ({total} instances): {', '.join(parts)}"

    def _format_profile(self, p: TagProfile) -> str:
        lines = [
            f"TAG: {p.tag}",
            f"  Frequency:   {p.count} ({p.example_coverage*100:.1f}% of {self.n} examples)",
            f"  Position:    mean={p.mean_position:.2f}, std={p.position_std:.2f} → {p.position_label}",
            f"  Top co-occurrents: {p.top_cooccurrents[:6]}",
            f"  Top sections: {p.top_sections[:4]}",
        ]
        # Append attested forms if available
        forms_str = self.get_morpheme_forms(p.tag)
        if forms_str:
            lines.append(f"  {forms_str}")
        return "\n".join(lines)

    def query_construction(self, tags: list[str]) -> str:
        """Find examples of a specific construction (ordered tag sequence)."""
        stats  = self.get_stats()
        tags_u = [t.upper() for t in tags]
        n      = len(tags_u)
        hits   = []
        for ex in self.examples:
            etags = ex.gloss_tags
            for i in range(len(etags) - n + 1):
                if etags[i:i+n] == tags_u:
                    hits.append(ex)
                    break
        if not hits:
            return f"No examples found with ordered sequence {tags_u}"
        lines = [f"Construction {tags_u}: {len(hits)} examples"]
        for ex in hits[:8]:
            morpheme_line = f"\n  Morphemes: {ex.morpheme}" if getattr(ex, 'morpheme', '') else ""
            lines.append(
                f"\n  [{ex.example_id}] {ex.source_section}\n"
                f"  Source:  {ex.source}{morpheme_line}\n"
                f"  Gloss:   {ex.gloss}\n"
                f"  Trans:   {ex.translation}"
            )
        return "\n".join(lines)

    def query_absent_evidence(self, category: str) -> str:
        """Report on the absence of a typological category."""
        stats    = self.get_stats()
        cat_u    = category.upper()
        cluster  = self.TYPOLOGICAL_CLUSTERS.get(cat_u, [])
        if not cluster:
            # Try fuzzy match on cluster names
            for k in self.TYPOLOGICAL_CLUSTERS:
                if cat_u in k or k in cat_u:
                    cluster = self.TYPOLOGICAL_CLUSTERS[k]
                    cat_u   = k
                    break

        if not cluster:
            # Unknown category: report which known clusters exist and
            # give a neutral "not checkable" verdict — NOT an absence verdict.
            # This prevents the LLM from misreading "unknown" as "absent".
            return (
                f"ABSENCE ANALYSIS: {category}\n"
                f"  Status: CATEGORY NOT IN TYPOLOGICAL CLUSTERS\n"
                f"  '{category}' is not a recognised typological cluster name.\n"
                f"  Known clusters: {list(self.TYPOLOGICAL_CLUSTERS.keys())}\n"
                f"  VERDICT: CANNOT ASSESS — absence cannot be confirmed or denied.\n"
                f"  Recommendation: use analyse_tag() or search_translations() instead."
            )

        hits_per_tag = {t: stats.tag_profiles[t].count
                        for t in cluster if t in stats.tag_profiles}
        total_hits   = sum(hits_per_tag.values())
        rate         = total_hits / self.n

        lines = [
            f"ABSENCE ANALYSIS: {cat_u}",
            f"  Canonical tags checked: {cluster}",
            f"  Tags found in corpus:   {hits_per_tag}",
            f"  Total hits:             {total_hits} / {self.n} examples ({rate*100:.2f}%)",
        ]
        if rate == 0:
            lines.append("  VERDICT: STRONG ABSENCE — zero evidence in IGT")
        elif rate < 0.01:
            lines.append("  VERDICT: WEAK PRESENCE — marginal, possibly loanwords or discourse particles")
        elif rate < 0.05:
            lines.append("  VERDICT: LIMITED — present but not grammaticalized as core category")
        else:
            lines.append(f"  VERDICT: PRESENT — {rate*100:.1f}% coverage suggests grammaticalized")
        return "\n".join(lines)

    def query_section_igt(self, section_query: str, max_examples: int = 10) -> str:
        """
        Return all IGT examples from sections matching the query,
        with quantitative summary. This gives the agent data grounded
        in where the author placed examples, not just tag matching.
        """
        q = section_query.lower()
        matching = [
            ex for ex in self.examples
            if q in (ex.source_section or "").lower()
            or q in (ex.section or "").lower()
            or q in (ex.chapter or "").lower()
            or q in (getattr(ex, 'subsubsection', '') or "").lower()
        ]
        if not matching:
            return f"No IGT examples found in sections matching '{section_query}'"

        tag_summary = Counter(t for ex in matching for t in ex.gloss_tags)
        lines = [
            f"IGT in sections matching '{section_query}': {len(matching)} examples",
            f"Tag distribution: {tag_summary.most_common(10)}",
            "",
        ]
        for ex in matching[:max_examples]:
            morpheme_line = f"\n    {ex.morpheme}" if getattr(ex, 'morpheme', '') else ""
            lines.append(
                f"  [{ex.example_id}] {ex.source_section}\n"
                f"    {ex.source}{morpheme_line}\n"
                f"    {ex.gloss}\n"
                f"    '{ex.translation}'"
            )
        if len(matching) > max_examples:
            lines.append(f"  ... and {len(matching) - max_examples} more")
        return "\n".join(lines)

    def compare_tags(self, tag_a: str, tag_b: str) -> str:
        """
        Compare two tags: do they co-occur? Are they complementary?
        This helps detect e.g. aspect vs tense distinctions,
        or whether two apparent markers are in free variation.
        """
        stats  = self.get_stats()
        a, b   = tag_a.upper(), tag_b.upper()
        pa     = stats.tag_profiles.get(a)
        pb     = stats.tag_profiles.get(b)
        if not pa or not pb:
            missing = a if not pa else b
            return f"Tag '{missing}' not in corpus"

        # Co-occurrence count
        pair     = tuple(sorted([a, b]))
        cooc_cnt = sum(
            1 for ex in self.examples
            if a in ex.gloss_tags and b in ex.gloss_tags
        )
        expected = pa.count * pb.count / self.n
        pmi      = math.log2(cooc_cnt / expected) if cooc_cnt > 0 else float("-inf")

        lines = [
            f"COMPARISON: {a} vs {b}",
            f"  {a}: {pa.count} examples ({pa.example_coverage*100:.1f}%), {pa.position_label}",
            f"  {b}: {pb.count} examples ({pb.example_coverage*100:.1f}%), {pb.position_label}",
            f"  Co-occurrence: {cooc_cnt} examples  (expected by chance: {expected:.1f}, PMI={pmi:.2f})",
        ]
        if pmi > 1.5:
            lines.append("  → Tags ATTRACT each other (likely part of same construction)")
        elif pmi < -1.0:
            lines.append("  → Tags REPEL each other (possibly in complementary distribution)")
        else:
            lines.append("  → Tags are roughly INDEPENDENT")

        # Positional comparison
        if abs(pa.mean_position - pb.mean_position) < 0.1:
            lines.append(f"  → Similar linear position ({pa.mean_position:.2f} vs {pb.mean_position:.2f})")
        else:
            lines.append(
                f"  → Different positions: {a}={pa.mean_position:.2f}, {b}={pb.mean_position:.2f}"
                f" — may be in different structural slots"
            )
        return "\n".join(lines)

    # ── Translation-based analysis ────────────────────────────────

    # Semantic keyword clusters for translation-line analysis.
    # Used by analyse_semantic_context() to identify what semantic
    # domain a tag's examples fall into.
    SEMANTIC_CLUSTERS = {
        "PAST":     ["yesterday", "ago", "before", "last", "previously", "already",
                     "used to", "had", "was", "were", "did", "came", "went"],
        "FUTURE":   ["tomorrow", "will", "going to", "soon", "later", "shall",
                     "would", "next"],
        "PRESENT":  ["now", "today", "currently", "still", "is", "are", "does"],
        "NEGATION": ["not", "no", "never", "nothing", "nobody", "nor", "without",
                     "don't", "doesn't", "didn't", "won't", "can't", "isn't"],
        "QUESTION": ["?", "who", "what", "where", "when", "why", "how", "which"],
        "CAUSATION":["because", "cause", "so that", "therefore", "make", "let",
                     "force", "allow"],
        "MODALITY": ["can", "could", "may", "might", "must", "should", "ought",
                     "able", "want", "wish", "need"],
        "REPORTED": ["said", "told", "asked", "thought", "heard", "they say",
                     "apparently", "reportedly"],
        "ASPECT_PFV": ["finished", "completed", "already", "done", "once",
                       "suddenly", "immediately"],
        "ASPECT_IPFV": ["while", "during", "kept", "was doing", "repeatedly",
                        "always", "usually", "habitually"],
    }

    def analyse_semantic_context(self, tag: str, max_examples: int = 8) -> str:
        """
        For all IGT examples containing <tag>, analyse the semantic content
        of their translation lines.

        Returns:
          - Keyword frequency in translations (which semantic domains are present)
          - Representative examples with full morpheme + gloss + translation
          - Comparison: does the translation content match the tag's assumed meaning?

        This cross-validates tag interpretations and can reveal polysemy or
        mismatch between the tag label and its actual semantic function.
        """
        tag_u = tag.upper()
        stats = self.get_stats()

        if tag_u not in stats.tag_profiles:
            matches = [t for t in stats.tag_profiles if t.startswith(tag_u)]
            if not matches:
                return f"Tag '{tag}' not found in corpus."
            tag_u = matches[0]

        # Collect examples with this tag
        tagged_examples = [
            ex for ex in self.examples if tag_u in ex.gloss_tags
        ]
        if not tagged_examples:
            return f"No examples found for tag '{tag_u}'."

        translations = [ex.translation.lower() for ex in tagged_examples]

        # Count semantic keyword hits
        cluster_hits = {}
        for cluster_name, keywords in self.SEMANTIC_CLUSTERS.items():
            hits = sum(
                1 for trans in translations
                if any(kw in trans for kw in keywords)
            )
            if hits > 0:
                cluster_hits[cluster_name] = hits

        # Word frequency across all translations
        from collections import Counter
        import re as _re
        words = _re.findall(r"\b[a-z]{3,}\b", " ".join(translations))
        # Filter out function words
        stopwords = {"the", "and", "that", "this", "with", "for", "from",
                     "have", "his", "her", "him", "they", "their", "its",
                     "was", "are", "but", "not", "has", "had", "into"}
        word_freq = Counter(w for w in words if w not in stopwords)

        lines = [
            f"SEMANTIC CONTEXT ANALYSIS: tag={tag_u}  ({len(tagged_examples)} examples)",
            "",
            "Semantic domain hits in translations:",
        ]
        if cluster_hits:
            for cluster, count in sorted(cluster_hits.items(), key=lambda x: -x[1]):
                pct = count / len(tagged_examples) * 100
                lines.append(f"  {cluster:<16} {count:>4}x  ({pct:.0f}% of {tag_u} examples)")
        else:
            lines.append("  (no matches in known semantic clusters)")

        lines += [
            "",
            f"Top translation words (excl. stopwords):",
            "  " + "  ".join(f"{w}({c})" for w, c in word_freq.most_common(15)),
            "",
            f"Representative examples (up to {max_examples}):",
        ]

        for ex in tagged_examples[:max_examples]:
            morpheme_line = f"\n    {ex.morpheme}" if ex.morpheme else ""
            lines.append(
                f"\n  [{ex.example_id}] {ex.source_section}"
                f"\n    {ex.source}{morpheme_line}"
                f"\n    {ex.gloss}"
                f"\n    '{ex.translation}'"
            )

        return "\n".join(lines)

    def search_translations(self, query: str, max_results: int = 15) -> str:
        """
        Search translation lines for a keyword or phrase.

        Useful for:
          - Finding all negation contexts even without consistent NEG tag
          - Finding temporal expressions ("yesterday", "will") to cross-validate tense tags
          - Discovering constructions by their semantic output rather than their tag
          - Identifying mismatches: translation says "not" but no NEG tag → annotation gap

        Returns matching IGT examples with morpheme, gloss, translation, and section.
        Also reports which tags appear in the matching examples, so you can see
        what grammatical machinery co-occurs with this semantic content.
        """
        q = query.lower().strip()
        matches = [
            ex for ex in self.examples
            if q in ex.translation.lower()
        ]

        if not matches:
            return f"No translation lines contain '{query}'."

        # Tag distribution in matching examples
        from collections import Counter
        tag_dist = Counter(t for ex in matches for t in ex.gloss_tags)

        lines = [
            f"TRANSLATION SEARCH: '{query}'  →  {len(matches)} matching examples",
            f"(out of {self.n} total, {len(matches)/self.n*100:.1f}%)",
            "",
            f"Tags in matching examples (what grammar co-occurs with '{query}'):",
            "  " + "  ".join(f"{t}({c})" for t, c in tag_dist.most_common(12)),
            "",
            f"Examples (showing up to {max_results}):",
        ]

        for ex in matches[:max_results]:
            morpheme_line = f"\n    {ex.morpheme}" if ex.morpheme else ""
            # Highlight the query in the translation
            trans_display = ex.translation.replace(
                query, f"[{query}]"
            ).replace(
                query.capitalize(), f"[{query.capitalize()}]"
            )
            lines.append(
                f"\n  [{ex.example_id}] {ex.source_section}"
                f"\n    {ex.source}{morpheme_line}"
                f"\n    {ex.gloss}"
                f"\n    '{trans_display}'"
                f"\n    tags: {ex.gloss_tags}"
            )

        if len(matches) > max_results:
            lines.append(f"\n  ... and {len(matches) - max_results} more examples")

        return "\n".join(lines)

    # ── Structural analysis tools ─────────────────────────────────

    def analyse_morpheme_position(self, tag: str) -> str:
        """
        For all IGT examples containing <tag>, determine where in the word
        the tagged morpheme appears: prefix (word-initial), suffix
        (word-final), stem (only morpheme in word), or medial/infix.

        Uses the hyphen/equals boundaries in the morpheme and gloss lines
        to locate the exact slot of the morpheme within its host word.
        This reveals whether a grammatical category is prefixal, suffixal,
        or an independent stem — a key fact for morphological typology.
        """
        tag_u = tag.upper()
        stats = self.get_stats()

        if tag_u not in stats.tag_profiles:
            candidates = [t for t in stats.tag_profiles if t.startswith(tag_u)]
            if not candidates:
                return f"Tag '{tag}' not found in IGT corpus."
            tag_u = candidates[0]

        counts       = {"prefix": 0, "suffix": 0, "stem": 0, "medial": 0}
        word_examples = []   # [(morpheme_word, gloss_word, translation_snippet)]

        for ex in self.examples:
            if tag_u not in ex.gloss_tags:
                continue
            if not ex.morpheme or not ex.gloss:
                continue

            morph_words = ex.morpheme.split()
            gloss_words = ex.gloss.split()

            for mw, gw in zip(morph_words, gloss_words):
                morphemes = re.split(r"[-=]", mw)
                glosses   = re.split(r"[-=]", gw)
                if len(morphemes) != len(glosses):
                    continue  # misaligned — skip

                for i, g in enumerate(glosses):
                    g_u = g.upper()
                    # Match exact tag or tag as prefix of a compound gloss (e.g. FUT.SG)
                    if not (g_u == tag_u
                            or g_u.startswith(tag_u + ".")
                            or g_u.startswith(tag_u + "-")):
                        continue

                    n = len(glosses)
                    if   n == 1:  counts["stem"]   += 1
                    elif i == 0:  counts["prefix"] += 1
                    elif i == n - 1: counts["suffix"] += 1
                    else:         counts["medial"]  += 1

                    if len(word_examples) < 6:
                        word_examples.append((mw, gw, ex.translation[:55]))

        total = sum(counts.values())
        if total == 0:
            return (
                f"Could not determine morpheme position for '{tag_u}': "
                f"no aligned morpheme-gloss data found in corpus "
                f"(morpheme line may be absent or use a different segmentation convention)."
            )

        dominant = max(counts, key=counts.get)
        lines = [
            f"MORPHEME POSITION ANALYSIS: {tag_u}",
            f"  Instances with morpheme data: {total}",
            f"  Prefix  (word-initial):  {counts['prefix']:>3}  "
            f"({100*counts['prefix']/total:>4.0f}%)",
            f"  Suffix  (word-final):    {counts['suffix']:>3}  "
            f"({100*counts['suffix']/total:>4.0f}%)",
            f"  Stem    (only morpheme): {counts['stem']:>3}  "
            f"({100*counts['stem']/total:>4.0f}%)",
        ]
        if counts["medial"]:
            lines.append(
                f"  Medial  (infix):         {counts['medial']:>3}  "
                f"({100*counts['medial']/total:>4.0f}%)"
            )
        lines.append(f"  → Dominant position: {dominant.upper()}")

        if word_examples:
            lines.append("\n  Word-level examples  (morpheme-word | gloss-word | translation):")
            for mw, gw, tr in word_examples:
                lines.append(f"    {mw:<25}  {gw:<25}  '{tr}'")

        return "\n".join(lines)

    def get_triline_examples(self, query: str, max_examples: int = 8) -> str:
        """
        Return properly formatted morpheme / gloss / translation trilines for
        IGT examples matching *query*.

        Matching order (first match wins):
          1. Exact uppercase tag match in gloss_tags
          2. Case-insensitive substring in translation line
          3. Case-insensitive substring in source_section name

        This is the primary tool for clause-structure analysis — it gives the
        agent the raw sentence data needed to reason about argument order,
        clause boundaries, and morpheme function beyond tag statistics.
        """
        query_u = query.upper()
        q_lower = query.lower()

        # 1. Tag match
        matches = [ex for ex in self.examples if query_u in ex.gloss_tags]
        source  = f"tag={query_u}"

        # 2. Translation keyword
        if not matches:
            matches = [ex for ex in self.examples
                       if q_lower in ex.translation.lower()]
            source  = f"translation='{query}'"

        # 3. Section name
        if not matches:
            matches = [ex for ex in self.examples
                       if q_lower in (ex.source_section or "").lower()]
            source  = f"section='{query}'"

        if not matches:
            return f"No examples found for '{query}'."

        n_shown = min(len(matches), max_examples)
        lines = [
            f"TRILINE EXAMPLES  ({source}): "
            f"{len(matches)} total, showing {n_shown}",
            "─" * 64,
        ]

        for ex in matches[:n_shown]:
            lines.append(f"[{ex.example_id}]  {ex.source_section}")
            if ex.morpheme:
                lines.append(f"  MORPHEME:    {ex.morpheme}")
            lines.append(f"  GLOSS:       {ex.gloss}")
            lines.append(f"  TRANSLATION: '{ex.translation}'")
            lines.append("")

        return "\n".join(lines)
