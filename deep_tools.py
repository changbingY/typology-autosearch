"""
deep_tools.py — DeepGrammarToolkit
====================================
Replaces GrammarToolkit with a toolkit that supports genuine deep search:

  1. Full section reading (all chunks, not 700-char snippets)
  2. Quantitative IGT analytics (positional, distributional, absence)
  3. Construction pattern lookup (ordered tag sequences)
  4. Cross-section chain following (multi-hop navigation)
  5. Author claim extraction (prose statements vs. examples)
  6. Contradiction-aware IGT querying
  7. Richer search that includes cross-references in results
  8. get_chunk_summaries(): returns LLM-generated summaries for cited chunks,
     injected into the conclusion prompt for richer synthesis
  9. AbbreviationRegistry integration: all tag-facing tool outputs are
     annotated with human-readable gloss meanings when an abbreviations
     file is supplied.
"""

import re
import json
import math
import logging
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import numpy as np

from section_reader import SectionReader
from igt_analysis import IGTAnalyser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Re-use Chunk / IGTExample / BM25 / mmr from tools.py
# (import rather than redefine)
# ─────────────────────────────────────────────
from tools import Chunk, IGTExample, BM25, mmr_rerank
from section_reader import SectionResult
from collections import namedtuple
from abbreviations import AbbreviationRegistry          # ← NEW

# All IGT-touching tools return this instead of a plain string,
# so the agent can collect example IDs without parsing text.
ToolResult = namedtuple("ToolResult", ["text", "igt_ids"])

def _result(text: str, igt_ids: list = None) -> ToolResult:
    return ToolResult(text=text, igt_ids=igt_ids or [])


class DeepGrammarToolkit:

    def __init__(
        self,
        grammar_path: str,
        igt_path: Optional[str] = None,
        mmr_lambda: float = 0.6,
        abbreviations_path: Optional[str] = None,    # ← NEW
    ):
        self.grammar_path = Path(grammar_path)
        self.igt_path     = Path(igt_path) if igt_path else None
        self.mmr_lambda   = mmr_lambda

        # ── Abbreviation registry ──────────────────────────────────
        # Loaded before everything else so tag-enrichment is available
        # as soon as IGT data is loaded.
        self.abbrev = AbbreviationRegistry(abbreviations_path)  # ← NEW

        self.chunks:       list = []
        self.igt_examples: list = []
        self._embeddings        = None
        self._encoder           = None
        self._bm25              = None
        self._igt_tag_index: dict = defaultdict(list)
        self._igt_by_id:    dict = {}   # populated by _load_igt(); stays empty when no IGT

        logger.info(f"Loading grammar: {self.grammar_path}")
        self._load_grammar()

        if self.igt_path:
            logger.info(f"Loading IGT: {self.igt_path}")
            self._load_igt()

        logger.info("Building search indices...")
        self._build_index()

        # Deep search components
        logger.info("Building section reader...")
        self.section_reader = SectionReader(self.chunks)

        logger.info("Building IGT analyser...")
        self.igt_analyser = IGTAnalyser(self.igt_examples) if self.igt_examples else None

        # Chunk-ID → Chunk fast lookup (used by get_chunk_summaries)
        self._chunk_by_id: dict = {c.chunk_id: c for c in self.chunks}

        logger.info("DeepGrammarToolkit ready.")

    # ── Loading (same as tools.py, but loads summary field) ──────────────────

    def _load_grammar(self):
        suffix = self.grammar_path.suffix.lower()
        if suffix == ".json":
            with open(self.grammar_path, encoding="utf-8") as f:
                data = json.load(f)
            for i, entry in enumerate(data):
                chapter       = entry.get("chapter", "")
                section       = entry.get("section", "")
                subsection    = entry.get("subsection", "")
                subsubsection = entry.get("subsubsection", "")
                level = (4 if subsubsection else
                         3 if subsection else
                         2 if section else 1)
                self.chunks.append(Chunk(
                    chunk_id=entry.get("chunk_id", f"chunk_{i:04d}"),
                    text=entry.get("text", ""),
                    chapter=chapter,
                    section=section,
                    subsection=subsection,
                    subsubsection=subsubsection,
                    label=entry.get("label", ""),
                    page=entry.get("page"),
                    level=level,
                    summary=entry.get("summary", ""),   # ← load pre-generated summary
                ))
        else:
            text = self.grammar_path.read_text(encoding="utf-8")
            self.chunks = self._chunk_plain_text(text)
        logger.info(f"  {len(self.chunks)} chunks loaded")

    def _chunk_plain_text(self, text: str) -> list:
        # Same as tools.py
        header_re = re.compile(
            r"^(?:Chapter\s+\d+|Section\s+\d+|§[\d.]+|\d+\.(?:\d+\.?)*)\s*.+$",
            re.MULTILINE | re.IGNORECASE,
        )
        chunks = []
        cur = {"chapter": "", "section": "", "subsection": "", "level": 1}
        buf = []

        def flush():
            t = "\n".join(buf).strip()
            if len(t) > 50:
                chunks.append(Chunk(
                    chunk_id=f"chunk_{len(chunks):04d}",
                    text=t, chapter=cur["chapter"],
                    section=cur["section"], subsection=cur["subsection"],
                    label="", page=None, level=cur["level"],
                    summary="",
                ))
            buf.clear()

        for line in text.split("\n"):
            if header_re.match(line.strip()):
                flush()
                h = line.strip()
                if re.match(r"^Chapter\s+\d+", h, re.IGNORECASE):
                    cur.update(chapter=h, section="", subsection="", level=1)
                elif re.match(r"^§\d+\.\d+\.\d+", h):
                    cur.update(subsection=h, level=3)
                elif re.match(r"^§\d+\.\d+", h):
                    cur.update(section=h, subsection="", level=2)
                else:
                    cur.update(section=h, subsection="", level=2)
            buf.append(line)
            if len(buf) > 60:
                flush()
        flush()
        return chunks

    def _load_igt(self):
        with open(self.igt_path, encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            tags          = [t.upper() for t in entry.get("gloss_tags", [])]
            chapter       = entry.get("chapter", "")
            section       = entry.get("section", "")
            subsection    = entry.get("subsection", "")
            subsubsection = entry.get("subsubsection", "")
            breadcrumb    = " > ".join(filter(None, [chapter, section, subsection, subsubsection]))
            if not breadcrumb:
                breadcrumb = entry.get("source_section", "")
            ex = IGTExample(
                example_id=entry.get("example_id", f"igt_{i:04d}"),
                source=entry.get("source", ""),
                morpheme=entry.get("morpheme", ""),
                gloss=entry.get("gloss", ""),
                translation=entry.get("translation", ""),
                gloss_tags=tags,
                chapter=chapter,
                section=section,
                subsection=subsection,
                subsubsection=subsubsection,
                tones=entry.get("tones", ""),
                label=entry.get("label", ""),
                source_section=breadcrumb,
            )
            self.igt_examples.append(ex)
            for tag in set(tags):
                self._igt_tag_index[tag].append(i)
        # Build example_id → IGTExample index for fast lookup
        self._igt_by_id = {ex.example_id: ex for ex in self.igt_examples}
        logger.info(f"  {len(self.igt_examples)} IGT examples, "
                    f"{len(self._igt_tag_index)} unique tags indexed")

    def _build_index(self):
        texts = [
            f"{c.chapter} {c.section} {c.subsection} {c.text}"
            for c in self.chunks
        ]
        self._bm25 = BM25(texts)
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._embeddings = self._encoder.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            logger.info("  Dense embedding index ready")
        except ImportError:
            logger.warning("  sentence-transformers not available; BM25 only")

    # ── Core hybrid retrieval (same as tools.py) ──────────────────

    def _hybrid_search(self, query: str, top_k: int, alpha: float = 0.5,
                       use_mmr: bool = True) -> list:
        n    = len(self.chunks)
        pool = min(n, top_k * 5)
        bm25_results = self._bm25.search(query, top_k=pool)

        if self._encoder is not None and self._embeddings is not None:
            q_emb = self._encoder.encode([query], convert_to_numpy=True)[0]
            norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
            dense_scores = self._embeddings @ q_emb / norms
            top_dense  = np.argsort(dense_scores)[::-1][:pool]
            dense_dict = {int(i): float(dense_scores[i]) for i in top_dense}
            bm25_dict  = dict(bm25_results)
            max_bm25   = max(bm25_dict.values()) if bm25_dict else 1.0
            all_idx    = set(dense_dict) | set(bm25_dict)
            fused      = [
                (idx, alpha * dense_dict.get(idx, 0.0) +
                 (1 - alpha) * bm25_dict.get(idx, 0.0) / max_bm25)
                for idx in all_idx
            ]
            fused.sort(key=lambda x: x[1], reverse=True)
            candidates = fused[:pool]
            if use_mmr:
                candidates = mmr_rerank(
                    q_emb, candidates, self._embeddings,
                    top_k=top_k, lambda_=self.mmr_lambda
                )
            else:
                candidates = candidates[:top_k]
        else:
            max_s      = max((s for _, s in bm25_results), default=1.0)
            candidates = [(i, s / max_s) for i, s in bm25_results[:top_k]]

        return [(self.chunks[i], score) for i, score in candidates]

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC TOOL API
    # ═══════════════════════════════════════════════════════════════

    def get_toc(self) -> str:
        """TOC from section_reader — one entry per section, with char counts."""
        return self.section_reader.format_toc()

    def get_toc_with_summaries(
        self,
        max_entries: int = 300,
        max_summary_chars: int = 120,
        skip_subsections: bool = False,
    ) -> str:
        """
        Enriched TOC: one entry per unique section, showing the full 4-level
        chapter / section / subsection / subsubsection hierarchy and, when
        available, the LLM-generated summary of that section's linguistic content.
        """
        seen: dict = {}   # section_key → first Chunk in that section
        order = []
        for c in self.chunks:
            if skip_subsections and c.subsection:
                key = f"{c.chapter}||{c.section}||||"
            else:
                key = f"{c.chapter}||{c.section}||{c.subsection}||{c.subsubsection}"
            if key not in seen:
                seen[key] = c
                order.append(key)

        lines = []
        for key in order[:max_entries]:
            c = seen[key]
            if skip_subsections:
                header = " > ".join(filter(None, [c.chapter, c.section]))
                level  = min(c.level, 2)
            else:
                header = " > ".join(
                    filter(None, [c.chapter, c.section, c.subsection, c.subsubsection])
                )
                level  = c.level
            indent = "  " * max(0, level - 1)
            label  = f"  [{c.label}]" if c.label else ""
            lines.append(f"{indent}{header}{label}  (id:{c.chunk_id})")

            if c.summary:
                summary_text = c.summary[:max_summary_chars].replace("\n", " ").strip()
                if len(c.summary) > max_summary_chars:
                    summary_text += "..."
                lines.append(f"{indent}  Summary: {summary_text}")

        if len(order) > max_entries:
            lines.append(f"[... {len(order) - max_entries} more sections not shown]")

        return "\n".join(lines)

    # ── 0. Chunk summaries ────────────────────────────────────────

    def get_chunk_summaries(self, chunk_ids: list, max_summaries: int = 12) -> str:
        """
        Return LLM-generated summaries for the given chunk IDs.
        """
        if not chunk_ids:
            return "No chunk summaries available (no chunks cited)."

        seen = set()
        unique_ids = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        lines = ["CHUNK SUMMARIES (LLM-generated synopsis of cited grammar sections):"]
        found = 0
        for cid in unique_ids[:max_summaries]:
            chunk = self._chunk_by_id.get(cid)
            if chunk is None:
                base_id = re.sub(r"_p\d+$", "", cid)
                chunk = self._chunk_by_id.get(base_id)
            if chunk is None:
                continue

            summary = chunk.summary.strip()
            if not summary:
                continue

            section_path = " > ".join(
                filter(None, [chunk.chapter, chunk.section, chunk.subsection])
            )
            lines.append(f"\n[{cid}] {section_path}")
            lines.append(f"  {summary}")
            found += 1

        if found == 0:
            return (
                "No chunk summaries found. "
                "Re-run preprocessing with --summarize to generate them."
            )

        if len(unique_ids) > max_summaries:
            lines.append(
                f"\n[... {len(unique_ids) - max_summaries} additional cited chunks not shown]"
            )

        return "\n".join(lines)

    # ── 1. Full section reading ───────────────────────────────────

    def read_full_section(self, query: str) -> SectionResult:
        """
        Read the COMPLETE text of a section (all chunks merged).
        """
        sr = self.section_reader.get_full_section(query, max_chars=8000)
        if "[NOT FOUND]" in sr.text:
            fallback = query + "_p0"
            sr2 = self.section_reader.get_full_section(fallback, max_chars=8000)
            if "[NOT FOUND]" not in sr2.text:
                return sr2
        return sr

    def follow_cross_references(self, query: str) -> SectionResult:
        """
        Read a section AND follow all its cross-references to other sections.
        """
        chain = self.section_reader.find_cross_references(query, max_hops=2)
        header = (
            f"Cross-reference chain for '{query}': "
            f"{len(chain.primary_sections)} primary + {len(chain.linked_sections)} linked sections, "
            f"{chain.total_text_chars} total chars"
        )
        all_chunk_ids = []
        all_paths = []
        for sr in list(chain.primary_sections) + list(chain.linked_sections):
            if hasattr(sr, 'chunk_ids'):
                all_chunk_ids.extend(sr.chunk_ids)
            if hasattr(sr, 'section_path'):
                all_paths.append(sr.section_path)
        return SectionResult(
            text=header + "\n\n" + chain.formatted_text,
            chunk_ids=list(dict.fromkeys(all_chunk_ids)),
            section_path=" + ".join(all_paths[:3]),
        )

    def extract_author_claims(self, query: str) -> str:
        """
        Extract only the author's explicit analytical statements.
        """
        claims = self.section_reader.extract_claims(query)
        if not claims:
            return f"No explicit author claims found for '{query}'"
        lines = [f"AUTHOR CLAIMS related to '{query}' ({len(claims)} found):"]
        for i, c in enumerate(claims, 1):
            lines.append(f"  {i}. {c}")
        return "\n".join(lines)

    # ── 2. Quantitative IGT analysis ─────────────────────────────

    def get_igt_summary(self) -> str:
        """Full quantitative IGT overview — tag frequencies, positions,
        constructions, absent categories, paradigm coverage.
        Tag abbreviations are annotated with their full meanings when
        an AbbreviationRegistry is loaded."""
        if not self.igt_analyser:
            return "No IGT data loaded."
        raw = self.igt_analyser.get_stats().summary_text
        # ← NEW: annotate every recognised bare tag with its gloss meaning
        return self.abbrev.enrich_igt_summary(raw)

    def _section_aware_ids(
        self,
        candidates: list,
        preferred_sections: list,
        max_ids: int = 8,
    ) -> list:
        """
        Given a list of IGTExample objects, return up to max_ids example IDs
        ordered so that examples from preferred_sections come first.
        """
        if not preferred_sections:
            return [ex.example_id for ex in candidates[:max_ids]]

        prefs_lower = [p.lower() for p in preferred_sections]

        preferred = []
        other     = []
        for ex in candidates:
            breadcrumb = (ex.source_section or "").lower()
            if any(p in breadcrumb for p in prefs_lower):
                preferred.append(ex.example_id)
            else:
                other.append(ex.example_id)

        pref_quota = min(len(preferred), max(1, max_ids // 2 + 1))
        result     = preferred[:pref_quota]
        remaining  = max_ids - len(result)
        result    += other[:remaining]
        return result

    def analyse_tag(self, tag: str, preferred_sections: list = None) -> "ToolResult":
        """
        Deep profile for a single gloss tag: frequency, linear position,
        top co-occurrents, section distribution.
        The tag header is annotated with its full meaning when the
        AbbreviationRegistry knows the tag.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        tag_label = self.abbrev.expand(tag)                    # ← NEW  e.g. "PST (past)"
        raw_text  = self.igt_analyser.query_tag(tag)
        # Prepend expanded label + enrich the body text
        text = f"[{tag_label}]\n" + self.abbrev.enrich_igt_summary(raw_text)  # ← NEW

        tag_u      = tag.upper()
        candidates = [ex for ex in self.igt_examples if tag_u in ex.gloss_tags]
        ids = self._section_aware_ids(candidates, preferred_sections or [], max_ids=8)
        return _result(text, ids)

    def analyse_construction(self, tags: list, preferred_sections: list = None) -> "ToolResult":
        """
        Find all IGT examples containing a specific ordered sequence of gloss tags.
        The construction header is annotated with full tag meanings.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        tags_u   = [t.upper() for t in tags]
        n        = len(tags_u)
        # ← NEW: human-readable label for the whole construction
        expanded = " + ".join(self.abbrev.expand(t) for t in tags_u)

        hits = []
        for ex in self.igt_examples:
            etags = ex.gloss_tags
            for i in range(len(etags) - n + 1):
                if etags[i:i+n] == tags_u:
                    hits.append(ex)
                    break

        if not hits:
            text = f"No examples found with ordered sequence {tags_u}  [{expanded}]"  # ← NEW
        else:
            lines = [f"Construction {tags_u}  [{expanded}]: {len(hits)} examples"]   # ← NEW
            for ex in hits[:8]:
                morpheme_line = f"\n  Morphemes: {ex.morpheme}" if getattr(ex, 'morpheme', '') else ""
                lines.append(
                    f"\n  [{ex.example_id}] {ex.source_section}\n"
                    f"  Source:  {ex.source}{morpheme_line}\n"
                    f"  Gloss:   {ex.gloss}\n"
                    f"  Trans:   {ex.translation}"
                )
            text = "\n".join(lines)

        ids = self._section_aware_ids(hits, preferred_sections or [], max_ids=8)
        return _result(text, ids)

    def analyse_absence(self, category: str, preferred_sections: list = None) -> "ToolResult":
        """
        Quantify the absence of a typological category.
        The absence header is annotated with the category's full meaning.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        cat_label = self.abbrev.expand(category)               # ← NEW
        raw_text  = self.igt_analyser.query_absent_evidence(category)
        text      = f"[Absence check: {cat_label}]\n" + self.abbrev.enrich_igt_summary(raw_text)  # ← NEW

        cat_u   = category.upper()
        cluster = self.igt_analyser.TYPOLOGICAL_CLUSTERS.get(cat_u, [])
        if not cluster:
            for k in self.igt_analyser.TYPOLOGICAL_CLUSTERS:
                if cat_u in k or k in cat_u:
                    cluster = self.igt_analyser.TYPOLOGICAL_CLUSTERS[k]
                    break
        candidates = [
            ex for ex in self.igt_examples
            if any(t in ex.gloss_tags for t in cluster)
        ]
        ids = self._section_aware_ids(candidates, preferred_sections or [], max_ids=5)
        return _result(text, ids)

    def compare_tags(self, tag_a: str, tag_b: str, preferred_sections: list = None) -> "ToolResult":
        """
        Compare two tags statistically: co-occurrence vs complementary distribution.
        Both tags are annotated with their full meanings in the header.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        label_a = self.abbrev.expand(tag_a)                    # ← NEW
        label_b = self.abbrev.expand(tag_b)                    # ← NEW
        raw_text = self.igt_analyser.compare_tags(tag_a, tag_b)
        text = (                                                # ← NEW
            f"[Comparing: {label_a}  vs  {label_b}]\n"
            + self.abbrev.enrich_igt_summary(raw_text)
        )

        a, b = tag_a.upper(), tag_b.upper()
        candidates = [
            ex for ex in self.igt_examples
            if a in ex.gloss_tags and b in ex.gloss_tags
        ]
        ids = self._section_aware_ids(candidates, preferred_sections or [], max_ids=5)
        return _result(text, ids)

    def get_section_igt(self, section_query: str, max_examples: int = 15) -> "ToolResult":
        """
        Retrieve ALL IGT examples from sections matching the query.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        text = self.igt_analyser.query_section_igt(section_query, max_examples)
        # Enrich tag references in the output text
        text = self.abbrev.enrich_igt_summary(text)            # ← NEW
        q = section_query.lower()
        ids = [
            ex.example_id for ex in self.igt_examples
            if q in (ex.source_section or "").lower()
            or q in (ex.section or "").lower()
            or q in (ex.chapter or "").lower()
            or q in (ex.subsubsection or "").lower()
        ][:10]
        return _result(text, ids)

    # ── 3. Hybrid text search ─────────────────────────────────────

    def search_text(self, query: str, top_k: int = 5, confidence: float = 1.0) -> str:
        """
        Hybrid BM25+dense search with MMR. Returns snippet-level results.
        """
        effective_k = top_k * 2 if confidence < 0.5 else top_k
        results = self._hybrid_search(query, top_k=effective_k)
        if not results:
            return "[NO RESULTS]"
        parts = []
        for chunk, score in results:
            loc = " > ".join(filter(None, [chunk.chapter, chunk.section, chunk.subsection]))
            parts.append(
                f"[{chunk.chunk_id}] score={score:.3f} | {loc}\n"
                f"{chunk.text[:500]}"
            )
        return "\n\n---\n\n".join(parts)

    # ── 4. Targeted IGT retrieval ─────────────────────────────────

    def get_igt_candidates(self, query_tags: list, query_text: str = "",
                           max_examples: int = 10) -> "ToolResult":
        """
        Retrieve IGT examples by tag match + keyword.
        """
        tags_upper = [t.upper() for t in query_tags]
        keywords   = [w.lower() for w in query_text.split() if len(w) > 3]
        scored     = []
        for i, ex in enumerate(self.igt_examples):
            tag_hits  = sum(1 for t in tags_upper if t in ex.gloss_tags)
            text_hits = sum(
                1 for kw in keywords
                if kw in ex.translation.lower() or kw in ex.gloss.lower()
            )
            score = tag_hits * 2 + text_hits
            if score > 0:
                scored.append((score, i))
        if not scored:
            return _result(f"No IGT examples found for tags={tags_upper}")

        scored.sort(reverse=True)
        shown = scored[:max_examples]
        # ← NEW: show expanded tag labels in the header
        expanded_tags = self.abbrev.enrich_tag_list(tags_upper)
        lines = [f"IGT candidates: {len(scored)} total, showing top {len(shown)} (tags={expanded_tags}):"]
        shown_ids = []
        for score, idx in shown:
            ex = self.igt_examples[idx]
            shown_ids.append(ex.example_id)
            morpheme_line = f"\n  Morphemes:   {ex.morpheme}" if ex.morpheme else ""
            lines.append(
                f"\n[{ex.example_id}] match_score={score} | {ex.source_section}\n"
                f"  Source:      {ex.source}{morpheme_line}\n"
                f"  Gloss:       {ex.gloss}\n"
                f"  Translation: {ex.translation}\n"
                f"  Tags:        {ex.gloss_tags}"
            )
        return _result("\n".join(lines), shown_ids)

    def lookup_igt_examples(self, example_ids: list, notes: dict = None) -> list:
        """
        Given a list of real IGT example IDs, return a list of dicts with
        source, morpheme, gloss, translation, location, and (if provided) note.
        """
        results = []
        for eid in example_ids:
            ex = self._igt_by_id.get(eid)
            if ex is None:
                continue
            entry = {
                "example_id":  ex.example_id,
                "source":      ex.source,
                "gloss":       ex.gloss,
                "translation": ex.translation,
                "section":     ex.source_section,
            }
            if ex.morpheme:
                entry["morpheme"] = ex.morpheme
            if notes and eid in notes:
                entry["note"]       = notes[eid]["note"]
                entry["cited_from"] = notes[eid]["source"]
            results.append(entry)
        return results

    def get_context(self, chunk_id: str, window: int = 2) -> str:
        """Surrounding chunks for context."""
        for i, c in enumerate(self.chunks):
            if c.chunk_id == chunk_id:
                start = max(0, i - window)
                end   = min(len(self.chunks), i + window + 1)
                parts = []
                for j in range(start, end):
                    marker = " <<<TARGET>>>" if j == i else ""
                    loc = " > ".join(filter(None, [
                        self.chunks[j].chapter,
                        self.chunks[j].section,
                        self.chunks[j].subsection,
                    ]))
                    parts.append(
                        f"[{self.chunks[j].chunk_id}]{marker} {loc}\n"
                        f"{self.chunks[j].text[:600]}"
                    )
                return "\n\n---\n\n".join(parts)
        return f"[NOT FOUND] chunk_id='{chunk_id}'"

    # ═══════════════════════════════════════════════════════════════
    # IGT DISCOVERY TOOLS  (used by IGTOnlyResearchAgent)
    # ═══════════════════════════════════════════════════════════════

    def get_tag_inventory(self, top_n: int = 60) -> "ToolResult":
        """
        Return a sorted inventory of all tags with raw counts and corpus
        coverage (%).  Grouped by typological cluster.
        Tag meanings are appended in parentheses when known.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        stats = self.igt_analyser.get_stats()
        total = self.igt_analyser.n

        tag_to_cluster: dict = {}
        for cluster_name, tags in self.igt_analyser.TYPOLOGICAL_CLUSTERS.items():
            for t in tags:
                tag_to_cluster[t] = cluster_name

        sorted_profiles = sorted(
            stats.tag_profiles.values(),
            key=lambda p: p.count,
            reverse=True,
        )[:top_n]

        clustered: dict = {}
        for p in sorted_profiles:
            cluster = tag_to_cluster.get(p.tag, "OTHER")
            clustered.setdefault(cluster, []).append(p)

        lines = [
            f"TAG INVENTORY  ({total} total IGT examples, showing top {top_n} tags)",
            f"{'TAG':<28} {'COUNT':>6}  {'COV%':>6}  {'POSITION':<22}  CLUSTER",
            "-" * 80,
        ]

        order = list(self.igt_analyser.TYPOLOGICAL_CLUSTERS.keys()) + ["OTHER"]
        for cluster in order:
            profiles = clustered.get(cluster, [])
            if not profiles:
                continue
            lines.append(f"\n[{cluster}]")
            for p in profiles:
                # ← NEW: show "PST (past)" instead of bare "PST"
                tag_display = self.abbrev.expand(p.tag)
                lines.append(
                    f"  {tag_display:<26} {p.count:>6}  {p.example_coverage*100:>5.1f}%"
                    f"  {p.position_label:<22}"
                )

        return _result("\n".join(lines))

    def get_construction_inventory(self, top_n: int = 30) -> "ToolResult":
        """
        Return the top ordered bigrams and trigrams with counts.
        Tag labels in construction sequences are expanded with meanings.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        stats = self.igt_analyser.get_stats()
        lines = [f"CONSTRUCTION INVENTORY  (top ordered tag sequences)\n"]

        bigrams  = [cp for cp in stats.constructions if len(cp.tags) == 2][:top_n]
        trigrams = [cp for cp in stats.constructions if len(cp.tags) == 3][:15]

        lines.append(f"TOP BIGRAMS ({len(bigrams)}):")
        for cp in bigrams:
            trans = cp.typical_translations[0][:60] if cp.typical_translations else ""
            secs  = ", ".join(list(cp.sections)[:2])
            # ← NEW: show meanings inline
            a = self.abbrev.expand(cp.tags[0])
            b = self.abbrev.expand(cp.tags[1])
            lines.append(
                f"  {a:<22} → {b:<22}  {cp.count:>5}x"
                f"  e.g. '{trans}'  [{secs}]"
            )

        if trigrams:
            lines.append(f"\nTOP TRIGRAMS ({len(trigrams)}):")
            for cp in trigrams:
                trans = cp.typical_translations[0][:55] if cp.typical_translations else ""
                seq   = " → ".join(self.abbrev.expand(t) for t in cp.tags)  # ← NEW
                lines.append(
                    f"  {seq:<50}  {cp.count:>5}x"
                    f"  e.g. '{trans}'"
                )

        return _result("\n".join(lines))

    def find_tag_cluster(self, seed_tag: str, top_n: int = 12) -> "ToolResult":
        """
        Discover which tags co-occur most strongly with seed_tag (PMI ranking).
        Both seed and co-tags are annotated with their full meanings.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")

        import math
        seed = seed_tag.upper()
        stats = self.igt_analyser.get_stats()

        if seed not in stats.tag_profiles:
            matches = [t for t in stats.tag_profiles if t.startswith(seed)]
            if not matches:
                return _result(f"Tag '{seed_tag}' not found in corpus.")
            seed = matches[0]

        seed_profile = stats.tag_profiles[seed]
        seed_count   = seed_profile.count
        total        = self.igt_analyser.n

        pmi_scores = []
        for (other_tag, cooc_count) in seed_profile.top_cooccurrents:
            other_profile = stats.tag_profiles.get(other_tag)
            if not other_profile:
                continue
            expected = seed_count * other_profile.count / total
            if expected < 1:
                continue
            pmi = math.log2(cooc_count / expected)
            pmi_scores.append((other_tag, cooc_count, pmi, other_profile.position_label))

        pmi_scores.sort(key=lambda x: x[2], reverse=True)

        # ← NEW: annotate seed tag
        seed_label = self.abbrev.expand(seed)
        lines = [
            f"TAG CLUSTER FOR: {seed_label}",
            f"  Seed frequency: {seed_count} ({seed_count/total*100:.1f}%), "
            f"position: {seed_profile.position_label}",
            f"\n  {'CO-TAG':<26} {'CO-OCC':>7}  {'PMI':>6}  POSITION  INTERPRETATION",
            "  " + "-" * 70,
        ]

        for other_tag, cooc_count, pmi, pos_label in pmi_scores[:top_n]:
            if pmi > 1.5:
                interp = "ATTRACTS (same construction)"
            elif pmi < -1.0:
                interp = "REPELS (complementary distribution)"
            else:
                interp = "independent"
            # ← NEW: annotate co-tag
            other_label = self.abbrev.expand(other_tag)
            lines.append(
                f"  {other_label:<26} {cooc_count:>7}  {pmi:>+6.2f}  "
                f"{pos_label:<18}  {interp}"
            )

        return _result("\n".join(lines))

    def analyse_semantic_context(self, tag: str) -> "ToolResult":
        """
        For all examples with <tag>, analyse translation-line semantics.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        text = self.igt_analyser.analyse_semantic_context(tag)
        # ← NEW: annotate tag in header
        text = f"[Semantic context: {self.abbrev.expand(tag)}]\n" + text
        tag_u = tag.upper()
        ids = [
            ex.example_id for ex in self.igt_examples
            if tag_u in ex.gloss_tags
        ][:10]
        return _result(text, ids)

    def search_translations(self, query: str, max_results: int = 15) -> "ToolResult":
        """
        Keyword search across all translation lines.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        text = self.igt_analyser.search_translations(query, max_results)
        # Enrich any tag references that appear in the result body
        text = self.abbrev.enrich_igt_summary(text)            # ← NEW
        q = query.lower()
        ids = [
            ex.example_id for ex in self.igt_examples
            if q in ex.translation.lower()
        ][:max_results]
        return _result(text, ids)

    # ── Structural analysis tools ─────────────────────────────────

    def analyse_morpheme_position(self, tag: str) -> "ToolResult":
        """
        Determine whether a morpheme is a prefix, suffix, or stem
        by analysing its position within words in the morpheme line.
        Tag header is annotated with full meaning.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        tag_label = self.abbrev.expand(tag)
        raw  = self.igt_analyser.analyse_morpheme_position(tag)
        text = f"[Morpheme position: {tag_label}]\n" + raw
        tag_u = tag.upper()
        ids = [
            ex.example_id for ex in self.igt_examples
            if tag_u in ex.gloss_tags and ex.morpheme
        ][:8]
        return _result(text, ids)

    def get_triline_examples(self, query: str, max_examples: int = 8) -> "ToolResult":
        """
        Return aligned morpheme / gloss / translation trilines for
        examples matching query (tag, translation keyword, or section name).
        Essential for clause-structure and argument-structure analysis.
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        text = self.igt_analyser.get_triline_examples(query, max_examples)
        query_u = query.upper()
        q_lower = query.lower()
        ids = [
            ex.example_id for ex in self.igt_examples
            if query_u in ex.gloss_tags
            or q_lower in ex.translation.lower()
            or q_lower in (ex.source_section or "").lower()
        ][:max_examples]
        return _result(text, ids)

    def get_morpheme_forms(self, tag: str) -> "ToolResult":
        """
        Show the actual surface morpheme forms that carry <tag> in the corpus,
        ranked by frequency.  Answers "what does the FUT/NEG/PST morpheme
        look like in this language?" — e.g. "'-ma' (22x), '=ma' (6x), 'ma' (5x)".
        """
        if not self.igt_analyser:
            return _result("No IGT data loaded.")
        tag_label = self.abbrev.expand(tag)
        forms_str = self.igt_analyser.get_morpheme_forms(tag)
        if not forms_str:
            return _result(
                f"[{tag_label}] No morpheme-level data found for '{tag}' "
                f"(morpheme line may be absent in this corpus)."
            )
        text = f"[Surface forms: {tag_label}]\n  {forms_str}"
        tag_u = tag.upper()
        ids = [
            ex.example_id for ex in self.igt_examples
            if tag_u in ex.gloss_tags and ex.morpheme
        ][:8]
        return _result(text, ids)
