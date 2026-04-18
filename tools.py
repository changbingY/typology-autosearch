"""
GrammarToolkit — Upgraded Retrieval Layer
==========================================
Upgrades over v1:
  - Hybrid BM25 + embedding search with score fusion
  - MMR (Maximal Marginal Relevance) for result diversity
  - Adaptive top_k: doubles when confidence is low
  - search_text appends relevant IGT examples inline
  - get_igt_candidates: structured IGT retrieval scored by tag matches
  - Chunk.summary field: LLM-generated per-chunk summary (loaded from JSON)
"""

import re
import json
import math
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    chapter: str
    section: str
    subsection: str
    subsubsection: str = ""   # 4th hierarchy level (new field)
    label: str = ""           # optional LaTeX label; absent in most grammars
    page: Optional[int] = None
    level: int = 2            # computed from hierarchy depth when loading
    summary: str = ""         # LLM-generated summary; empty if preprocessing ran without --summarize


@dataclass
class IGTExample:
    example_id: str
    source: str               # original-language utterance (object language line)
    morpheme: str             # morpheme segmentation line (e.g. "buy-APPL-FUT.SG")
    gloss: str
    translation: str
    gloss_tags: list
    chapter: str
    section: str
    subsection: str
    subsubsection: str = ""   # 4th hierarchy level
    tones: str = ""           # tone tier, if present (kept for backward compat)
    label: str = ""           # section label, e.g. "sec:4.2" (absent in many grammars)
    source_section: str = ""  # human-readable breadcrumb: up to 4 levels


# ─────────────────────────────────────────────
# BM25 (no extra dependencies)
# ─────────────────────────────────────────────

class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.tokenized = [self._tok(d) for d in docs]
        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.n, 1)
        df = Counter()
        for tokens in self.tokenized:
            for tok in set(tokens):
                df[tok] += 1
        self.idf = {
            tok: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def _tok(self, text: str) -> list:
        # Include digits: linguistic abbreviations like "1sg", "3pl", "v2" must not be split
        return re.findall(r"[a-z0-9]+", text.lower())

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tok(query)
        doc_tokens = self.tokenized[doc_idx]
        dl = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0
        for tok in tokens:
            if tok not in self.idf:
                continue
            tf = tf_map.get(tok, 0)
            score += self.idf[tok] * (
                tf * (self.k1 + 1) /
                (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            )
        return score

    def search(self, query: str, top_k: int) -> list:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores[:top_k] if s > 0]


# ─────────────────────────────────────────────
# MMR reranking
# ─────────────────────────────────────────────

def mmr_rerank(
    query_emb: np.ndarray,
    candidates: list,           # [(chunk_idx, fused_score)]
    embeddings: np.ndarray,
    top_k: int,
    lambda_: float = 0.6,       # 1=pure relevance, 0=pure diversity
) -> list:
    if not candidates:
        return []
    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            sel_embs = embeddings[[i for i, _ in selected]]

            def mmr_score(item):
                idx, rel = item
                emb = embeddings[idx]
                norms = np.linalg.norm(sel_embs, axis=1) * np.linalg.norm(emb) + 1e-9
                max_sim = float(np.max(sel_embs @ emb / norms))
                return lambda_ * rel - (1 - lambda_) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


# ─────────────────────────────────────────────
# GrammarToolkit
# ─────────────────────────────────────────────

class GrammarToolkit:

    def __init__(
        self,
        grammar_path: str,
        igt_path: Optional[str] = None,
        mmr_lambda: float = 0.6,
    ):
        self.grammar_path = Path(grammar_path)
        self.igt_path = Path(igt_path) if igt_path else None
        self.mmr_lambda = mmr_lambda

        self.chunks: list = []
        self.igt_examples: list = []
        self._embeddings = None
        self._encoder = None
        self._bm25 = None
        self._igt_tag_index: dict = defaultdict(list)

        logger.info(f"Loading grammar: {self.grammar_path}")
        self._load_grammar()

        if self.igt_path:
            logger.info(f"Loading IGT: {self.igt_path}")
            self._load_igt()

        logger.info("Building search indices...")
        self._build_index()

    # ── Loading ──────────────────────────────────────────────

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
                # Compute hierarchy depth so TOC indentation is correct
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
                    summary=entry.get("summary", ""),
                ))
        else:
            text = self.grammar_path.read_text(encoding="utf-8")
            self.chunks = self._chunk_plain_text(text)
        logger.info(f"  {len(self.chunks)} chunks loaded")

    def _chunk_plain_text(self, text: str) -> list:
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
                    text=t,
                    chapter=cur["chapter"],
                    section=cur["section"],
                    subsection=cur["subsection"],
                    label="",
                    page=None,
                    level=cur["level"],
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
        logger.info(f"  {len(self.igt_examples)} IGT examples, "
                    f"{len(self._igt_tag_index)} unique tags indexed")

    # ── Index building ────────────────────────────────────────

    def _build_index(self):
        texts = [
            f"{c.chapter} {c.section} {c.subsection} {c.text}"
            for c in self.chunks
        ]
        self._bm25 = BM25(texts)
        logger.info("  BM25 index ready")

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

    # ── Core hybrid retrieval ─────────────────────────────────

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        alpha: float = 0.5,
        use_mmr: bool = True,
    ) -> list:
        """
        Fuse BM25 + dense cosine scores, then MMR-rerank for diversity.
        alpha: weight on dense score (0=BM25 only, 1=dense only).
        """
        n = len(self.chunks)
        pool = min(n, top_k * 5)
        bm25_results = self._bm25.search(query, top_k=pool)

        if self._encoder is not None and self._embeddings is not None:
            q_emb = self._encoder.encode([query], convert_to_numpy=True)[0]
            norms = (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
            )
            dense_scores = self._embeddings @ q_emb / norms

            top_dense = np.argsort(dense_scores)[::-1][:pool]
            dense_dict = {int(i): float(dense_scores[i]) for i in top_dense}
            bm25_dict = dict(bm25_results)
            max_bm25 = max(bm25_dict.values()) if bm25_dict else 1.0

            all_idx = set(dense_dict) | set(bm25_dict)
            fused = [
                (idx,
                 alpha * dense_dict.get(idx, 0.0) +
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
            # BM25 only
            max_s = max((s for _, s in bm25_results), default=1.0)
            candidates = [(i, s / max_s) for i, s in bm25_results[:top_k]]

        return [(self.chunks[i], score) for i, score in candidates]

    # ── Public tools ──────────────────────────────────────────

    def get_toc(self) -> str:
        """Structured TOC with chunk IDs for direct section access.

        One entry per unique (chapter, section, subsection, subsubsection) 4-tuple.
        The chunk_id shown is the first sub-chunk of that section, so the
        agent can pass it directly to get_section() or get_context().
        Long sections split into chunk_NNNN_p0/p1/… are collapsed to one row.
        """
        seen_section: dict = {}   # section_key → first chunk seen
        order = []                # insertion-ordered section keys
        for c in self.chunks:
            key = f"{c.chapter}|{c.section}|{c.subsection}|{c.subsubsection}"
            if key not in seen_section:
                seen_section[key] = c   # store the whole chunk for metadata
                order.append(key)

        lines = []
        for key in order[:300]:
            c = seen_section[key]
            indent = "  " * max(0, c.level - 1)
            header = " > ".join(filter(None, [c.chapter, c.section, c.subsection, c.subsubsection]))
            label  = f"  [{c.label}]" if c.label else ""
            lines.append(f"{indent}[{c.chunk_id}] {header}{label}")
        return "\n".join(lines)

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        confidence: float = 1.0,
        include_igt: bool = True,
    ) -> str:
        """
        Hybrid BM25+dense search with MMR diversity.
        confidence < 0.5 → doubles top_k automatically.
        Appends matching IGT examples at the end.
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
                f"{chunk.text[:700]}"
            )
        output = "\n\n---\n\n".join(parts)

        if include_igt and self.igt_examples:
            igt_hits = self._igt_keyword_search(query, max_examples=5)
            if igt_hits:
                output += f"\n\n=== SUPPORTING IGT EXAMPLES ===\n{igt_hits}"

        return output

    def get_section(self, section_id: str) -> str:
        """
        Retrieve section by chunk_id or partial name.
        Reads TOC→chapter→section in order, so definitional content isn't missed.
        """
        # Direct chunk_id lookup
        for c in self.chunks:
            if c.chunk_id == section_id:
                loc = " > ".join(filter(None, [c.chapter, c.section, c.subsection]))
                return f"[{c.chunk_id}] {loc}  label={c.label}\n\n{c.text}"

        # Partial match against chapter/section/subsection/label
        q = section_id.lower()
        matches = [
            c for c in self.chunks
            if (q in c.chapter.lower() or q in c.section.lower()
                or q in c.subsection.lower() or q in c.label.lower())
        ]
        if not matches:
            return f"[NOT FOUND] No section matching '{section_id}'"

        parts = []
        for c in matches[:4]:
            loc = " > ".join(filter(None, [c.chapter, c.section, c.subsection]))
            parts.append(f"[{c.chunk_id}] {loc}  label={c.label}\n\n{c.text[:1200]}")
        return "\n\n---\n\n".join(parts)

    def get_igt_candidates(
        self,
        query_tags: list,
        query_text: str = "",
        max_examples: int = 10,
    ) -> str:
        """
        Retrieve IGT examples that support a feature.
        Scored by: number of matching gloss tags × 2 + text keyword match.
        Returns structured output with match scores, tones tier when present,
        and full section breadcrumb.
        """
        tags_upper = [t.upper() for t in query_tags]
        keywords = [w.lower() for w in query_text.split() if len(w) > 3]

        scored = []
        for i, ex in enumerate(self.igt_examples):
            tag_hits = sum(1 for t in tags_upper if t in ex.gloss_tags)
            text_hits = sum(
                1 for kw in keywords
                if kw in ex.translation.lower() or kw in ex.gloss.lower()
            )
            score = tag_hits * 2 + text_hits
            if score > 0:
                scored.append((score, i))

        if not scored:
            return f"No IGT examples found for tags={tags_upper}"

        scored.sort(reverse=True)
        total = len(scored)
        shown = scored[:max_examples]

        lines = [
            f"IGT candidates: {total} total, showing top {len(shown)} "
            f"(tags searched: {tags_upper}):"
        ]
        for score, idx in shown:
            ex = self.igt_examples[idx]
            morpheme_line = f"\n  Morphemes:   {ex.morpheme}" if ex.morpheme else ""
            lines.append(
                f"\n[{ex.example_id}] match_score={score} | {ex.source_section}\n"
                f"  Source:      {ex.source}"
                f"{morpheme_line}\n"
                f"  Gloss:       {ex.gloss}\n"
                f"  Translation: {ex.translation}\n"
                f"  Tags:        {ex.gloss_tags}"
            )
        return "\n".join(lines)

    def get_context(self, chunk_id: str, window: int = 2) -> str:
        """Retrieve surrounding chunks for context around a retrieved passage."""
        for i, c in enumerate(self.chunks):
            if c.chunk_id == chunk_id:
                start = max(0, i - window)
                end = min(len(self.chunks), i + window + 1)
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

    def get_gloss_statistics(self) -> str:
        """Tag frequency overview + section distribution — use during planning."""
        if not self.igt_examples:
            return "No IGT data loaded."
        all_tags = [t for ex in self.igt_examples for t in ex.gloss_tags]
        c = Counter(all_tags)
        lines = [
            f"Total examples: {len(self.igt_examples)}",
            f"Unique tags:    {len(c)}", "",
            "Top 50 gloss tags:",
        ]
        for tag, count in c.most_common(50):
            lines.append(f"  {tag:<22} {count:>5}")

        # Section distribution: useful for planning which sections have rich IGT
        section_counts = Counter(
            ex.source_section for ex in self.igt_examples if ex.source_section
        )
        lines += ["", "Top 20 sections by IGT example count:"]
        for sec, cnt in section_counts.most_common(20):
            lines.append(f"  {sec:<50} {cnt:>4}")

        tones_count = sum(1 for ex in self.igt_examples if ex.tones)
        if tones_count:
            lines += ["", f"Examples with tone tier: {tones_count}"]

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────

    def _igt_keyword_search(self, query: str, max_examples: int = 5) -> str:
        keywords = [w.upper() for w in re.findall(r"[a-zA-Z]{3,}", query)]
        scored = []
        for i, ex in enumerate(self.igt_examples):
            gloss_up = ex.gloss.upper()
            trans_up = ex.translation.upper()
            hits = sum(
                1 for kw in keywords
                if kw in gloss_up or kw in trans_up or kw in ex.gloss_tags
            )
            if hits:
                scored.append((hits, i))
        if not scored:
            return ""
        scored.sort(reverse=True)
        lines = []
        for _, idx in scored[:max_examples]:
            ex = self.igt_examples[idx]
            morpheme_line = f"\n  {ex.morpheme}" if ex.morpheme else ""
            lines.append(
                f"[{ex.example_id}] {ex.source_section}\n"
                f"  {ex.source}"
                f"{morpheme_line}\n"
                f"  {ex.gloss}\n"
                f"  '{ex.translation}'"
            )
        return "\n\n".join(lines)
