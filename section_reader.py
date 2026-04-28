"""
section_reader.py — Full Section Reading and Cross-Section Linking
===================================================================
The original tools.py returns 700-char snippets per chunk.
This module reads FULL sections (all chunks combined), identifies
cross-references between sections, and supports multi-hop navigation.

Key capabilities:
  - get_full_section(): returns complete text of a section (all chunks)
  - find_cross_references(): finds sections that discuss the same concept
  - get_section_chain(): follows cross-references to collect all evidence
    about a phenomenon across the grammar
  - extract_claims(): pulls out author's explicit analytical statements
    (distinct from examples or glosses)
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from collections import namedtuple

# Returned by get_full_section() so callers get chunk IDs alongside text
SectionResult = namedtuple("SectionResult", ["text", "chunk_ids", "section_path"])

logger = logging.getLogger(__name__)


@dataclass
class SectionNode:
    """A fully assembled grammar section (all chunks merged)."""
    section_id: str             # first chunk_id in this section
    chapter: str
    section: str
    subsection: str
    subsubsection: str          # 4th hierarchy level (empty string if absent)
    label: str
    level: int
    full_text: str              # all chunks concatenated
    char_count: int
    chunk_ids: list             # all chunk_ids that make this section
    # discovered during indexing:
    outgoing_refs: list = field(default_factory=list)   # labels/section names referenced
    key_terms: list    = field(default_factory=list)    # prominent linguistic terms


@dataclass
class CrossRefChain:
    """Result of following cross-references for a query."""
    query: str
    primary_sections: list      # SectionNodes directly matching query
    linked_sections: list       # SectionNodes reached via cross-reference
    total_text_chars: int
    formatted_text: str         # ready for LLM consumption


class SectionReader:

    # Patterns for cross-references inside grammar text
    XREF_PATTERNS = [
        re.compile(r"(?:see|cf\.?|→|discussed in|see also|as in)\s+(?:§|section|chapter)?\s*([\d.]+)", re.IGNORECASE),
        re.compile(r"§\s*([\d.]+)"),
        re.compile(r"\\label\{(sec:[^}]+)\}"),
        re.compile(r"\\ref\{(sec:[^}]+)\}"),
    ]

    # Patterns for author analytical statements (claims vs examples)
    CLAIM_PATTERNS = [
        re.compile(r"[A-Z][^.!?]*(?:is|are|has|have|does|do not|lacks?|shows?|exhibits?|marks?|expresses?|encodes?|grammaticalizes?)[^.!?]*[.!?]"),
        re.compile(r"[A-Z][^.!?]*(?:obligator|optionall|productive|systematic|restricted|absent|attested|unattested|never occurs?|always occurs?)[^.!?]*[.!?]"),
    ]

    def __init__(self, chunks: list):
        """
        chunks: list of Chunk dataclass instances from tools.py
        """
        self.chunks = chunks
        self._sections: dict   = {}     # section_key → SectionNode
        self._label_index: dict = {}    # label → section_key
        self._term_index: dict  = defaultdict(list)  # term → [section_key]
        self._build()

    def _build(self):
        # Group chunks into sections using the full 4-level hierarchy
        section_groups = defaultdict(list)
        for c in self.chunks:
            key = f"{c.chapter}||{c.section}||{c.subsection}||{getattr(c, 'subsubsection', '')}"
            section_groups[key].append(c)

        for key, section_chunks in section_groups.items():
            # Sort by chunk_id to maintain document order
            section_chunks.sort(key=lambda c: c.chunk_id)
            first = section_chunks[0]
            full_text = "\n\n".join(c.text for c in section_chunks)
            subsubsection = getattr(first, 'subsubsection', '')

            # Compute level from hierarchy depth (not stored in data)
            level = (4 if subsubsection else
                     3 if first.subsection else
                     2 if first.section else 1)

            node = SectionNode(
                section_id=first.chunk_id,
                chapter=first.chapter,
                section=first.section,
                subsection=first.subsection,
                subsubsection=subsubsection,
                label=getattr(first, 'label', ''),
                level=level,
                full_text=full_text,
                char_count=len(full_text),
                chunk_ids=[c.chunk_id for c in section_chunks],
                outgoing_refs=self._extract_refs(full_text),
                key_terms=self._extract_terms(full_text),
            )
            self._sections[key] = node

            if node.label:
                self._label_index[node.label] = key

            for term in node.key_terms:
                self._term_index[term.lower()].append(key)

        logger.info(
            f"SectionReader: {len(self._sections)} sections assembled, "
            f"avg {sum(n.char_count for n in self._sections.values())//max(1,len(self._sections))} chars/section"
        )

    def _extract_refs(self, text: str) -> list:
        refs = []
        for pat in self.XREF_PATTERNS:
            refs.extend(pat.findall(text))
        return list(set(refs))

    def _extract_terms(self, text: str) -> list:
        """
        Extract prominent linguistic terms — capitalized multi-word phrases
        and technical abbreviations in running text.
        """
        # Capture things like "TMA marker", "serial verb construction", "preverbal particle"
        phrase_re = re.compile(
            r"\b(?:[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3}|[A-Z]{2,}(?:\s+[A-Za-z]+)?)\b"
        )
        terms = []
        for m in phrase_re.finditer(text):
            t = m.group().strip()
            # Filter noise
            if len(t) >= 3 and not t.isnumeric():
                terms.append(t)
        # Deduplicate, keep most common
        c = {}
        for t in terms:
            c[t] = c.get(t, 0) + 1
        return [t for t, n in sorted(c.items(), key=lambda x: -x[1]) if n >= 2][:20]

    # ── Public API ────────────────────────────────────────────────

    def get_full_section(self, query: str, max_chars: int = 8000) -> "SectionResult":
        """
        Return the FULL text of a section matching query as a SectionResult(text, chunk_ids, section_path).
        Query can be: chunk_id, section name, label, or keyword.
        If the section is very long, returns the first max_chars chars but notes truncation.
        """
        node = self._find_section(query)
        if node is None:
            results = self._text_search_sections(query)
            if not results:
                return SectionResult(
                    text=f"[NOT FOUND] No section matching '{query}'",
                    chunk_ids=[],
                    section_path="",
                )
            node = results[0]

        section_path = " > ".join(filter(None, [node.chapter, node.section, node.subsection, node.subsubsection]))
        header = f"[{node.section_id}] {section_path}"
        header += f"  label={node.label}  ({node.char_count} chars, {len(node.chunk_ids)} chunks)"
        header += f"  chunks=[{', '.join(str(c) for c in node.chunk_ids[:5])}{'...' if len(node.chunk_ids)>5 else ''}]"

        text = node.full_text
        truncated = ""
        if len(text) > max_chars:
            truncated = f"\n\n[... truncated: {node.char_count - max_chars} chars remaining ...]"
            text = text[:max_chars]

        return SectionResult(
            text=f"{header}\n\n{text}{truncated}",
            chunk_ids=node.chunk_ids,
            section_path=section_path,
        )

    def get_full_section_node(self, query: str) -> Optional[SectionNode]:
        """Return the SectionNode itself, not formatted text."""
        node = self._find_section(query)
        if node is None:
            results = self._text_search_sections(query)
            return results[0] if results else None
        return node

    def find_cross_references(self, query: str, max_hops: int = 2) -> CrossRefChain:
        """
        Find all sections that discuss the same concept via cross-references.
        Starts from sections matching the query, then follows outgoing_refs
        up to max_hops deep.
        """
        primary = self._text_search_sections(query)[:3]
        visited = {n.section_id for n in primary}
        linked  = []

        frontier = list(primary)
        for _hop in range(max_hops):
            next_frontier = []
            for node in frontier:
                for ref in node.outgoing_refs:
                    # Resolve ref to a section
                    target = self._resolve_ref(ref)
                    if target and target.section_id not in visited:
                        visited.add(target.section_id)
                        linked.append(target)
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break

        total_chars = sum(n.char_count for n in primary + linked)

        # Format for LLM — primary sections in full, linked in summary
        parts = []
        for node in primary:
            parts.append(self._format_section(node, max_chars=3000, label="PRIMARY"))
        for node in linked[:4]:
            parts.append(self._format_section(node, max_chars=1500, label="CROSS-REF"))

        return CrossRefChain(
            query=query,
            primary_sections=primary,
            linked_sections=linked,
            total_text_chars=total_chars,
            formatted_text="\n\n" + ("─" * 60) + "\n\n".join(parts),
        )

    def extract_claims(self, query: str) -> list[str]:
        """
        Extract the author's explicit analytical claims from sections
        matching the query. Filters out example sentences, glosses,
        and bibliographic text. Returns just declarative statements
        about the language.
        """
        results = self._text_search_sections(query)
        claims  = []
        for node in results[:3]:
            for pat in self.CLAIM_PATTERNS:
                for m in pat.finditer(node.full_text):
                    claim = m.group().strip()
                    # Filter out obvious non-claims
                    if (len(claim) > 30
                            and not claim.startswith("\\")
                            and "%" not in claim
                            and claim not in claims):
                        claims.append(f"[{node.section} / {node.subsection}] {claim}")
        return claims[:25]

    def get_related_sections(self, query: str, top_k: int = 5) -> list:
        """Return top-k section nodes whose key_terms overlap with the query."""
        q_terms = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        scored  = []
        for node in self._sections.values():
            node_terms = {t.lower() for t in node.key_terms}
            name_terms = set(re.findall(
                r"[a-zA-Z]{4,}",
                f"{node.chapter} {node.section} {node.subsection} {node.subsubsection}".lower()
            ))
            overlap = len(q_terms & (node_terms | name_terms))
            if overlap > 0:
                scored.append((overlap, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:top_k]]

    # ── Internal helpers ──────────────────────────────────────────

    def _find_section(self, query: str) -> Optional[SectionNode]:
        # 1. Exact chunk_id match
        for node in self._sections.values():
            if query in node.chunk_ids:
                return node
        # 2. Label match
        if query in self._label_index:
            return self._sections.get(self._label_index[query])
        # 3. Section_id match
        for node in self._sections.values():
            if node.section_id == query:
                return node
        return None

    def _text_search_sections(self, query: str) -> list[SectionNode]:
        """
        Score sections by query term overlap.

        Two-tier scoring:
          Tier 1 (weight=1): token matches in chapter/section/subsection/subsubsection
                             names and key_terms extracted from section content.
          Tier 2 (weight=0.4): token matches inside the section's full_text.
                             Lower weight avoids promoting tangentially-mentioning
                             sections above genuinely focused ones, while ensuring
                             sections with generic titles are not completely invisible.
        A +5 boost is added for exact substring match in any of the section name fields.
        """
        q = query.lower()
        q_tokens = set(re.findall(r"[a-zA-Z]{3,}", q))
        scored = []
        for node in self._sections.values():
            header_combined = (
                f"{node.chapter} {node.section} {node.subsection} {node.subsubsection} "
                + " ".join(node.key_terms)
            ).lower()
            # Tier 1: name + key_terms match
            score = sum(1 for tok in q_tokens if tok in header_combined)
            # Tier 2: full-text match (lower weight to avoid demoting focused sections)
            body_lower = node.full_text.lower()
            score += 0.4 * sum(1 for tok in q_tokens if tok in body_lower)
            # Boost exact substring matches in any section name field
            if (q in node.section.lower() or q in node.subsection.lower()
                    or q in node.subsubsection.lower()):
                score += 5
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:8]]

    def _resolve_ref(self, ref: str) -> Optional[SectionNode]:
        """Resolve a cross-reference string to a SectionNode."""
        # Try label index
        if ref in self._label_index:
            return self._sections.get(self._label_index[ref])
        # Try section number prefix match
        for key, node in self._sections.items():
            if ref in node.label:
                return node
        return None

    def _format_section(self, node: SectionNode, max_chars: int, label: str) -> str:
        path_parts = filter(None, [node.chapter, node.section, node.subsection, node.subsubsection])
        header = (
            f"[{label}] [{node.section_id}] "
            + " > ".join(path_parts)
            + (f"  label={node.label}" if node.label else "")
        )
        text = node.full_text[:max_chars]
        if node.char_count > max_chars:
            text += f"\n[... +{node.char_count - max_chars} chars truncated]"
        return f"{header}\n\n{text}"

    def format_toc(self) -> str:
        """Return a clean, deduplicated TOC with section IDs."""
        lines = []
        seen  = set()
        for node in sorted(self._sections.values(), key=lambda n: n.section_id):
            key = f"{node.chapter}|{node.section}|{node.subsection}|{node.subsubsection}"
            if key in seen:
                continue
            seen.add(key)
            indent = "  " * max(0, node.level - 1)
            header = " > ".join(filter(None, [node.chapter, node.section, node.subsection, node.subsubsection]))
            label  = f"  [{node.label}]" if node.label else ""
            chars  = f"  ({node.char_count}c)"
            lines.append(f"{indent}[{node.section_id}] {header}{label}{chars}")
        return "\n".join(lines[:300])
