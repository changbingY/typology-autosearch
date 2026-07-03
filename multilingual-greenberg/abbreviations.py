"""
abbreviations.py — Glossing Abbreviation Registry
===================================================
Loads a tab-separated abbreviation file (TAG\\tFull meaning) and provides:

  expand(tag)           → "PST → past"  (or just "PST" if unknown)
  label(tag)            → "past"         (full form only, or tag if unknown)
  enrich_tag_list(tags) → ["PST (past)", "NEG (negation)", ...]
  enrich_igt_summary(text) → replaces every bare TAG in a summary string
                              with "TAG (full meaning)"
  annotated_table()     → markdown table of all known abbreviations

Usage
-----
  registry = AbbreviationRegistry("abbreviations.txt")
  registry.expand("PST")           # "PST (past)"
  registry.label("PST")            # "past"
  registry.enrich_tag_list(["PST", "NEG", "XYZ"])
  # → ["PST (past)", "NEG (negation)", "XYZ"]
"""

from __future__ import annotations
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AbbreviationRegistry:
    """
    Lightweight registry that maps gloss abbreviations to their full meanings.

    File format (tab-separated, one entry per line):
        PST\\tpast
        NEG\\tnegation
        1\\tfirst person
        ...
    Lines starting with '#' and blank lines are ignored.
    """

    def __init__(self, path: str | Path | None = None):
        self._map: dict[str, str] = {}   # upper-cased tag → full meaning
        if path:
            self.load(path)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> None:
        """Load (or reload) abbreviations from a file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"AbbreviationRegistry: file not found: {path}")
            return

        loaded = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Accept both tab and multiple-spaces as delimiter
                parts = re.split(r"\t| {2,}", line, maxsplit=1)
                if len(parts) == 2:
                    tag, meaning = parts[0].strip(), parts[1].strip()
                    self._map[tag.upper()] = meaning
                    loaded += 1
                else:
                    logger.debug(f"AbbreviationRegistry: skipping malformed line: {line!r}")

        logger.info(f"AbbreviationRegistry: loaded {loaded} abbreviations from {path}")

    # ── Core lookup ───────────────────────────────────────────────────────────

    def label(self, tag: str) -> str:
        """Return the full meaning of a tag, or the tag itself if unknown."""
        return self._map.get(tag.upper(), tag)

    def known(self, tag: str) -> bool:
        return tag.upper() in self._map

    def expand(self, tag: str) -> str:
        """
        Return 'TAG (full meaning)' if known, otherwise just 'TAG'.
        Examples:
            expand("PST")  → "PST (past)"
            expand("XYZ")  → "XYZ"
        """
        meaning = self._map.get(tag.upper())
        return f"{tag.upper()} ({meaning})" if meaning else tag.upper()

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def enrich_tag_list(self, tags: list[str]) -> list[str]:
        """
        Given a list of tag strings, return each expanded with its meaning.

        Example:
            ["PST", "NEG", "XYZ"] → ["PST (past)", "NEG (negation)", "XYZ"]
        """
        return [self.expand(t) for t in tags]

    def enrich_igt_summary(self, text: str) -> str:
        """
        Scan a free-form IGT statistics summary string and annotate every
        recognised bare tag (uppercase word token) with its meaning.

        Only tags that are KNOWN in the registry are annotated, so unknown
        language-specific tags are left as-is.

        Example input:
            "PST: 140 examples (8.4%), NEG: 87 (5.2%)"
        Example output:
            "PST (past): 140 examples (8.4%), NEG (negation): 87 (5.2%)"
        """
        if not self._map:
            return text

        # Match bare uppercase tokens (e.g. PST, NEG, CAUS.I, 1SG)
        # followed by a word-boundary — but NOT already annotated (no open-paren follows)
        def _replace(match: re.Match) -> str:
            tok = match.group(0)
            meaning = self._map.get(tok.upper())
            # Only annotate if the next char is not '(' (already expanded)
            return f"{tok} ({meaning})" if meaning else tok

        # Pattern: all-uppercase token optionally containing dots/digits (e.g. CAUS.I, 3SG)
        # Uses a negative lookahead to skip already-expanded "TAG (..." tokens
        pattern = re.compile(r'\b([A-Z][A-Z0-9]*(?:\.[A-Z][A-Z0-9]*)*)(?!\s*\()', re.ASCII)
        return pattern.sub(_replace, text)

    def find_tags_for_phenomenon(self, query: str, top_n: int = 10) -> list:
        """
        Reverse lookup: given a linguistic phenomenon description (free text),
        return the abbreviation tags whose meanings best match.

        Scoring (per tag):
          - 2 pts per query word that appears as a whole word in the meaning
          - 1 pt per query word that appears as a substring in the meaning
          - 1 pt per query word that appears in the tag itself

        Returns [(tag, meaning), ...] sorted by descending score, max top_n.

        Examples:
          find_tags_for_phenomenon("definite specific articles")
            → [("DEF", "definite article"), ("DEM", "demonstrative"), ...]
          find_tags_for_phenomenon("past tense")
            → [("PST", "past"), ("IMPF", "imperfective"), ...]
        """
        q_tokens = re.findall(r"[a-z]{3,}", query.lower())
        if not q_tokens or not self._map:
            return []

        q_words = set(q_tokens)
        results = []

        for tag, meaning in self._map.items():
            meaning_lower  = meaning.lower()
            tag_lower      = tag.lower()
            meaning_words  = set(re.findall(r"[a-z]{3,}", meaning_lower))

            # Whole-word overlap with meaning (strongest signal)
            word_score   = len(q_words & meaning_words) * 2
            # Substring match inside meaning (catches plurals, compounds)
            substr_score = sum(1 for w in q_words if w in meaning_lower and w not in meaning_words)
            # Tag-text match (e.g. query "definite" → tag "DEF")
            tag_score    = sum(1 for w in q_words if w in tag_lower)

            total = word_score + substr_score + tag_score
            if total > 0:
                results.append((total, tag, meaning))

        results.sort(key=lambda x: -x[0])
        return [(tag, meaning) for _, tag, meaning in results[:top_n]]

    def enrich_tag_stats(self, tag_freq_dict: dict[str, int]) -> dict[str, dict]:
        """
        Enrich a {TAG: count} dict into {TAG: {"count": n, "meaning": "..."}} dict.
        Useful for structured JSON output.
        """
        return {
            tag: {"count": count, "meaning": self.label(tag)}
            for tag, count in tag_freq_dict.items()
        }

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def prompt_legend(self, tags: list[str] | None = None) -> str:
        """
        Return a compact legend string suitable for injection into an LLM prompt.

        If `tags` is given, only include those tags.
        Otherwise, include all known abbreviations.

        Example output:
            GLOSS ABBREVIATION LEGEND:
            PST=past | NEG=negation | IMPF=imperfective | ...
        """
        if tags:
            entries = [
                f"{t.upper()}={self.label(t)}"
                for t in tags
                if self.known(t)
            ]
        else:
            entries = [f"{k}={v}" for k, v in sorted(self._map.items())]

        if not entries:
            return ""

        # Wrap into lines of ~100 chars
        lines = ["GLOSS ABBREVIATION LEGEND:"]
        line, length = [], 0
        for e in entries:
            if length + len(e) + 3 > 100 and line:
                lines.append(" | ".join(line))
                line, length = [], 0
            line.append(e)
            length += len(e) + 3
        if line:
            lines.append(" | ".join(line))
        return "\n".join(lines)

    def annotated_table(self) -> str:
        """Return a markdown table of all known abbreviations."""
        if not self._map:
            return "No abbreviations loaded."
        lines = ["| Tag | Meaning |", "|-----|---------|"]
        for tag, meaning in sorted(self._map.items()):
            lines.append(f"| {tag} | {meaning} |")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self) -> str:
        return f"AbbreviationRegistry({len(self._map)} entries)"
