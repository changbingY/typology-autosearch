"""
evaluate.py — Grambank Coding Evaluator
=========================================
Standalone script — no dependency on llm.py or any other project file.
Calls the OpenAI API directly to code agent JSON outputs against
Grambank coding guidelines.

Usage
-----
# Single language folder:
python evaluate.py \
    --json  "results/Aguaruna/igt_query_*.json" \
    --guidelines grambank_queries.csv \
    --output eval_results.csv \
    --api-key sk-...

# All languages at once:
python evaluate.py \
    --json  "results/*/*_igt_queries.json" \
    --guidelines grambank_queries.csv \
    --output eval_results.csv \
    --api-key sk-...

# API key from environment variable (no --api-key needed):
export OPENAI_API_KEY=sk-...
python evaluate.py \
    --json  "results/*/*_igt_queries.json" \
    --guidelines grambank_queries.csv \
    --output eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model config — change here if needed ──────────────────────────────────────
DEFAULT_MODEL       = "gpt-5.4-mini-2026-03-17"
DEFAULT_TEMPERATURE = 0.0     # deterministic for coding tasks
DEFAULT_MAX_TOKENS  = 8       # only need a single character output
DEFAULT_RETRIES     = 3

VALID_CODES = {"0", "1", "2", "3", "?"}

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a linguistic typologist applying the Grambank coding scheme to fieldwork data.

## TASK
Read the AGENT ANSWER and apply the CODING INSTRUCTIONS to decide the correct code.
Output ONLY the single code character: 0, 1, 2, 3, or ?
No explanation. No punctuation. No extra text. Just one character.

## FEATURE QUESTION
{query}

## CODING INSTRUCTIONS
{coding}

## FEATURE DEFINITION
{definition}

## AGENT ANSWER
{answer}

## YOUR CODE (output exactly one character — 0, 1, 2, 3, or ?):"""


# ── OpenAI caller ─────────────────────────────────────────────────────────────

def call_openai(
    client,
    prompt:      str,
    model:       str,
    temperature: float,
    retries:     int,
) -> str:
    """
    Call the OpenAI chat completions API and return a single valid code character.
    Retries up to `retries` times if the output is not a valid code.
    Returns '?' on total failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=DEFAULT_MAX_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a linguistic typologist. "
                            "Output ONLY a single character: 0, 1, 2, 3, or ?. "
                            "Nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()

            # Extract the first valid code character from the response
            for ch in raw:
                if ch in VALID_CODES:
                    return ch

            logger.warning(f"  Attempt {attempt}: unexpected output {raw!r}, retrying...")

        except Exception as exc:
            logger.warning(f"  Attempt {attempt}: API error — {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # exponential back-off

    logger.warning("  All retries failed — defaulting to '?'")
    return "?"


# ── Guidelines loader ──────────────────────────────────────────────────────────

def load_guidelines(csv_path: Path) -> dict:
    """
    Load Grambank coding guidelines CSV.

    Returns dict with two lookup tables:
        by_id    : {"GB305": {id, query, coding, definition}, ...}
        by_query : {normalised_query_string: {...}, ...}
    """
    by_id:    dict[str, dict] = {}
    by_query: dict[str, dict] = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "id":         row.get("ID", "").strip(),
                "query":      row.get("Query", "").strip(),
                "coding":     row.get("Coding", "").strip(),
                "definition": row.get("Definition", "").strip(),
            }
            by_id[rec["id"]] = rec
            by_query[_norm(rec["query"])] = rec

    logger.info(f"Loaded {len(by_id)} coding guidelines from {csv_path.name}")
    return {"by_id": by_id, "by_query": by_query}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_guideline(query: str, query_id: str, guidelines: dict) -> dict | None:
    """Match a query result to a coding guideline (ID > exact text > token overlap)."""
    by_id    = guidelines["by_id"]
    by_query = guidelines["by_query"]

    if query_id and query_id in by_id:
        return by_id[query_id]

    nq = _norm(query)
    if nq in by_query:
        return by_query[nq]

    # Best token-overlap fallback
    q_tokens = set(nq.split())
    best, best_score = None, 0
    for key, rec in by_query.items():
        score = len(q_tokens & set(key.split()))
        if score > best_score:
            best, best_score = rec, score

    return best if best_score >= 3 else None


# ── JSON loader ────────────────────────────────────────────────────────────────

def load_query_results(json_path: Path) -> list[dict]:
    """
    Load one or more QueryResult dicts from a file.
    Handles both single-query files and batch wrapper files.
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data

    if "queries" in data and isinstance(data["queries"], list):
        lang = data.get("language", json_path.parent.name)
        results = []
        for q in data["queries"]:
            q.setdefault("language", lang)
            results.append(q)
        return results

    data.setdefault("language", json_path.parent.name)
    return [data]


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(
    json_paths:  list[Path],
    guidelines:  dict,
    client,
    model:       str,
    temperature: float,
    retries:     int,
    output_path: Path,
    show_answer: bool,
) -> None:

    fieldnames = [
        "language", "query_id", "gb_id", "query",
        "code", "confidence", "needs_human_review", "audit_verdict",
        "agent_llm_calls",
    ]
    if show_answer:
        fieldnames.append("agent_answer")

    rows: list[dict] = []
    n_matched = n_unmatched = 0

    for json_path in json_paths:
        logger.info(f"Processing: {json_path.name}")
        try:
            query_results = load_query_results(json_path)
        except Exception as exc:
            logger.error(f"  Could not load {json_path}: {exc}")
            continue

        for qr in query_results:
            query    = qr.get("query", "")
            query_id = qr.get("query_id", "")
            language = qr.get("language", json_path.parent.name)
            answer   = qr.get("answer", "")

            guideline = find_guideline(query, query_id, guidelines)

            if guideline is None:
                logger.warning(f"  No guideline matched: {query[:60]}")
                n_unmatched += 1
                rows.append({
                    "language":           language,
                    "query_id":           query_id,
                    "gb_id":              "",
                    "query":              query,
                    "code":               "UNMATCHED",
                    "confidence":         qr.get("confidence", ""),
                    "needs_human_review": qr.get("needs_human_review", ""),
                    "audit_verdict":      qr.get("audit_verdict", ""),
                    "agent_llm_calls":    qr.get("token_usage", {}).get("llm_calls", ""),
                    **({"agent_answer": answer} if show_answer else {}),
                })
                continue

            n_matched += 1
            logger.info(f"  [{language}] {guideline['id']}  {query[:55]}...")

            prompt = PROMPT_TEMPLATE.format(
                query      = guideline["query"],
                coding     = guideline["coding"],
                definition = guideline["definition"],
                answer     = answer,
            )

            t0   = time.time()
            code = call_openai(client, prompt, model, temperature, retries)
            logger.info(f"    → code={code}  ({time.time()-t0:.1f}s)")

            rows.append({
                "language":           language,
                "query_id":           query_id,
                "gb_id":              guideline["id"],
                "query":              query,
                "code":               code,
                "confidence":         qr.get("confidence", ""),
                "needs_human_review": qr.get("needs_human_review", ""),
                "audit_verdict":      qr.get("audit_verdict", ""),
                "agent_llm_calls":    qr.get("token_usage", {}).get("llm_calls", ""),
                **({"agent_answer": answer} if show_answer else {}),
            })

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"\nDone.  {len(rows)} rows → {output_path}\n"
        f"  Matched   : {n_matched}\n"
        f"  Unmatched : {n_unmatched}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Typology Autosearch outputs using Grambank coding guidelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        nargs="+",
        metavar="PATH",
        required=True,
        help="Agent output JSON files (glob patterns supported).",
    )
    parser.add_argument(
        "--guidelines",
        metavar="PATH",
        required=True,
        help="Grambank coding guidelines CSV.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="eval_results.csv",
        help="Output CSV. Default: eval_results.csv",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=DEFAULT_MODEL,
        help=f"OpenAI model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per query on invalid output. Default: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Include agent's answer text in the output CSV.",
    )

    args = parser.parse_args()

    # ── Resolve JSON file paths ────────────────────────────────────────────────
    json_paths: list[Path] = []
    for pattern in args.json:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        json_paths.extend(matches)

    json_paths = sorted(set(json_paths))
    if not json_paths:
        logger.error("No JSON files found. Check your --json paths.")
        sys.exit(1)
    logger.info(f"Found {len(json_paths)} JSON file(s).")

    # ── Load guidelines ────────────────────────────────────────────────────────
    guidelines_path = Path(args.guidelines)
    if not guidelines_path.exists():
        logger.error(f"Guidelines CSV not found: {guidelines_path}")
        sys.exit(1)
    guidelines = load_guidelines(guidelines_path)

    # ── Init OpenAI client ─────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error(
            "No API key found. Pass --api-key sk-... or set OPENAI_API_KEY."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    logger.info(f"OpenAI client ready — model: {args.model}")

    # ── Run ────────────────────────────────────────────────────────────────────
    evaluate(
        json_paths  = json_paths,
        guidelines  = guidelines,
        client      = client,
        model       = args.model,
        temperature = args.temperature,
        retries     = args.retries,
        output_path = Path(args.output),
        show_answer = args.show_answer,
    )


if __name__ == "__main__":
    main()