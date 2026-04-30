"""
evaluate.py — Grambank Coding Evaluator
=========================================
Reads one or more agent output JSON files, matches each query to the
Grambank coding guidelines CSV, calls an LLM to produce a coding value
(0 / 1 / 2 / 3 / ?), and writes results to a summary CSV.

The LLM sees ONLY:
  - The agent's `answer` field
  - The coding instructions from the CSV (step-by-step rules)
  - The feature definition from the CSV
  - A strict instruction to output ONLY the coding value

Usage
-----
# Single file:
python evaluate.py \\
    --json  results/Aguaruna/igt_query_151_*.json \\
    --guidelines grambank_queries.csv \\
    --output eval_results.csv

# Whole language folder (all igt_query_*.json files):
python evaluate.py \\
    --json  results/Aguaruna/igt_query_*.json \\
    --guidelines grambank_queries.csv \\
    --output eval_results.csv

# Entire batch output (all languages at once):
python evaluate.py \\
    --json  results/*/*_igt_queries.json \\
    --guidelines grambank_queries.csv \\
    --output eval_results.csv \\
    --backend anthropic --model claude-haiku-4-5-20251001

# With local Qwen (default):
python evaluate.py \\
    --json  results/Aguaruna/igt_query_*.json \\
    --guidelines grambank_queries.csv \\
    --output eval_results.csv

Input JSON formats accepted
---------------------------
1. Single QueryResult object  (one query file, e.g. igt_query_151_*.json)
   {"query_id": ..., "query": ..., "answer": ..., ...}

2. Batch wrapper              (language-level file, e.g. Aguaruna_igt_queries.json)
   {"language": ..., "queries": [ {QueryResult}, {QueryResult}, ... ]}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Valid coding values across all Grambank features
VALID_CODES = {"0", "1", "2", "3", "?"}


# ── Coding guidelines loader ───────────────────────────────────────────────────

def load_guidelines(csv_path: Path) -> dict[str, dict]:
    """
    Parse the Grambank coding guidelines CSV.

    Returns:
        {
          "GB305": {
              "id":         "GB305",
              "query":      "Is there a phonologically independent reflexive pronoun?",
              "coding":     "1. Code 1 if ...\n2. Code 0 if ...",
              "definition": "Reflexive markers indicate ...",
          },
          ...
        }
    Keyed both by ID ("GB305") and by normalised query text for fuzzy matching.
    """
    by_id:    dict[str, dict] = {}
    by_query: dict[str, dict] = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
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
    """Normalise query string for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_guideline(
    query: str,
    query_id: str,
    guidelines: dict,
) -> dict | None:
    """
    Find the matching coding guideline for a query.

    Matching order:
    1. Exact ID match  (query_id field, e.g. "GB305")
    2. Exact query text match (normalised)
    3. Longest common-token overlap
    """
    by_id    = guidelines["by_id"]
    by_query = guidelines["by_query"]

    # 1. ID match
    if query_id and query_id in by_id:
        return by_id[query_id]

    # 2. Exact query match
    nq = _norm(query)
    if nq in by_query:
        return by_query[nq]

    # 3. Token overlap (best match)
    q_tokens = set(nq.split())
    best, best_score = None, 0
    for key, rec in by_query.items():
        score = len(q_tokens & set(key.split()))
        if score > best_score:
            best, best_score = rec, score

    if best and best_score >= 3:   # require at least 3 matching tokens
        return best

    return None


# ── JSON loader ────────────────────────────────────────────────────────────────

def load_query_results(json_path: Path) -> list[dict]:
    """
    Load one or more QueryResult dicts from a JSON file.

    Accepts both single-query files and batch wrapper files
    ({"queries": [...]}).
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data

    if "queries" in data and isinstance(data["queries"], list):
        # Batch wrapper — annotate with language name if present
        lang = data.get("language", json_path.parent.name)
        results = []
        for q in data["queries"]:
            q.setdefault("language", lang)
            results.append(q)
        return results

    # Single QueryResult
    data.setdefault("language", json_path.parent.name)
    return [data]


# ── Prompt builder ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a linguistic typologist evaluating fieldwork data against the Grambank coding scheme.

## TASK
Read the AGENT ANSWER below and apply the CODING INSTRUCTIONS to decide the correct code.
Output ONLY the single code character: 0, 1, 2, 3, or ?
No explanation. No punctuation. No extra text. Just one character.

## FEATURE
Question: {query}

## CODING INSTRUCTIONS
{coding}

## FEATURE DEFINITION
{definition}

## AGENT ANSWER
{answer}

## YOUR CODE (output exactly one character — 0, 1, 2, 3, or ?):"""


def build_prompt(query_result: dict, guideline: dict) -> str:
    return PROMPT_TEMPLATE.format(
        query      = guideline["query"],
        coding     = guideline["coding"],
        definition = guideline["definition"],
        answer     = query_result.get("answer", "").strip(),
    )


# ── LLM call with retry ────────────────────────────────────────────────────────

def call_llm(llm, prompt: str, retries: int = 3) -> str:
    """
    Call the LLM and extract a single valid code character.
    Retries up to `retries` times if the output is not a valid code.
    Returns '?' on total failure.
    """
    for attempt in range(1, retries + 1):
        raw = llm.generate(prompt, max_new_tokens=8, json_mode=False).strip()
        # Extract first character that is a valid code
        for ch in raw:
            if ch in VALID_CODES:
                return ch
        logger.warning(
            f"  Attempt {attempt}: unexpected output {raw!r}, retrying..."
        )
    logger.warning("  All retries failed — defaulting to '?'")
    return "?"


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(
    json_paths:   list[Path],
    guidelines:   dict,
    llm,
    output_path:  Path,
    show_answer:  bool = False,
) -> None:
    """Run evaluation over all query result files and write a CSV."""

    fieldnames = [
        "language", "query_id", "gb_id", "query",
        "code", "confidence", "needs_human_review", "audit_verdict",
        "llm_calls_used",
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
                logger.warning(f"  No guideline found for: {query[:60]}")
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
                    "llm_calls_used":     qr.get("token_usage", {}).get("llm_calls", ""),
                    **({"agent_answer": answer} if show_answer else {}),
                })
                continue

            n_matched += 1
            logger.info(
                f"  [{language}] {guideline['id']}  {query[:55]}..."
            )

            prompt = build_prompt(qr, guideline)
            t0     = time.time()
            code   = call_llm(llm, prompt)
            elapsed = time.time() - t0

            logger.info(f"    → code={code}  ({elapsed:.1f}s)")

            rows.append({
                "language":           language,
                "query_id":           query_id,
                "gb_id":              guideline["id"],
                "query":              query,
                "code":               code,
                "confidence":         qr.get("confidence", ""),
                "needs_human_review": qr.get("needs_human_review", ""),
                "audit_verdict":      qr.get("audit_verdict", ""),
                "llm_calls_used":     qr.get("token_usage", {}).get("llm_calls", ""),
                **({"agent_answer": answer} if show_answer else {}),
            })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"\nDone. {len(rows)} results written to {output_path}\n"
        f"  Matched   : {n_matched}\n"
        f"  Unmatched : {n_unmatched}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agent JSON outputs against Grambank coding guidelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--json",
        nargs="+",
        metavar="PATH",
        required=True,
        help=(
            "One or more agent output JSON files (glob patterns work in shells). "
            "Accepts single-query files (igt_query_*.json) or batch wrappers "
            "(*_igt_queries.json)."
        ),
    )
    parser.add_argument(
        "--guidelines",
        metavar="PATH",
        required=True,
        help="Grambank coding guidelines CSV (ID, Query, Coding, Definition columns).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="eval_results.csv",
        help="Output CSV path. Default: eval_results.csv",
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Include the agent's full answer text in the output CSV.",
    )

    # ── Model selection (mirrors batch_run.py) ──
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm", "openai", "anthropic"],
        help="LLM backend. Default: transformers",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Model name. Defaults per backend: "
            "transformers/vllm → Qwen/Qwen2.5-7B-Instruct, "
            "openai → gpt-4o-mini, "
            "anthropic → claude-haiku-4-5-20251001"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key for openai / anthropic backends.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        metavar="URL",
        help="Custom base URL for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="4-bit quantization for local models.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per query if LLM outputs an invalid code. Default: 3",
    )

    args = parser.parse_args(argv)

    # ── Resolve JSON paths ─────────────────────────────────────────────────────
    import glob
    json_paths: list[Path] = []
    for pattern in args.json:
        matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if not matches:
            # Treat as literal path
            p = Path(pattern)
            if p.exists():
                matches = [p]
            else:
                logger.warning(f"No files matched: {pattern}")
        json_paths.extend(matches)

    if not json_paths:
        logger.error("No JSON files found. Check your --json paths.")
        sys.exit(1)

    json_paths = sorted(set(json_paths))
    logger.info(f"Found {len(json_paths)} JSON file(s) to evaluate.")

    # ── Load guidelines ────────────────────────────────────────────────────────
    guidelines_path = Path(args.guidelines)
    if not guidelines_path.exists():
        logger.error(f"Guidelines CSV not found: {guidelines_path}")
        sys.exit(1)
    guidelines = load_guidelines(guidelines_path)

    # ── Load LLM ──────────────────────────────────────────────────────────────
    logger.info(f"Loading LLM — backend={args.backend}  model={args.model or '(default)'}")
    try:
        from llm import LLM          # new multi-backend llm.py
        llm = LLM(
            backend      = args.backend,
            model_name   = args.model,
            api_key      = args.api_key,
            api_base     = args.api_base,
            load_in_4bit = args.load_in_4bit,
        )
    except ImportError:
        # Fallback: original llm.py that only has QwenLLM
        logger.warning(
            "New LLM factory not found — falling back to original QwenLLM. "
            "Replace llm.py with the updated version to use OpenAI/Anthropic backends."
        )
        if args.backend not in ("transformers", "vllm"):
            logger.error(
                f"Backend '{args.backend}' requires the updated llm.py. "
                "Please replace llm.py with the new version and run: "
                "pip install openai   (for openai backend) or "
                "pip install anthropic  (for anthropic backend)."
            )
            sys.exit(1)
        from llm import QwenLLM
        llm = QwenLLM(
            model_name   = args.model or "Qwen/Qwen2.5-7B-Instruct",
            use_vllm     = (args.backend == "vllm"),
            load_in_4bit = args.load_in_4bit,
        )
    logger.info("LLM ready.")

    # ── Run ────────────────────────────────────────────────────────────────────
    evaluate(
        json_paths  = json_paths,
        guidelines  = guidelines,
        llm         = llm,
        output_path = Path(args.output),
        show_answer = args.show_answer,
    )


if __name__ == "__main__":
    main()