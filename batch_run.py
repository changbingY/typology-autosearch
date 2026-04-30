"""
batch_run.py — Typology Autosearch Batch Runner
=================================================
Discovers languages by listing sub-folders inside --data-dir, loads every
query from a CSV file (ID, Query columns), and runs the pipeline for each
language × all queries.

Expected folder layout
----------------------
data/
    Choguita Rarámuri/          ← folder name  =  language name
        choguita_rarámuri_grammar.json
        choguita_rarámuri_igt.json
        choguita_rarámuri_abbreviations.txt   (optional)
    Chakali/
        chakali_grammar.json
        chakali_igt.json
        chakali_abbreviations.txt             (optional)

Data-file naming inside each language folder:
    {slug}_grammar.json         (required unless --igt-only)
    {slug}_igt.json             (required in --igt-only mode; optional otherwise)
    {slug}_abbreviations.txt    (always optional)
where slug = language_name.lower().replace(' ', '_')

CSV format (--queries-csv)
--------------------------
The CSV must have at least a column whose header contains "query" (case-insensitive).
An optional ID column is preserved in the output but not used for routing.

    ID,Query
    GB020,Are there definite or specific articles?
    GB021,Do indefinite nominals commonly have indefinite articles?
    ...

Usage
-----
# IGT-only, all queries from CSV, language folders inside data/:
python batch_run.py \\
    --data-dir  data/ \\
    --queries-csv grambank_queries.csv \\
    --output-dir results/ \\
    --igt-only

# Grammar + IGT mode:
python batch_run.py \\
    --data-dir  data/ \\
    --queries-csv grambank_queries.csv \\
    --output-dir results/

# Run only specific languages (skips other folders):
python batch_run.py \\
    --data-dir  data/ \\
    --queries-csv grambank_queries.csv \\
    --output-dir results/ \\
    --igt-only \\
    --only-languages "Choguita Rarámuri" "Chakali"

# Full typological pipeline (no CSV needed):
python batch_run.py \\
    --data-dir  data/ \\
    --output-dir results/ \\
    --igt-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(language: str) -> str:
    """'Choguita Rarámuri' → 'choguita_rarámuri'  (used for output folder names only)"""
    return language.strip().lower().replace(" ", "_")


def _find_file(lang_dir: Path, stem: str, suffix: str) -> Path | None:
    """
    Case-insensitive file lookup inside lang_dir.

    Tries candidates in this order:
      1. Exact match:            {stem}{suffix}          e.g. Abawiri_igt.json
      2. Lowercased stem:        {stem.lower()}{suffix}  e.g. abawiri_igt.json
      3. Any file whose name matches case-insensitively

    Returns the first Path that exists, or None.
    """
    # 1. Exact
    exact = lang_dir / f"{stem}{suffix}"
    if exact.exists():
        return exact

    # 2. Lower
    lower = lang_dir / f"{stem.lower()}{suffix}"
    if lower.exists():
        return lower

    # 3. Full case-insensitive scan
    target = f"{stem}{suffix}".lower()
    for p in lang_dir.iterdir():
        if p.name.lower() == target:
            return p

    return None


def discover_languages(data_dir: Path, only: list[str] | None = None) -> list[str]:
    """
    Return a sorted list of language names found as immediate sub-folders
    inside data_dir.  Hidden folders (starting with '.') and files are skipped.
    If `only` is provided, only those names are returned (order preserved from `only`).
    """
    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"--data-dir does not exist or is not a folder: {data_dir}"
        )

    found = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if only:
        found_set = set(found)
        missing   = [lang for lang in only if lang not in found_set]
        if missing:
            raise ValueError(
                f"--only-languages: these languages have no folder in {data_dir}: {missing}"
            )
        return only  # preserve caller's order

    return found


def load_queries(csv_path: Path) -> list[dict]:
    """
    Parse the queries CSV.

    Accepts any CSV that has a column whose header contains 'query'
    (case-insensitive).  An 'id' column is preserved when present.

    Returns:
        [{"id": "GB020", "query": "Are there definite or specific articles?"}, ...]
        'id' is empty string when the column is absent.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader  = csv.DictReader(fh)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        query_col = next(
            (h for h in headers if "query" in h.lower()),
            None,
        )
        if query_col is None:
            raise ValueError(
                f"CSV must have a column containing 'query' in its name. "
                f"Found columns: {headers}"
            )

        id_col = next(
            (h for h in headers if h.lower() in {"id", "feature_id", "gb_id"}),
            None,
        )

        rows = []
        for row in reader:
            q = row.get(query_col, "").strip()
            if not q:
                continue
            rows.append({
                "id":    row.get(id_col, "").strip() if id_col else "",
                "query": q,
            })

    logger.info(f"Loaded {len(rows)} queries from {csv_path.name}")
    return rows


def resolve_paths(language: str, data_dir: Path, igt_only: bool) -> dict:
    """
    Build and validate file paths for one language inside its sub-folder.

    File lookup is case-insensitive and tries both the original folder name
    and its lowercased form as the filename stem, so both
        Abawiri/Abawiri_igt.json   and   Abawiri/abawiri_igt.json
    are found automatically.

    Returns a dict with keys:
        grammar_path        : str | None
        igt_path            : str | None
        abbreviations_path  : str | None
        missing             : list[str]   — required files that were not found
    """
    lang_dir = data_dir / language           # e.g.  data/Abawiri/
    # Use the folder name as the stem (preserves original casing like "Abawiri")
    stem     = language.replace(" ", "_")    # e.g. "Abawiri", "Eastern_Geshiza"

    result = {
        "grammar_path":       None,
        "igt_path":           None,
        "abbreviations_path": None,
        "missing":            [],
    }

    grammar_file = _find_file(lang_dir, f"{stem}_grammar", ".json")
    igt_file     = _find_file(lang_dir, f"{stem}_igt",     ".json")
    abbrev_file  = _find_file(lang_dir, f"{stem}_abbreviations", ".txt")

    if not igt_only:
        if grammar_file:
            result["grammar_path"] = str(grammar_file)
        else:
            result["missing"].append(f"{lang_dir}/{stem}_grammar.json")

    if igt_file:
        result["igt_path"] = str(igt_file)
    elif igt_only:
        result["missing"].append(f"{lang_dir}/{stem}_igt.json")
    # In grammar mode, IGT is optional — no error if absent

    if abbrev_file:
        result["abbreviations_path"] = str(abbrev_file)

    return result


# ── Single-language runner ─────────────────────────────────────────────────────

def run_one_language(
    language:    str,
    paths:       dict,
    queries:     list[str],
    output_dir:  Path,
    igt_only:    bool,
    llm,                    # shared LLM instance — created once in main()
    max_iter:    int,
    conf_thresh: float,
) -> dict:
    """
    Run one language using a pre-loaded shared LLM.
    The LLM is never re-instantiated here — it is passed in from main()
    so that the expensive model load happens exactly once per batch run.

    Returns a summary dict (always — even on error).
    """
    summary = {
        "language":        language,
        "igt_only":        igt_only,
        "n_queries":       len(queries),
        "status":          "pending",
        "output_files":    [],
        "error":           None,
        "elapsed_seconds": None,
    }

    lang_output = output_dir / _slug(language)
    lang_output.mkdir(parents=True, exist_ok=True)

    grammar_path = paths["grammar_path"]
    igt_path     = paths["igt_path"]
    abbrev_path  = paths["abbreviations_path"]

    t0 = time.time()
    try:
        if igt_only:
            from igt_agent import IGTOnlyResearchAgent
            import json, re

            agent = IGTOnlyResearchAgent(
                language=language,
                igt_path=igt_path,
                llm=llm,
                max_iterations_per_feature=max_iter,
                abbreviations_path=abbrev_path,
            )

            if queries:
                results = []
                for i, query in enumerate(queries, 1):
                    logger.info(f"  [IGT Query {i}/{len(queries)}] {query[:70]}")
                    result = agent.answer_query(query, max_iterations=max_iter)
                    results.append(result)

                    safe = re.sub(r"[^\w\s-]", "", query[:40]).strip().replace(" ", "_").lower()
                    qfile = lang_output / f"igt_query_{i:03d}_{safe}.json"
                    with open(qfile, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                out = lang_output / f"{_slug(language)}_igt_queries.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(
                        {"language": language, "mode": "igt_only",
                         "queries": [r.to_dict() for r in results]},
                        f, ensure_ascii=False, indent=2,
                    )
            else:
                state = agent.run()
                out   = lang_output / f"{_slug(language)}_igt_features.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        else:
            from deep_agent import DeepLanguageResearchAgent, LanguageConfig
            import json, re

            agent = DeepLanguageResearchAgent(
                language=language,
                grammar_path=grammar_path,
                igt_path=igt_path,
                llm=llm,
                max_iterations_per_feature=max_iter,
                confidence_threshold=conf_thresh,
                abbreviations_path=abbrev_path,
            )

            if queries:
                results = []
                for i, query in enumerate(queries, 1):
                    logger.info(f"  [Query {i}/{len(queries)}] {query[:70]}")
                    result = agent.answer_query(query, max_iterations=max_iter)
                    results.append(result)

                    safe  = re.sub(r"[^\w\s-]", "", query[:40]).strip().replace(" ", "_").lower()
                    qfile = lang_output / f"query_{i:03d}_{safe}.json"
                    with open(qfile, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                out = lang_output / f"{_slug(language)}_queries.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(
                        {"language": language,
                         "queries": [r.to_dict() for r in results]},
                        f, ensure_ascii=False, indent=2,
                    )
            else:
                state = agent.run()
                out   = lang_output / f"{_slug(language)}_features.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        summary["status"]       = "success"
        summary["output_files"] = [str(out)]
        logger.info(f"[{language}] ✓  ({time.time()-t0:.0f}s)")

    except Exception as exc:
        summary["status"] = "error"
        summary["error"]  = traceback.format_exc()
        logger.error(f"[{language}] ✗  {exc}")

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run Typology Autosearch: discover languages from sub-folders "
            "and run all CSV queries on each."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ──
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        required=True,
        help=(
            "Root folder whose immediate sub-folders are language names. "
            "Each sub-folder must contain the language's data files."
        ),
    )
    parser.add_argument(
        "--queries-csv",
        metavar="PATH",
        default=None,
        help=(
            "CSV with ID and Query columns. Every query is sent to every language. "
            "Omit to run the full typological pipeline instead."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default="output",
        help="Root directory for all results. Default: output/",
    )

    # ── Language filter ──
    parser.add_argument(
        "--only-languages",
        nargs="+",
        metavar="LANG",
        default=None,
        help="Process only these languages (must match folder names exactly).",
    )

    # ── Mode ──
    parser.add_argument(
        "--igt-only",
        action="store_true",
        help="Run every language in IGT-only mode (no grammar book required).",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for faster inference.",
    )

    # ── Tuning ──
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        metavar="N",
        help="ReAct loop budget per query. Default: 15",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        metavar="F",
        help="Minimum confidence for confirmed features. Default: 0.75",
    )

    # ── Error handling ──
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip languages with missing required files instead of aborting.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue to the next language if a pipeline run raises an exception.",
    )

    args = parser.parse_args(argv)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover languages from folder names ───────────────────────────────────
    try:
        languages = discover_languages(data_dir, only=args.only_languages)
    except (NotADirectoryError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    if not languages:
        logger.error(f"No language sub-folders found in: {data_dir}")
        sys.exit(1)

    logger.info(f"Discovered {len(languages)} language folder(s):")
    for lang in languages:
        logger.info(f"  • {lang}")

    # ── Load queries from CSV ──────────────────────────────────────────────────
    query_rows: list[dict] = []
    if args.queries_csv:
        try:
            query_rows = load_queries(Path(args.queries_csv))
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            sys.exit(1)

    query_texts = [r["query"] for r in query_rows]  # plain strings for pipeline

    # ── Load LLM ONCE for the whole batch ─────────────────────────────────────
    # This is the critical fix: the model is heavy (~14GB). Loading it inside
    # each language's run would reload it 34 times. We load it once here and
    # pass the same instance to every agent.
    logger.info("Loading LLM (this happens once for the whole batch)...")
    from llm import QwenLLM
    llm = QwenLLM(use_vllm=args.use_vllm)
    logger.info("LLM ready.")

    # ── Batch loop ─────────────────────────────────────────────────────────────
    batch_start = time.time()
    summaries:  list[dict] = []
    n_success = n_skipped = n_error = 0

    logger.info(
        f"\n{'='*64}\n"
        f"Batch : {len(languages)} language(s) × {len(query_texts)} quer{'y' if len(query_texts)==1 else 'ies'}\n"
        f"Mode  : {'IGT-only' if args.igt_only else 'Grammar + IGT'}\n"
        f"{'='*64}"
    )

    for idx, language in enumerate(languages, 1):
        logger.info(f"\n[{idx}/{len(languages)}] {language}")

        paths = resolve_paths(language, data_dir, args.igt_only)

        if paths["missing"]:
            msg = (
                f"[{language}] Missing required file(s): "
                + ", ".join(paths["missing"])
            )
            if args.skip_missing:
                logger.warning(f"{msg} — skipping")
                summaries.append({
                    "language":        language,
                    "status":          "skipped",
                    "reason":          msg,
                    "elapsed_seconds": 0,
                })
                n_skipped += 1
                continue
            else:
                logger.error(msg)
                sys.exit(1)

        summary = run_one_language(
            language=language,
            paths=paths,
            queries=query_texts,
            output_dir=output_dir,
            igt_only=args.igt_only,
            llm=llm,
            max_iter=args.max_iterations,
            conf_thresh=args.confidence_threshold,
        )
        summaries.append(summary)

        if summary["status"] == "success":
            n_success += 1
        else:
            n_error += 1
            if not args.skip_errors:
                logger.error(
                    f"Aborting after error on '{language}'. "
                    "Use --skip-errors to continue past failures."
                )
                break

    # ── Write batch summary JSON ───────────────────────────────────────────────
    total_elapsed = time.time() - batch_start

    batch_report = {
        "data_dir":          str(data_dir),
        "output_dir":        str(output_dir),
        "igt_only":          args.igt_only,
        "max_iterations":    args.max_iterations,
        "confidence_thresh": args.confidence_threshold,
        "n_languages":       len(languages),
        "n_queries":         len(query_texts),
        "n_success":         n_success,
        "n_skipped":         n_skipped,
        "n_error":           n_error,
        "total_elapsed_s":   round(total_elapsed, 1),
        # Store query IDs (GB020, GB021…) alongside their text for traceability
        "queries": [
            {"id": r["id"], "query": r["query"]} for r in query_rows
        ],
        "languages": summaries,
    }

    report_path = output_dir / "batch_summary.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(batch_report, fh, ensure_ascii=False, indent=2)

    # ── Console summary ────────────────────────────────────────────────────────
    mins = total_elapsed / 60
    print(f"\n{'='*64}")
    print(f"Batch complete  ({mins:.1f} min  /  {total_elapsed:.0f}s)")
    print(f"  Languages processed : {len(languages)}")
    print(f"  Queries per language: {len(query_texts)}")
    print(f"  Success  : {n_success}")
    print(f"  Skipped  : {n_skipped}")
    print(f"  Error    : {n_error}")
    print(f"  Report   : {report_path}")
    if n_error:
        print("\nFailed languages:")
        for s in summaries:
            if s.get("status") == "error":
                first = (s.get("error") or "").splitlines()[0]
                print(f"  • {s['language']}  —  {first}")
    print(f"{'='*64}\n")

    sys.exit(1 if n_error else 0)


if __name__ == "__main__":
    main()