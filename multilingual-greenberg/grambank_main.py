"""
grambank_main.py — Grambank Feature Annotation Pipeline
=========================================================
Runs each Grambank feature question through the deep search pipeline
(grammar + IGT, or IGT-only) and assigns a coded label per the
Grambank coding policy.

Design
------
Each feature is treated as fully independent. The Grambank coding policy
is appended directly to the research query, so the model answers the
linguistic question and assigns the label in a single pass — no separate
labeling step is needed.

Usage examples
--------------
# Full grammar + IGT, all 195 features:
python grambank_main.py \\
    --language "Aguaruna" \\
    --grammar  Aguaruna_grammar.json \\
    --igt      Aguaruna_igt.json \\
    --grambank Typology-grambank-valid_-_grambank.csv \\
    --abbreviations Aguaruna_abbreviations.txt \\
    --output   results/grambank/

# Only specific features:
python grambank_main.py \\
    --language "Aguaruna" \\
    --grammar  Aguaruna_grammar.json \\
    --igt      Aguaruna_igt.json \\
    --grambank Typology-grambank-valid_-_grambank.csv \\
    --ids GB020 GB021 GB130 \\
    --output   results/grambank/

# IGT-only mode:
python grambank_main.py \\
    --language "Aguaruna" \\
    --igt      Aguaruna_igt.json \\
    --igt-only \\
    --grambank Typology-grambank-valid_-_grambank.csv \\
    --abbreviations Aguaruna_abbreviations.txt \\
    --output   results/grambank/

Output files
------------
results/grambank/
  GB020_ARTDef.json              — full QueryResult for each feature
  GB021_ARTIndef.json
  ...
  aguaruna_grambank_labels.csv   — summary table (ID, query, label, confidence, ...)
  aguaruna_grambank_labels.json  — all results in one JSON
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

from grambank_labeler import GrambankFeature, load_grambank_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Grambank feature queries through the deep-search pipeline and label them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--language",  required=True,
                   help="Language name (e.g. 'Aguaruna')")
    p.add_argument("--grammar",   default=None,
                   help="Path to grammar JSON. Required unless --igt-only.")
    p.add_argument("--igt",       default=None,
                   help="Path to IGT JSON (optional alongside grammar; required for --igt-only).")
    p.add_argument("--grambank",  required=True,
                   help="Path to Grambank CSV (ID, Query, Coding, Definition, ...).")
    p.add_argument("--abbreviations", default=None, metavar="PATH",
                   help="Path to tab-separated gloss abbreviation file.")
    p.add_argument("--output",    default="output/grambank",
                   help="Directory for output files (default: output/grambank/).")
    p.add_argument("--ids", nargs="*", default=None, metavar="GB020",
                   help="Run only these Grambank IDs (e.g. --ids GB020 GB021). "
                        "Default: all features in the CSV.")
    p.add_argument("--igt-only", action="store_true",
                   help="Use IGT-only agent (no grammar required).")
    p.add_argument("--use-vllm",         action="store_true")
    p.add_argument("--max-iterations",   type=int, default=10,
                   help="ReAct loop budget per feature query (default: 10).")
    p.add_argument("--confidence-threshold", type=float, default=0.70,
                   help="Confidence threshold for the research agent (default: 0.70).")
    p.add_argument("--resume", action="store_true",
                   help="Skip features whose output JSON already exists in --output.")

    return p


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _safe_filename(feature: GrambankFeature) -> str:
    base = re.sub(r"[^\w\-]", "_", feature.id_desc or feature.grambank_id)
    return base.strip("_")


def _save_feature_json(
    output_dir: Path,
    feature: GrambankFeature,
    query_result,
) -> Path:
    fname = output_dir / f"{_safe_filename(feature)}.json"
    payload = {
        "grambank_id":      feature.grambank_id,
        "query":            feature.query,
        "id_desc":          feature.id_desc,
        "label":            query_result.grambank_label,
        "label_reasoning":  query_result.grambank_reasoning,
        "research": query_result.to_dict() if hasattr(query_result, "to_dict") else {},
    }
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return fname


def _save_summary_csv(
    output_dir: Path,
    language: str,
    rows: list[dict],
) -> Path:
    lang_slug = language.lower().replace(" ", "_")
    csv_path  = output_dir / f"{lang_slug}_grambank_labels.csv"
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _save_summary_json(
    output_dir: Path,
    language: str,
    all_results: list[dict],
) -> Path:
    lang_slug = language.lower().replace(" ", "_")
    json_path = output_dir / f"{lang_slug}_grambank_labels.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"language": language, "features": all_results},
            fh, ensure_ascii=False, indent=2,
        )
    return json_path


def _print_progress_header(
    feature: GrambankFeature,
    idx: int,
    total: int,
) -> None:
    bar = "═" * 60
    print(f"\n{bar}", flush=True)
    print(f"[{idx}/{total}] {feature.grambank_id}  —  {feature.query}", flush=True)
    print(bar, flush=True)


def _print_label_box(query_result, coding_policy: str = "") -> None:
    w = 58
    label     = query_result.grambank_label or ""
    reasoning = query_result.grambank_reasoning or "(no reasoning provided)"
    print(f"\n  ┌─ GRAMBANK LABEL {'─' * w}", flush=True)
    print(f"  │  Label     : {label}", flush=True)
    print(f"  │  Reasoning : {reasoning[:110]}", flush=True)
    if coding_policy:
        print(f"  │", flush=True)
        print(f"  │  CODING POLICY:", flush=True)
        for line in coding_policy.strip().splitlines():
            print(f"  │    {line[:110]}", flush=True)
    print(f"  └─{'─' * w}\n", flush=True)


def _make_csv_row(gid: str, feature: GrambankFeature, result_dict: dict) -> dict:
    research = result_dict.get("research", {})
    return {
        "grambank_id":         gid,
        "id_desc":             feature.id_desc,
        "query":               feature.query,
        "label":               result_dict.get("label", ""),
        "label_reasoning":     (result_dict.get("label_reasoning") or "")[:200],
        "research_confidence": research.get("confidence", ""),
        "needs_human_review":  research.get("needs_human_review", ""),
        "audit_verdict":       research.get("audit_verdict", ""),
        "igt_support":         research.get("igt_support", ""),
        "answer_snippet":      (research.get("answer", "") or "")[:150],
    }


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run_grambank_pipeline(
    language: str,
    grammar_path: str | None,
    igt_path: str | None,
    grambank_path: str,
    output_dir: str,
    ids_filter: list[str] | None = None,
    abbreviations_path: str | None = None,
    igt_only: bool = False,
    use_vllm: bool = False,
    max_iterations: int = 10,
    confidence_threshold: float = 0.70,
    resume: bool = False,
) -> list[dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features_all = load_grambank_csv(grambank_path)

    if ids_filter:
        features = {k: v for k, v in features_all.items() if k in ids_filter}
        missing  = [i for i in ids_filter if i not in features_all]
        if missing:
            logger.warning(f"IDs not found in CSV: {missing}")
    else:
        features = features_all

    if not features:
        logger.error("No features to process. Check --ids or --grambank path.")
        return []

    logger.info(f"Processing {len(features)} Grambank feature(s) for '{language}'.")

    from llm import QwenLLM
    llm = QwenLLM(use_vllm=use_vllm)

    if igt_only:
        if not igt_path:
            raise ValueError("--igt-only requires --igt <path>")
        from igt_agent import IGTOnlyResearchAgent
        agent = IGTOnlyResearchAgent(
            language                   = language,
            igt_path                   = igt_path,
            llm                        = llm,
            max_iterations_per_feature = max_iterations,
            abbreviations_path         = abbreviations_path,
        )
        def _answer_query(query: str, grambank_policy: str = ""):
            return agent.answer_query(
                query, max_iterations=max_iterations, grambank_policy=grambank_policy
            )
    else:
        if not grammar_path:
            raise ValueError("--grammar is required unless --igt-only is set.")
        from deep_agent import DeepLanguageResearchAgent
        agent = DeepLanguageResearchAgent(
            language                   = language,
            grammar_path               = grammar_path,
            igt_path                   = igt_path,
            llm                        = llm,
            max_iterations_per_feature = max_iterations,
            confidence_threshold       = confidence_threshold,
            abbreviations_path         = abbreviations_path,
        )
        def _answer_query(query: str, grambank_policy: str = ""):
            return agent.answer_query(
                query, max_iterations=max_iterations, grambank_policy=grambank_policy
            )

    all_results: list[dict] = []
    csv_rows:    list[dict] = []
    total = len(features)

    for idx, (gid, feature) in enumerate(features.items(), 1):
        _print_progress_header(feature, idx, total)

        out_json = output_path / f"{_safe_filename(feature)}.json"
        if resume and out_json.exists():
            print(f"  [RESUME] skipping {gid} — output already exists.", flush=True)
            try:
                with open(out_json, encoding="utf-8") as fh:
                    existing = json.load(fh)
                all_results.append(existing)
                csv_rows.append(_make_csv_row(gid, feature, existing))
            except Exception as e:
                logger.warning(f"  Could not re-read {out_json}: {e}")
            continue

        try:
            query_result = _answer_query(
                feature.query,
                grambank_policy=feature.coding_policy,
            )
        except Exception as e:
            logger.error(f"[{gid}] Research failed: {e}", exc_info=True)
            from state import QueryResult
            query_result = QueryResult(
                query_id=gid.lower(),
                query=feature.query,
                phenomena=[],
                linguistic_definition="",
                structural_description="",
                answer="[Research failed — no evidence gathered.]",
                key_evidence=[],
                igt_examples_used=[],
                igt_support=False,
                search_trace=[],
                confidence=0.0,
                needs_human_review=True,
                review_reason=f"Research step raised exception: {e}",
                audit_verdict="overturned",
                audit_objections=[],
                token_usage={},
                grambank_label="",
                grambank_reasoning=f"Research failed: {e}",
            )

        if not query_result.grambank_label:
            logger.warning(f"[{gid}] grambank_label missing from model output.")
            query_result.grambank_reasoning = query_result.grambank_reasoning or "Label not produced by model."

        _print_label_box(query_result, coding_policy=feature.coding_policy)

        saved_path = _save_feature_json(output_path, feature, query_result)
        print(f"  Saved: {saved_path}", flush=True)

        result_dict = {
            "grambank_id":     gid,
            "query":           feature.query,
            "id_desc":         feature.id_desc,
            "label":           query_result.grambank_label,
            "label_reasoning": query_result.grambank_reasoning,
            "research":        query_result.to_dict() if hasattr(query_result, "to_dict") else {},
        }
        all_results.append(result_dict)
        csv_rows.append(_make_csv_row(gid, feature, result_dict))

        _save_summary_csv(output_path, language, csv_rows)
        _save_summary_json(output_path, language, all_results)

    # ── Final summary ──────────────────────────────────────────────
    label_counts: dict[str, int] = {}
    for r in all_results:
        lab = r.get("label", "")
        label_counts[lab] = label_counts.get(lab, 0) + 1

    print(f"\n{'═'*60}", flush=True)
    print(f"[{language}] Grambank annotation complete.", flush=True)
    print(f"  Features processed : {total}", flush=True)
    for val, cnt in sorted(label_counts.items()):
        pct = 100 * cnt / max(total, 1)
        print(f"  Label {val:<3}          : {cnt:>3} ({pct:.1f}%)", flush=True)

    csv_path  = _save_summary_csv(output_path, language, csv_rows)
    json_path = _save_summary_json(output_path, language, all_results)
    print(f"  CSV  : {csv_path}", flush=True)
    print(f"  JSON : {json_path}", flush=True)
    print(f"{'═'*60}\n", flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.igt_only:
        if not args.igt:
            _build_parser().error("--igt-only requires --igt <path>")
    else:
        if not args.grammar:
            _build_parser().error("--grammar is required unless --igt-only is set.")

    run_grambank_pipeline(
        language             = args.language,
        grammar_path         = args.grammar,
        igt_path             = args.igt,
        grambank_path        = args.grambank,
        output_dir           = args.output,
        ids_filter           = args.ids,
        abbreviations_path   = args.abbreviations,
        igt_only             = args.igt_only,
        use_vllm             = args.use_vllm,
        max_iterations       = args.max_iterations,
        confidence_threshold = args.confidence_threshold,
        resume               = args.resume,
    )