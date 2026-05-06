"""
multi_main.py — Multi-Language Cross-Linguistic Analysis Pipeline
=================================================================
Three phases:

  PHASE 0 — Global Planning
    Read ALL language TOCs / IGT summaries at once.
    LLM generates N unified questions applicable across the whole sample.
    Saved to: {output}/global_plan.json

  PHASE 1 — Question-First Investigation + Inline Synthesis
    For each question Q (one at a time):
      For each language L: run ReAct search → get raw answer
      Immediately synthesize all answers into one FeatureEntry:
        - Canonical feature name  (e.g. "Order of Subject, Object and Verb")
        - Language-neutral definition
        - Types attested in this sample  (SOV, VSO, ...)
        - Per type: description + which languages + how each realizes it + evidence
    Saved per question: {output}/features/NN_{slug}/
      {language}.json    ← raw QueryResult (for debugging / resume)
      feature.json       ← synthesized FeatureEntry

  PHASE 2 — Final Report
    Collect all FeatureEntry objects → render Markdown + JSON report.
    Saved to: {output}/universals.md  and  {output}/universals.json

No existing files are modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error(f"Config not found: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not cfg.get("languages"):
        logger.error("Config must have a non-empty 'languages' list.")
        sys.exit(1)
    for i, lang in enumerate(cfg["languages"]):
        if not lang.get("name"):
            logger.error(f"Language entry {i} missing 'name'.")
            sys.exit(1)
        if lang.get("igt_only") and not lang.get("igt"):
            logger.error(f"'{lang['name']}': igt_only=true needs 'igt' path.")
            sys.exit(1)
        if not lang.get("igt_only") and not lang.get("grammar"):
            logger.error(f"'{lang['name']}': 'grammar' required unless igt_only=true.")
            sys.exit(1)
    return cfg


def _slug(text: str, maxlen: int = 40) -> str:
    return re.sub(r"[^\w]+", "_", text.lower())[:maxlen].strip("_")


# ════════════════════════════════════════════════════════════════
# Phase 0 — Global planning
# ════════════════════════════════════════════════════════════════

def _load_existing_plan(output_dir: Path) -> Optional[list[str]]:
    f = output_dir / "global_plan.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        qs = [q["question"] for q in data.get("questions", []) if q.get("question")]
        return qs or None
    except Exception as e:
        logger.warning(f"Could not load existing plan: {e}")
        return None


def run_global_planning(
    language_configs: list[dict],
    llm,
    n_questions: int,
    output_dir:  Path,
) -> list[str]:
    from global_planner import GlobalPlanner
    planner = GlobalPlanner(llm=llm, n_questions=n_questions)
    plan    = planner.run(language_configs, output_dir=output_dir)
    return plan.question_strings()


# ════════════════════════════════════════════════════════════════
# Phase 1 — Agent pool initialization
# ════════════════════════════════════════════════════════════════

def init_agents(
    language_configs: list[dict],
    llm,
    max_iterations:   int,
) -> dict:
    """
    Initialize one agent per language. All agents are kept alive in memory
    so each grammar/IGT is loaded exactly once across all questions.

    Returns {language_name: agent_object}
    """
    from deep_agent import DeepLanguageResearchAgent
    from igt_agent  import IGTOnlyResearchAgent

    agents = {}
    n = len(language_configs)

    print(f"\n  Initializing {n} language agents (grammars/IGT loaded once)...\n")
    for i, lang_cfg in enumerate(language_configs, 1):
        name        = lang_cfg["name"]
        grammar     = lang_cfg.get("grammar")
        igt         = lang_cfg.get("igt")
        abbrev      = lang_cfg.get("abbreviations")
        igt_only    = lang_cfg.get("igt_only", False)

        print(f"  [{i}/{n}] Loading {name}...", end=" ", flush=True)
        try:
            if igt_only:
                agent = IGTOnlyResearchAgent(
                    language=name,
                    igt_path=igt,
                    llm=llm,
                    max_iterations_per_feature=max_iterations,
                    abbreviations_path=abbrev,
                )
            else:
                agent = DeepLanguageResearchAgent(
                    language=name,
                    grammar_path=grammar,
                    igt_path=igt,
                    llm=llm,
                    max_iterations_per_feature=max_iterations,
                    abbreviations_path=abbrev,
                )
            agents[name] = agent
            mode = "IGT-only" if igt_only else ("Grammar+IGT" if igt else "Grammar")
            print(f"✓ ({mode})")
        except Exception as exc:
            logger.error(f"Failed to init agent for '{name}': {exc}", exc_info=True)
            print(f"✗  FAILED: {exc}")

    print()
    return agents


# ════════════════════════════════════════════════════════════════
# Phase 1 — Per-question investigation + inline synthesis
# ════════════════════════════════════════════════════════════════

def run_one_question(
    question:      str,
    question_idx:  int,
    agents:        dict,
    llm,
    max_iterations: int,
    question_dir:  Path,
    skip_existing: bool,
) -> Optional[object]:   # returns FeatureEntry or None on total failure
    """
    For one question:
      1. Ask every agent → collect raw QueryResult dicts
      2. Synthesize into a FeatureEntry
      3. Save raw answers + feature.json to question_dir
    """
    from feature_synthesizer import FeatureSynthesizer, FeatureEntry

    question_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already done ──────────────────────────────────
    feature_file = question_dir / "feature.json"
    if skip_existing and feature_file.exists():
        try:
            data = json.loads(feature_file.read_text(encoding="utf-8"))
            print(f"  [SKIP] Already synthesized — loaded from {feature_file.name}")
            # Re-hydrate a minimal FeatureEntry for the report
            from feature_synthesizer import FeatureEntry, FeatureType, LanguageRealization
            types = []
            for t in data.get("types", []):
                lrs = [
                    LanguageRealization(
                        language=lr["language"],
                        realization=lr.get("realization",""),
                        evidence=lr.get("evidence",""),
                        confidence=lr.get("confidence",0.0),
                        raw_answer="",
                    )
                    for lr in t.get("languages", [])
                ]
                types.append(FeatureType(
                    type_label=t["type_label"],
                    description=t.get("description",""),
                    languages=lrs,
                ))
            return FeatureEntry(
                question=data.get("question", question),
                feature_name=data.get("feature_name", question),
                definition=data.get("definition",""),
                types=types,
                cross_linguistic_notes=data.get("cross_linguistic_notes",""),
                typological_significance=data.get("typological_significance",""),
                languages_covered=data.get("languages_covered",[]),
            )
        except Exception:
            pass   # fall through and re-run

    # ── Step A: investigate each language ──────────────────────
    lang_results: dict[str, dict] = {}
    n_agents = len(agents)

    print(f"\n  Investigating {n_agents} languages...")
    for j, (lang_name, agent) in enumerate(agents.items(), 1):
        raw_file = question_dir / f"{_slug(lang_name)}.json"

        # Check if this language's raw answer already saved
        if skip_existing and raw_file.exists():
            try:
                qr_dict = json.loads(raw_file.read_text(encoding="utf-8"))
                lang_results[lang_name] = qr_dict
                print(f"    [{j}/{n_agents}] {lang_name}: loaded from cache")
                continue
            except Exception:
                pass

        print(f"    [{j}/{n_agents}] {lang_name}: searching...", flush=True)
        try:
            qr     = agent.answer_query(question, max_iterations=max_iterations)
            # answer_query returns QueryResult (has .to_dict()) or a plain dict
            qr_dict = qr.to_dict() if hasattr(qr, "to_dict") else qr
            lang_results[lang_name] = qr_dict

            raw_file.write_text(
                json.dumps(qr_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            conf = qr_dict.get("confidence", 0)
            print(f"    [{j}/{n_agents}] {lang_name}: done (confidence={conf:.2f})")
        except Exception as exc:
            logger.error(f"    [{j}/{n_agents}] {lang_name} failed: {exc}", exc_info=True)
            print(f"    [{j}/{n_agents}] {lang_name}: FAILED — {exc}")

    if not lang_results:
        print(f"  No language data collected for question {question_idx}. Skipping synthesis.")
        return None

    # ── Step B: synthesize into FeatureEntry ───────────────────
    print(f"\n  Synthesizing cross-linguistic feature entry...")
    synthesizer = FeatureSynthesizer(llm)
    entry       = synthesizer.synthesize(question, lang_results)

    # Print brief result
    type_summary = " | ".join(
        f"{t.type_label} ({', '.join(t.language_names())})"
        for t in entry.types
    )
    print(f"  ✓ {entry.feature_name}")
    print(f"    Types: {type_summary}")

    # ── Save feature.json ──────────────────────────────────────
    feature_file.write_text(
        json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved: {feature_file}")

    return entry


# ════════════════════════════════════════════════════════════════
# Phase 2 — Final report
# ════════════════════════════════════════════════════════════════

def build_final_report(
    entries:        list,
    language_names: list[str],
    output_dir:     Path,
) -> None:
    from feature_synthesizer import build_report

    md   = build_report(entries, language_names)
    data = {
        "languages":           language_names,
        "n_languages":         len(language_names),
        "n_features":          len(entries),
        "features": [e.to_dict() for e in entries],
    }

    md_file   = output_dir / "universals.md"
    json_file = output_dir / "universals.json"

    md_file.write_text(md, encoding="utf-8")
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  Report : {md_file}")
    print(f"  Data   : {json_file}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-language typological analysis: global plan → question-first investigation → report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",  required=True, metavar="PATH",
                        help="Path to the languages.json config file.")
    parser.add_argument("--output",  default="output", metavar="DIR",
                        help="Root output directory. Default: output/")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vllm backend for the local LLM.")
    parser.add_argument("--max-iterations", type=int, default=10, metavar="N",
                        help="ReAct budget per question per language. Default: 10")
    parser.add_argument("--skip-existing", action="store_true",
                        help=(
                            "Reuse existing global_plan.json and skip individual "
                            "language answers / feature entries that are already on disk."
                        ))
    parser.add_argument("--report-only", action="store_true",
                        help=(
                            "Skip phases 0 and 1. Only collect existing feature.json "
                            "files under {output}/features/ and regenerate the report."
                        ))
    parser.add_argument("--no-report", action="store_true",
                        help="Run phases 0 and 1 only; skip the final report.")
    args = parser.parse_args()

    if args.report_only and args.no_report:
        parser.error("--report-only and --no-report are mutually exclusive.")

    # ── Config ────────────────────────────────────────────────────
    cfg              = load_config(args.config)
    language_configs = cfg["languages"]
    planning_cfg     = cfg.get("planning",  {})
    manual_queries   = cfg.get("queries")   or None
    n_questions      = int(planning_cfg.get("n_questions", 15))

    output_root  = Path(args.output)
    features_dir = output_root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    lang_names = [lc["name"] for lc in language_configs]

    print(f"\n{'━'*62}")
    print(f"  TYPOLOGY AUTOSEARCH — MULTI-LANGUAGE PIPELINE")
    print(f"{'━'*62}")
    print(f"  Languages : {', '.join(lang_names)}")
    print(f"  Output    : {output_root.resolve()}")
    print(f"  Mode      : {'manual queries' if manual_queries else f'auto-planning ({n_questions} questions)'}")
    print()

    # ── Init LLM ──────────────────────────────────────────────────
    if not args.report_only:
        from llm import QwenLLM
        llm = QwenLLM(use_vllm=args.use_vllm)
    else:
        llm = None

    # ════════════════════════════════════════════════════════════
    # PHASE 0 — Planning
    # ════════════════════════════════════════════════════════════

    queries: list[str]

    if args.report_only:
        queries = []   # not needed

    elif manual_queries:
        queries = manual_queries
        print(f"[Phase 0] Skipped — {len(queries)} explicit questions from config.")

    elif args.skip_existing and (saved := _load_existing_plan(output_root)):
        queries = saved
        print(f"[Phase 0] Skipped — reusing {len(queries)} questions from global_plan.json")

    else:
        print(f"[Phase 0] Global planning — reading all {len(language_configs)} languages...\n")
        from global_planner import GlobalPlanner
        planner = GlobalPlanner(llm=llm, n_questions=n_questions)
        try:
            plan    = planner.run(language_configs, output_dir=output_root)
            queries = plan.question_strings()
        except Exception as exc:
            logger.error(f"Planning failed: {exc}", exc_info=True)
            print(f"\n[WARNING] Planning failed. Falling back to generic questions.\n")
            queries = planner._fallback_plan([]).question_strings()

    # ════════════════════════════════════════════════════════════
    # PHASE 1 — Question-first investigation + inline synthesis
    # ════════════════════════════════════════════════════════════

    entries = []   # collected FeatureEntry objects

    if not args.report_only and queries:
        print(f"\n[Phase 1] {len(queries)} questions × {len(language_configs)} languages\n")
        print("  Strategy: for each question, all languages are investigated first,")
        print("  then immediately synthesized into one cross-linguistic feature entry.\n")

        # Initialize all agents once (grammars/IGT loaded once, kept in memory)
        agents = init_agents(language_configs, llm, args.max_iterations)

        if not agents:
            print("[ERROR] No agents could be initialized. Exiting.")
            sys.exit(1)

        for q_idx, question in enumerate(queries, 1):
            q_slug = f"{q_idx:02d}_{_slug(question)}"
            q_dir  = features_dir / q_slug

            print(f"\n{'─'*62}")
            print(f"  Q{q_idx}/{len(queries)}: {question}")
            print(f"{'─'*62}")

            entry = run_one_question(
                question=question,
                question_idx=q_idx,
                agents=agents,
                llm=llm,
                max_iterations=args.max_iterations,
                question_dir=q_dir,
                skip_existing=args.skip_existing,
            )
            if entry is not None:
                entries.append(entry)

    # ── If report-only, load all existing feature.json files ──────
    if args.report_only:
        print(f"[Report-only] Loading existing feature entries from {features_dir}...\n")
        from feature_synthesizer import FeatureEntry, FeatureType, LanguageRealization

        for q_dir in sorted(features_dir.iterdir()):
            f = q_dir / "feature.json"
            if not f.exists():
                continue
            try:
                data  = json.loads(f.read_text(encoding="utf-8"))
                types = []
                for t in data.get("types", []):
                    lrs = [
                        LanguageRealization(
                            language=lr["language"],
                            realization=lr.get("realization",""),
                            evidence=lr.get("evidence",""),
                            confidence=lr.get("confidence",0.0),
                            raw_answer="",
                        )
                        for lr in t.get("languages",[])
                    ]
                    types.append(FeatureType(
                        type_label=t["type_label"],
                        description=t.get("description",""),
                        languages=lrs,
                    ))
                entry = FeatureEntry(
                    question=data.get("question",""),
                    feature_name=data.get("feature_name",""),
                    definition=data.get("definition",""),
                    types=types,
                    cross_linguistic_notes=data.get("cross_linguistic_notes",""),
                    typological_significance=data.get("typological_significance",""),
                    languages_covered=data.get("languages_covered",[]),
                )
                entries.append(entry)
                print(f"  Loaded: {f.parent.name}")
            except Exception as exc:
                logger.warning(f"Could not load {f}: {exc}")

    # ════════════════════════════════════════════════════════════
    # PHASE 2 — Final report
    # ════════════════════════════════════════════════════════════

    if not args.no_report and entries:
        print(f"\n[Phase 2] Building report from {len(entries)} feature entries...")
        build_final_report(entries, lang_names, output_root)

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n{'━'*62}")
    print(f"  ALL DONE — {len(entries)} features analyzed across {len(lang_names)} languages")
    print(f"{'━'*62}")
    print(f"  Output: {output_root.resolve()}\n")


if __name__ == "__main__":
    main()
