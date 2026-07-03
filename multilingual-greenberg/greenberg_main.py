"""
greenberg_main.py — CLI for Greenberg Universal Verification
=============================================================
Tests one or more Greenberg universals against a multi-language sample.

Usage examples
--------------
# Test a single universal by ID:
python greenberg_main.py \\
    --config    languages.json \\
    --greenberg greenberg_universals.csv \\
    --ids       U1 \\
    --output    results/greenberg/

# Test several universals:
python greenberg_main.py \\
    --config    languages.json \\
    --greenberg greenberg_universals.csv \\
    --ids       U1 U2 U3 U4 \\
    --output    results/greenberg/

# Test a whole domain:
python greenberg_main.py \\
    --config    languages.json \\
    --greenberg greenberg_universals.csv \\
    --domain    WORD_ORDER \\
    --output    results/greenberg/

# Test all universals (runs long!):
python greenberg_main.py \\
    --config    languages.json \\
    --greenberg greenberg_universals.csv \\
    --all \\
    --output    results/greenberg/

# Pass a universal as free text (no CSV needed for the statement itself):
python greenberg_main.py \\
    --config     languages.json \\
    --greenberg  greenberg_universals.csv \\
    --statement  "If a language has dominant SOV order, it is almost always postpositional." \\
    --stmt-logic implication \\
    --output     results/greenberg/

# Resume a crashed run without re-running finished languages:
python greenberg_main.py \\
    --config    languages.json \\
    --greenberg greenberg_universals.csv \\
    --ids       U1 U2 U3 \\
    --output    results/greenberg/ \\
    --skip-existing

Config file format (same as multi_main.py):
--------------------------------------------
{
  "languages": [
    {
      "name": "Aguaruna",
      "grammar": "Aguaruna_grammar.json",
      "igt": "Aguaruna_igt.json",
      "abbreviations": "Aguaruna_abbrevs.txt"
    },
    {
      "name": "Raramuri",
      "grammar": "Raramuri_grammar.json"
    },
    {
      "name": "Duna",
      "igt": "Duna_igt.json",
      "igt_only": true
    }
  ]
}

Output layout
-------------
{output}/
  universals_report.md        ← full Markdown report (all universals)
  universals_report.json      ← machine-readable summary
  {uid}/                      ← one directory per universal
    universal_verdict.json    ← aggregated verdict + all per-language verdicts
    {Language}.json           ← raw QueryResult for each language (for debugging)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Config loader  (same schema as multi_main.py)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Agent initializer  (mirrors multi_main.init_agents)
# ═══════════════════════════════════════════════════════════════════════════════

def init_agents(language_configs: list[dict], llm, max_iterations: int) -> dict:
    """
    Initialize one research agent per language.
    Each grammar/IGT corpus is loaded exactly ONCE and kept in memory
    across all universals — expensive I/O happens only at startup.

    Returns {language_name: agent_object}
    """
    from deep_agent import DeepLanguageResearchAgent
    from igt_agent  import IGTOnlyResearchAgent

    agents = {}
    n = len(language_configs)
    print(f"\n  Initializing {n} language agent(s)...\n")

    for i, cfg in enumerate(language_configs, 1):
        name     = cfg["name"]
        grammar  = cfg.get("grammar")
        igt      = cfg.get("igt")
        abbrev   = cfg.get("abbreviations")
        igt_only = cfg.get("igt_only", False)

        print(f"  [{i}/{n}] Loading {name}...", end=" ", flush=True)
        try:
            if igt_only:
                agent = IGTOnlyResearchAgent(
                    language              = name,
                    igt_path              = igt,
                    llm                   = llm,
                    max_iterations_per_feature = max_iterations,
                    abbreviations_path    = abbrev,
                )
            else:
                agent = DeepLanguageResearchAgent(
                    language              = name,
                    grammar_path          = grammar,
                    igt_path              = igt,
                    llm                   = llm,
                    max_iterations_per_feature = max_iterations,
                    abbreviations_path    = abbrev,
                )
            agents[name] = agent
            mode = "IGT-only" if igt_only else ("Grammar+IGT" if igt else "Grammar")
            print(f"✓ ({mode})")
        except Exception as exc:
            logger.error(f"Failed to init agent for '{name}': {exc}", exc_info=True)
            print(f"✗  FAILED: {exc}")

    print()
    return agents


# ═══════════════════════════════════════════════════════════════════════════════
# Universal selection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def select_universals(
    all_universals: dict,
    ids:            list[str] | None,
    domain:         str | None,
    run_all:        bool,
    statement:      str | None,
    stmt_logic:     str,
    stmt_domain:    str,
    stmt_antecedent: str,
    stmt_consequent: str,
) -> list:
    """
    Return the list of GreenbergUniversal objects to test.

    Priority:
      1. --statement (free-text universal defined on the command line)
      2. --ids (specific IDs from the CSV)
      3. --domain (all universals in that domain)
      4. --all (everything in the CSV)
    """
    from greenberg_verifier import GreenbergUniversal

    if statement:
        u = GreenbergUniversal(
            uid        = "CUSTOM",
            statement  = statement,
            antecedent = stmt_antecedent or statement,
            consequent = stmt_consequent or statement,
            logic      = stmt_logic,
            domain     = stmt_domain,
            source     = "user-supplied",
        )
        return [u]

    if not all_universals:
        logger.error("No universals loaded — check --greenberg path.")
        sys.exit(1)

    if ids:
        selected = []
        for uid in ids:
            if uid in all_universals:
                selected.append(all_universals[uid])
            else:
                logger.warning(f"Universal '{uid}' not found in CSV — skipping.")
        if not selected:
            logger.error("None of the requested IDs found in the CSV.")
            sys.exit(1)
        return selected

    if domain:
        selected = [u for u in all_universals.values()
                    if u.domain.upper() == domain.upper()]
        if not selected:
            available = sorted({u.domain for u in all_universals.values()})
            logger.error(
                f"No universals found for domain '{domain}'. "
                f"Available domains: {', '.join(available)}"
            )
            sys.exit(1)
        return selected

    if run_all:
        return list(all_universals.values())

    # Nothing selected
    logger.error(
        "No universals selected. Use --ids, --domain, --all, or --statement."
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test Greenberg universals against a multi-language sample using "
            "deep grammar + IGT search."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Config ────────────────────────────────────────────────────
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to languages.json (same format as multi_main.py).",
    )
    parser.add_argument(
        "--greenberg", default=None, metavar="PATH",
        help="Path to Greenberg universals CSV. Required unless --statement is used.",
    )
    parser.add_argument(
        "--output", default="output/greenberg", metavar="DIR",
        help="Output directory. Default: output/greenberg/",
    )

    # ── Universal selection ───────────────────────────────────────
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--ids", nargs="+", metavar="ID",
        help="Test specific universals by ID (e.g. --ids U1 U2 U4).",
    )
    selection.add_argument(
        "--domain", metavar="DOMAIN",
        help=(
            "Test all universals in a domain "
            "(WORD_ORDER, AGREEMENT, CASE, MORPHOLOGICAL_COMPLEXITY, "
            "TENSE_ASPECT_MODALITY, NOUN_PHRASE_STRUCTURE, INFORMATION_STRUCTURE)."
        ),
    )
    selection.add_argument(
        "--all", dest="run_all", action="store_true",
        help="Test ALL universals in the CSV (may run very long).",
    )
    selection.add_argument(
        "--statement", metavar="TEXT",
        help=(
            "Free-text universal to test, if not in the CSV. "
            "Must also supply --stmt-antecedent and --stmt-consequent."
        ),
    )

    # ── Free-text statement options ───────────────────────────────
    parser.add_argument(
        "--stmt-antecedent", default="", metavar="TEXT",
        help="The 'IF' clause for a --statement universal.",
    )
    parser.add_argument(
        "--stmt-consequent", default="", metavar="TEXT",
        help="The 'THEN' clause for a --statement universal.",
    )
    parser.add_argument(
        "--stmt-logic",
        choices=["absolute", "implication", "correlation"],
        default="implication",
        help="Logic type for a --statement universal. Default: implication.",
    )
    parser.add_argument(
        "--stmt-domain", default="UNKNOWN", metavar="DOMAIN",
        help="Domain label for a --statement universal. Default: UNKNOWN.",
    )

    # ── Runtime options ───────────────────────────────────────────
    parser.add_argument(
        "--openrouter", action="store_true",
        help=(
            "Use OpenRouter instead of a local Qwen model. "
            "Requires --api-key or the OPENROUTER_API_KEY env variable."
        ),
    )
    parser.add_argument(
        "--api-key", default="", metavar="KEY",
        help=(
            "OpenRouter API key (sk-or-…). "
            "Alternatively, set the OPENROUTER_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--or-model", default="google/gemma-4-31b-it", metavar="MODEL",
        help=(
            "OpenRouter model string. Default: google/gemma-4-31b-it. "
            "Only used when --openrouter is set."
        ),
    )
    parser.add_argument(
        "--token-scale", type=float, default=2.0, metavar="X",
        help=(
            "Multiply every max_new_tokens budget by X before sending to OpenRouter. "
            "Default: 2.0 — cloud models like Gemma write more verbose JSON than the "
            "local Qwen the budgets were tuned for; 2x prevents mid-JSON truncation "
            "on planning/decision calls (512->1024) and audit calls (256->512). "
            "Only used when --openrouter is set."
        ),
    )
    parser.add_argument(
        "--max-tokens-cap", type=int, default=8192, metavar="N",
        help=(
            "Hard upper limit on output tokens after scaling. Default: 8192. "
            "Only used when --openrouter is set."
        ),
    )
    parser.add_argument(
        "--force-max-tokens", action="store_true",
        help=(
            "Ignore per-call token budgets and always send max_tokens_cap to "
            "the API. Every call — planning, decisions, conclusions, audits — "
            "gets the same ceiling. Eliminates all truncation at the cost of "
            "higher latency and spend on short calls. "
            "Only used when --openrouter is set."
        ),
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vllm backend for the local Qwen model (faster, needs GPU). "
             "Ignored when --openrouter is set.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, metavar="N",
        help="ReAct loop budget per language per universal. Default: 10.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip universals / languages that already have saved results on disk. "
            "Useful for resuming a crashed run."
        ),
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help=(
            "Do not run any new queries. Load existing universal_verdict.json "
            "files from the output directory and regenerate the report only."
        ),
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────
    if not args.statement and not args.greenberg:
        parser.error("--greenberg is required unless you use --statement.")
    if not args.statement and not (args.ids or args.domain or args.run_all):
        parser.error(
            "Specify which universals to test: --ids, --domain, --all, or --statement."
        )

    # ── Load universals ───────────────────────────────────────────
    from greenberg_verifier import load_greenberg_csv, GreenbergVerifier

    all_universals = {}
    if args.greenberg:
        all_universals = load_greenberg_csv(args.greenberg)

    universals = select_universals(
        all_universals   = all_universals,
        ids              = args.ids,
        domain           = args.domain,
        run_all          = args.run_all,
        statement        = args.statement,
        stmt_logic       = args.stmt_logic,
        stmt_domain      = args.stmt_domain,
        stmt_antecedent  = args.stmt_antecedent,
        stmt_consequent  = args.stmt_consequent,
    )

    # ── Load config ───────────────────────────────────────────────
    cfg              = load_config(args.config)
    language_configs = cfg["languages"]
    lang_names       = [lc["name"] for lc in language_configs]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────
    print(f"\n{'━'*64}")
    print(f"  GREENBERG UNIVERSAL VERIFIER")
    print(f"{'━'*64}")
    print(f"  Universals : {len(universals)}")
    print(f"  Languages  : {', '.join(lang_names)}")
    print(f"  Output     : {output_dir.resolve()}")
    if args.report_only:
        print(f"  Mode       : report-only (no new queries)")
    elif args.skip_existing:
        print(f"  Mode       : resume (skip existing results)")
    print()

    # ── Report-only mode: load saved verdicts ─────────────────────
    if args.report_only:
        from greenberg_verifier import UniversalVerdict, LanguageVerdict, GreenbergVerifier

        saved_verdicts = []
        for u in universals:
            vf = output_dir / u.uid / "universal_verdict.json"
            if vf.exists():
                uv = GreenbergVerifier._load_verdict(vf, u)
                if uv:
                    saved_verdicts.append(uv)
                    print(f"  Loaded: {u.uid}")
            else:
                print(f"  Missing: {u.uid} — no verdict file found, skipping.")

        if saved_verdicts:
            # Create a dummy verifier just for report building
            from llm import QwenLLM
            dummy_llm = None
            verifier = GreenbergVerifier.__new__(GreenbergVerifier)
            verifier.llm = None
            verifier._build_report(saved_verdicts, output_dir)
        else:
            print("\n  No saved verdicts found. Run without --report-only first.")
        return

    # ── Init LLM ──────────────────────────────────────────────────
    if args.openrouter:
        from llm import OpenRouterLLM
        llm = OpenRouterLLM(
            api_key          = args.api_key,
            model            = args.or_model,
            token_scale      = args.token_scale,
            max_tokens_cap   = args.max_tokens_cap,
            force_max_tokens = args.force_max_tokens,
        )
        if args.force_max_tokens:
            print(f"  LLM backend : OpenRouter ({args.or_model})  "
                  f"force_max_tokens={args.max_tokens_cap} (all calls)")
        else:
            print(f"  LLM backend : OpenRouter ({args.or_model})  "
                  f"token_scale={args.token_scale}x  cap={args.max_tokens_cap}")
    else:
        from llm import QwenLLM
        llm = QwenLLM(use_vllm=args.use_vllm)
        print(f"  LLM backend : local Qwen2.5-7B-Instruct"
              f"{' (vllm)' if args.use_vllm else ' (transformers)'}")

    # ── Init agents (grammars/IGT loaded once) ────────────────────
    agents = init_agents(language_configs, llm, args.max_iterations)
    if not agents:
        print("[ERROR] No agents could be initialized. Exiting.")
        sys.exit(1)

    # ── Run verifier ──────────────────────────────────────────────
    verifier = GreenbergVerifier(llm)
    results  = verifier.run(
        universals     = universals,
        agents         = agents,
        output_dir     = output_dir,
        max_iterations = args.max_iterations,
        skip_existing  = args.skip_existing,
    )

    # ── Final summary ─────────────────────────────────────────────
    true_count  = sum(1 for uv in results if uv.verdict == "TRUE")
    false_count = sum(1 for uv in results if uv.verdict == "FALSE")

    print(f"\n{'━'*64}")
    print(f"  ALL DONE")
    print(f"{'━'*64}")
    print(f"  Universals tested : {len(results)}")
    print(f"  ✅ TRUE            : {true_count}")
    print(f"  ❌ FALSE           : {false_count}")
    print(f"  Output            : {output_dir.resolve()}")
    print()

    # Print quick verdict list
    for uv in results:
        emoji = {"TRUE": "✅", "FALSE": "❌"}.get(uv.verdict, "❓")
        print(f"  {emoji} {uv.uid:<6} {uv.verdict:<6} "
              f"sup={uv.n_valid_support} vio={uv.n_valid_violation} "
              f"na={uv.n_antecedent_na} irr={uv.n_irrelevant} ins={uv.n_insufficient}")
    print()


if __name__ == "__main__":
    main()