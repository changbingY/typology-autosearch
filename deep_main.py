# deep_main.py
from deep_agent import run_deep_pipeline, run_query_pipeline, LanguageConfig
import argparse, json, sys

parser = argparse.ArgumentParser()
parser.add_argument("--language",   required=True)
parser.add_argument(
    "--grammar",
    default=None,
    help="Path to grammar JSON. Required unless --igt-only is set.",
)
parser.add_argument(
    "--igt",
    default=None,
    help="Path to IGT JSON (optional alongside grammar; required for --igt-only).",
)
parser.add_argument("--output",     default="output")
parser.add_argument("--use-vllm",   action="store_true")
parser.add_argument("--max-iterations", type=int, default=15)
parser.add_argument("--confidence-threshold", type=float, default=0.75)

# ── Abbreviations ─────────────────────────────────────────────────────────────
parser.add_argument(
    "--abbreviations",
    default=None,
    metavar="PATH",
    help=(
        "Path to a tab-separated glossing abbreviations file "
        "(e.g. abbreviations.txt). "
        "Format: ABBREV<TAB>Full meaning, one entry per line. "
        "When supplied, all IGT tag references in tool output and LLM prompts "
        "are annotated with their human-readable meanings, improving reasoning "
        "accuracy over unfamiliar tags."
    ),
)

# ── Mode flags ────────────────────────────────────────────────────────────────
parser.add_argument(
    "--igt-only",
    action="store_true",
    help=(
        "Run in IGT-only mode: infer typological features purely from IGT data "
        "without a reference grammar.  --grammar is ignored; --igt is required."
    ),
)

# ── Free-form query mode ──────────────────────────────────────────────────────
parser.add_argument(
    "--query",
    action="append",
    dest="queries",
    default=None,
    metavar="QUERY",
    help=(
        "Free-form query to deep-search the grammar or IGT corpus "
        "(can be repeated for multiple queries). "
        "If provided, skips the full typological pipeline. "
        "In --igt-only mode, queries are answered from IGT data alone."
    ),
)

args = parser.parse_args()

# ── Validate argument combinations ───────────────────────────────────────────

if args.igt_only:
    # IGT-only mode: grammar is not needed; IGT is required
    if not args.igt:
        parser.error("--igt-only requires --igt <path_to_igt.json>")
else:
    # Normal / query mode: grammar is required
    if not args.grammar:
        parser.error(
            "--grammar is required unless --igt-only is set. "
            "If you only have IGT data, use --igt-only --igt <path>."
        )

# ── Dispatch ─────────────────────────────────────────────────────────────────

if args.igt_only:
    # ── IGT-only mode ─────────────────────────────────────────────
    # Bottom-up discovery from IGT data, no grammar book required.
    from igt_agent import run_igt_pipeline, run_igt_query_pipeline

    if args.queries:
        # Free-form queries answered from IGT alone
        run_igt_query_pipeline(
            language=args.language,
            igt_path=args.igt,
            queries=args.queries,
            output_dir=args.output,
            use_vllm=args.use_vllm,
            max_iterations=args.max_iterations,
            abbreviations_path=args.abbreviations,   # ← NEW
        )
    else:
        # Full typological pipeline from IGT alone
        run_igt_pipeline(
            language=args.language,
            igt_path=args.igt,
            output_dir=args.output,
            use_vllm=args.use_vllm,
            max_iterations_per_feature=args.max_iterations,
            confidence_threshold=args.confidence_threshold,
            abbreviations_path=args.abbreviations,   # ← NEW
        )

elif args.queries:
    # ── Query mode (grammar + optional IGT) ───────────────────────
    # One or more free-form questions answered via grammar search.
    run_query_pipeline(
        language=args.language,
        grammar_path=args.grammar,
        igt_path=args.igt,
        queries=args.queries,
        output_dir=args.output,
        use_vllm=args.use_vllm,
        max_iterations=args.max_iterations,
        abbreviations_path=args.abbreviations,       # ← NEW
    )

else:
    # ── Full pipeline mode (grammar + optional IGT) ────────────────
    # Automatic domain discovery → feature planning → ReAct search
    # for all typological features.
    run_deep_pipeline(
        language_configs=[LanguageConfig(args.language, args.grammar, args.igt)],
        output_dir=args.output,
        use_vllm=args.use_vllm,
        max_iterations_per_feature=args.max_iterations,
        confidence_threshold=args.confidence_threshold,
        abbreviations_path=args.abbreviations,       # ← NEW
    )
