"""
baseline.py — Zero-Resource Grambank Baseline
===============================================
Asks an LLM directly about each language's typological features
with NO grammar book and NO IGT data — pure parametric knowledge.

The model sees only:
  - The language name
  - The Grambank feature question
  - The coding instructions
  - A strict instruction to output only the coding value

Usage
-----
# Single language:
python baseline.py \
    --languages "Aguaruna" \
    --guidelines grambank_queries.csv \
    --output baseline_results.csv \
    --api-key sk-...

# Multiple languages:
python baseline.py \
    --languages "Aguaruna" "Chakali" "Bargam" \
    --guidelines grambank_queries.csv \
    --output baseline_results.csv \
    --api-key sk-...

# From a text file (one language per line):
python baseline.py \
    --languages-file languages.txt \
    --guidelines grambank_queries.csv \
    --output baseline_results.csv \
    --api-key sk-...

# API key from environment variable:
export OPENAI_API_KEY=sk-...
python baseline.py \
    --languages "Aguaruna" \
    --guidelines grambank_queries.csv \
    --output baseline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS  = 8
DEFAULT_RETRIES     = 3

VALID_CODES = {"0", "1", "2", "3", "?"}

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a linguistic typologist with expert knowledge of the world's languages.

## TASK
Answer the following Grambank feature question about {language}.
Apply the coding instructions carefully and output ONLY the single code \
character: 0, 1, 2, 3, or ?
No explanation. No punctuation. No extra text. Just one character.

## LANGUAGE
{language}

## FEATURE QUESTION
{query}

## CODING INSTRUCTIONS
{coding}

## FEATURE DEFINITION
{definition}

## YOUR CODE (output exactly one character — 0, 1, 2, 3, or ?):"""


# ── Qwen caller ───────────────────────────────────────────────────────────────

def load_qwen(model_name: str, load_in_4bit: bool = False):
    """Load Qwen model and tokenizer once — call this before the main loop."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading {model_name} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    logger.info("Model loaded.")
    return model, tokenizer


def call_qwen(model, tokenizer, prompt: str, temperature: float,
              retries: int) -> str:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "You are a linguistic typologist. "
                "Output ONLY a single character: 0, 1, 2, 3, or ?. "
                "Nothing else."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, retries + 1):
        try:
            text   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_MAX_TOKENS,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            for ch in raw:
                if ch in VALID_CODES:
                    return ch

            logger.warning(f"  Attempt {attempt}: unexpected output {raw!r}, retrying...")

        except Exception as exc:
            logger.warning(f"  Attempt {attempt}: error — {exc}")

    logger.warning("  All retries failed — defaulting to '?'")
    return "?"


# ── Guidelines loader ──────────────────────────────────────────────────────────

def load_guidelines(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "id":         row.get("ID", "").strip(),
                "query":      row.get("Query", "").strip(),
                "coding":     row.get("Coding", "").strip(),
                "definition": row.get("Definition", "").strip(),
            })
    logger.info(f"Loaded {len(rows)} features from {csv_path.name}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-resource baseline: ask an LLM Grambank questions "
            "about a language using only its parametric knowledge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Language input ──
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument(
        "--languages",
        nargs="+",
        metavar="LANG",
        help="One or more language names.",
    )
    lang_group.add_argument(
        "--languages-file",
        metavar="PATH",
        help="Plain-text file with one language name per line.",
    )

    parser.add_argument(
        "--guidelines",
        metavar="PATH",
        required=True,
        help="Grambank coding guidelines CSV (ID, Query, Coding, Definition).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="baseline_results.csv",
        help="Output CSV path. Default: baseline_results.csv",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization (for GPUs with <16 GB VRAM).",
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
        help=f"Retries on invalid output. Default: {DEFAULT_RETRIES}",
    )

    args = parser.parse_args()

    # ── Resolve language list ──────────────────────────────────────────────────
    if args.languages:
        languages = args.languages
    else:
        p = Path(args.languages_file)
        if not p.exists():
            logger.error(f"Languages file not found: {p}")
            sys.exit(1)
        languages = [
            l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        ]

    if not languages:
        logger.error("No languages specified.")
        sys.exit(1)

    # ── Load guidelines ────────────────────────────────────────────────────────
    guidelines_path = Path(args.guidelines)
    if not guidelines_path.exists():
        logger.error(f"Guidelines CSV not found: {guidelines_path}")
        sys.exit(1)
    guidelines = load_guidelines(guidelines_path)

    # ── Load Qwen model ────────────────────────────────────────────────────────
    model, tokenizer = load_qwen(args.model, load_in_4bit=args.load_in_4bit)

    # ── Run ────────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["language", "gb_id", "query", "baseline_code"]
    total      = len(languages) * len(guidelines)
    done       = 0

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for language in languages:
            logger.info(
                f"\n[{language}] — {len(guidelines)} features to code"
            )
            for feat in guidelines:
                prompt = PROMPT_TEMPLATE.format(
                    language   = language,
                    query      = feat["query"],
                    coding     = feat["coding"],
                    definition = feat["definition"],
                )

                t0   = time.time()
                code = call_qwen(
                    model, tokenizer, prompt, args.temperature, args.retries
                )
                done += 1
                logger.info(
                    f"  [{done}/{total}] {feat['id']}  {feat['query'][:50]}...  "
                    f"→ {code}  ({time.time()-t0:.1f}s)"
                )

                writer.writerow({
                    "language":      language,
                    "gb_id":         feat["id"],
                    "query":         feat["query"],
                    "baseline_code": code,
                })
                fh.flush()   # write row immediately so progress is saved on crash

    logger.info(f"\nDone. {done} rows written to {output_path}")


if __name__ == "__main__":
    main()
