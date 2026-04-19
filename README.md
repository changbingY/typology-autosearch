# DeepLanguageResearchAgent

**DeepLanguageResearchAgent** is an autonomous research system designed to automate the discovery, verification, and structured analysis of typological linguistic features.

Unlike standard RAG (Retrieval-Augmented Generation) systems, this agent emulates the workflow of a field linguist by synthesizing qualitative descriptions from reference grammars with quantitative evidence from Interlinear Glossed Text (IGT) corpora. Each conclusion is typed, weighted, and audited — not merely retrieved.

---

## 🌟 Key Features

- **Dual-Engine Analysis**: Supports a **Deep Agent** (Grammar + optional IGT) and an **IGT-Only Agent** (bottom-up discovery from raw corpora without a reference grammar).
- **Structured Evidence Graph**: Every claim is typed (`GRAMMAR_STATEMENT`, `IGT_PATTERN`, `ABSENCE_EVIDENCE`, `INFERENCE`, `COUNTER_EVIDENCE`, `AUTHOR_CAVEAT`) and stored in a graph with automatic contradiction detection. Confidence is aggregated via log-odds weighting, with grammar prose outweighing raw IGT counts.
- **Contradiction Resolution**: When a `GRAMMAR_STATEMENT` conflicts with an `IGT_PATTERN` (e.g., "no gender marking" vs. 50 attested FEM/MASC tags), the agent triggers an explicit resolution step before concluding.
- **Adversarial Auditor**: After each conclusion is drafted, a separate auditor LLM call attempts to disprove it — downgrading `Yes` to `Partial` or flagging overconfident claims.
- **Quantitative IGT Grounding**: Computes tag frequencies, linear positional profiles (preverbal / postverbal / flexible), co-occurrence PMI, construction bigrams/trigrams, and confirmed absence scores.
- **Abbreviation Registry**: Injects human-readable gloss meanings into every LLM prompt (e.g., `PST` → `PST (past)`), improving reasoning accuracy for unfamiliar or low-resource language tags.
- **Multi-Hop Cross-Reference Following**: Automatically follows `§`/`see`/`cf.` references across grammar chapters to gather distributed evidence for complex features such as TMA (Tense-Aspect-Mood) systems.
- **Cross-Domain Deduplication**: After domain discovery, near-duplicate feature questions (Jaccard similarity ≥ 0.55 on content words) are detected and removed before investigation begins, preventing redundant LLM calls.
- **Token Usage Tracking**: Records input tokens, output tokens, and LLM call count per feature or query, saved alongside each result for cost and complexity analysis.

---

## 🛠️ Tool Set

The agent selects from 12 tools per iteration:

| Tool | Type | Purpose |
|---|---|---|
| `read_full_section` | Grammar | Complete text of a named section |
| `follow_cross_references` | Grammar | Section + all linked sections (multi-hop) |
| `extract_author_claims` | Grammar | Author's analytical statements only, no examples |
| `search_text` | Grammar | BM25 + dense hybrid search across all chunks |
| `analyse_tag` | IGT | Frequency, position, co-occurrence profile for a gloss tag |
| `analyse_construction` | IGT | Find ordered tag sequences (e.g. `[NEG, V, PST]`) |
| `analyse_absence` | IGT | Quantified negative evidence for a typological category |
| `compare_tags` | IGT | PMI-based complementary distribution test |
| `get_section_igt` | IGT | All examples from a matching section |
| `search_translations` | IGT | Keyword search across translation lines |
| `get_triline_examples` | IGT | Aligned morpheme / gloss / translation trilines |
| `conclude` | Control | Triggered only when all evidence constraints are met |

---

## 🚀 Usage

### 1. Full Typological Pipeline (Grammar + IGT)

Automatically discovers domains, generates feature questions, and investigates each one:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --igt data/choguita_raramuri_igt.json \
    --abbreviations data/abbreviations.txt \
    --output results/
```

### 2. Query Mode (Grammar + IGT)

Answer specific research questions using both the grammar and the corpus:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --igt data/choguita_raramuri_igt.json \
    --abbreviations data/abbreviations.txt \
    --queries "Describe future tense marking and its interaction with aspect." \
    --queries "Does the language have switch-reference morphology?"
```

### 3. Query Mode (Grammar only)

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --queries "What is the basic word order?"
```

### 4. IGT-Only Pipeline

Infer the language's typological profile bottom-up from raw IGT data, without a reference grammar:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --igt data/choguita_raramuri_igt.json \
    --igt-only \
    --abbreviations data/abbreviations.txt
```

### 5. IGT-Only Query Mode

Answer specific questions from IGT data alone:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --igt data/choguita_raramuri_igt.json \
    --igt-only \
    --queries "What morphological markers appear in negative clauses?"
```

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--output` | `output/` | Directory for JSON results |
| `--abbreviations` | *(none)* | Path to tab-separated gloss abbreviation file |
| `--max-iterations` | `15` | ReAct loop budget per feature / query |
| `--confidence-threshold` | `0.75` | Minimum confidence to classify a feature as confirmed |
| `--use-vllm` | off | Use vLLM backend instead of Transformers (faster on GPU) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
