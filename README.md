# AutoTypologist

**AutoTypologist** is an LLM agent for automated, evidence-grounded typological research using reference grammars and Interlinear Glossed Text (IGT).

Rather than treating typological analysis as simple retrieval, the agent follows a ReAct-style research workflow: it plans what evidence is needed, retrieves relevant grammar sections and IGT examples, analyzes the evidence with linguistically motivated tools, maintains a structured evidence graph, and produces an auditable typological judgment.

The framework supports two main tasks:

- **Typological Feature Coding** — predict typological feature values following coding schemes such as Grambank.
- **Typological Hypothesis Testing** — evaluate cross-linguistic hypotheses by gathering supporting evidence and counterexamples across languages.

> This repository currently provides the implementation based on **Qwen2.5-7B-Instruct**.

---

## 🌟 Key Features

- **Grammar + IGT Analysis**: Combines qualitative descriptions from reference grammars with quantitative evidence from IGT.
- **IGT-Only Analysis**: Supports bottom-up typological analysis when no reference grammar is available.
- **Structured Evidence Graph**: Stores supporting evidence, counter-evidence, uncertainty, contradictions, citations, and confidence scores.
- **ReAct-style Investigation**: Iteratively selects linguistic tools based on the current evidence and remaining gaps.
- **Contradiction Resolution**: Explicitly checks conflicting evidence before producing a conclusion.
- **Adversarial Auditor**: Re-examines the final conclusion and can uphold, weaken, or overturn it.
- **Quantitative IGT Analysis**: Supports tag frequencies, positional distributions, construction patterns, co-occurrence analysis, and absence checks.
- **Traceable Outputs**: Saves the evidence, reasoning trace, confidence, audit result, and token usage for each analysis.

---

## 🛠️ Tool Set

For grammar-grounded analysis, the agent can select from the following tools:

| Tool | Type | Purpose |
|---|---|---|
| `read_full_section` | Grammar | Read a relevant grammar section |
| `follow_cross_references` | Grammar | Follow references to related sections |
| `extract_author_claims` | Grammar | Extract explicit analytical claims from grammar prose |
| `search_text` | Grammar | Hybrid retrieval over grammar chunks |
| `analyse_tag` | IGT | Analyze frequency, position, and co-occurrence of a gloss tag |
| `analyse_construction` | IGT | Search for ordered gloss-tag sequences |
| `analyse_absence` | IGT | Quantify evidence for the absence of a category |
| `compare_tags` | IGT | Compare the distributions of two tags |
| `get_section_igt` | IGT | Retrieve IGT examples from relevant sections |
| `search_translations` | IGT | Search translation lines for semantic evidence |
| `get_triline_examples` | IGT | Retrieve aligned morpheme / gloss / translation examples |
| `conclude` | Control | Produce a conclusion once the required evidence is collected |

The IGT-only agent additionally provides exploratory tools for tag inventories, construction discovery, morpheme position, surface forms, semantic context, and clause structure analysis.

---

## 🚀 Usage

### 1. Automatic Typological Discovery Pipeline (Grammar + IGT)

Automatically discovers typological domains, generates feature questions, and investigates each feature:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --igt data/choguita_raramuri_igt.json \
    --abbreviations data/abbreviations.txt \
    --output results/
```

### 2. Query Mode (Grammar + IGT)

Answer specific typological questions using both grammar and IGT evidence:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --igt data/choguita_raramuri_igt.json \
    --abbreviations data/abbreviations.txt \
    --query "Describe future tense marking and its interaction with aspect." \
    --query "Does the language have switch-reference morphology?"
```

### 3. Query Mode (Grammar Only)

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --query "What is the basic word order?"
```

### 4. IGT-Only Pipeline

Infer typological features directly from IGT without access to a reference grammar:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --igt data/choguita_raramuri_igt.json \
    --igt-only \
    --abbreviations data/abbreviations.txt
```

### 5. IGT-Only Query Mode

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --igt data/choguita_raramuri_igt.json \
    --igt-only \
    --abbreviations data/abbreviations.txt \
    --query "What morphological markers appear in negative clauses?"
```

### 6. Typological Hypothesis Testing (Greenberg Universals)

The cross-linguistic hypothesis-testing code is located in `multilingual-greenberg/`. Before running it, update `languages.json` so that each language points to the correct local grammar, IGT, and abbreviation files.

Test a single Greenberg universal:

```bash
cd multilingual-greenberg

python greenberg_main.py \
    --config languages.json \
    --greenberg greenberg_universals.csv \
    --ids U1 \
    --output results/greenberg/ \
    --use-vllm
```

Test multiple universals:

```bash
python greenberg_main.py \
    --config languages.json \
    --greenberg greenberg_universals.csv \
    --ids U1 U8 U14 U18 \
    --output results/greenberg/ \
    --use-vllm
```

Test all universals in the file:

```bash
python greenberg_main.py \
    --config languages.json \
    --greenberg greenberg_universals.csv \
    --all \
    --output results/greenberg/ \
    --use-vllm
```

The adversarial hypotheses used in the paper can be tested in the same way by replacing the CSV file:

```bash
python greenberg_main.py \
    --config languages.json \
    --greenberg greenberg_adversarial_rules_selected.csv \
    --all \
    --output results/greenberg_adversarial/ \
    --use-vllm
```
### Optional Flags

| Flag | Default | Description |
|---|---|---|
| `--output` | `output/` | Directory for JSON results |
| `--abbreviations` | *(none)* | Path to gloss abbreviation file |
| `--max-iterations` | `15` | Maximum ReAct iterations per feature/query |
| `--confidence-threshold` | `0.75` | Confidence threshold for feature conclusions |
| `--use-vllm` | off | Use vLLM for faster local inference |

For reproducing the paper experiments, we use a maximum of **10 ReAct iterations**, a confidence threshold of **0.75**, temperature **0.1**, and bfloat16 inference.

---

## 📖 Paper

**LLM Agents as Computational Typologists**  
Changbing Yang, Christopher Hammerly, Freda Shi, and Jian Zhu

Paper link and citation information to come.
---

## ⚠️ Note

AutoTypologist is intended to **support linguistic research**. Agent outputs should be treated as evidence-grounded research assistance and should still be validated by linguistic experts.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
