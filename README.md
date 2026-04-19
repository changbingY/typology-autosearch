# DeepLanguageResearchAgent

**DeepLanguageResearchAgent** is a sophisticated autonomous research system designed to automate the discovery, verification, and analysis of typological linguistic features. 

Unlike standard RAG (Retrieval-Augmented Generation) systems, this agent emulates the workflow of a field linguist by synthesizing qualitative descriptions from reference grammars with quantitative evidence from Interlinear Glossed Text (IGT) corpora.

---

## 🌟 Key Features

* **Dual-Engine Analysis**: Supports both a **Deep Agent** (Grammar + IGT integration) and an **IGT-Only Agent** (bottom-up discovery from raw corpora).
* **Evidence Graph & Conflict Resolution**: Maintains a structured graph of "Claims". It automatically detects contradictions between authorial descriptions and empirical data, forcing the LLM to resolve discrepancies.
* **Quantitative Grounding**: Calculates tag frequencies, positional profiles (e.g., preverbal vs. postverbal), and co-occurrence patterns to provide statistical weight to linguistic claims.
* **Abbreviation Registry**: Injects human-readable meanings into LLM prompts (e.g., `PST` → `past`), significantly improving reasoning accuracy for low-resource languages.
* **Multi-Hop Reasoning**: Capable of following cross-references across grammar chapters to synthesize complex features like TMA (Tense, Aspect, Mood) systems.

---
## 🚀 Usage

### 1. Deep Research Mode (Grammar + IGT)
To answer specific linguistic queries using both the reference grammar and the corpus:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --igt data/choguita_raramuri_igt.json \
    --abbreviations data/abbreviations.txt \
    --queries "Describe the future tense marking and its interaction with aspect."
```

### 2. Deep Research Mode (Grammar)
To answer specific linguistic queries using only the reference grammar:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --grammar data/choguita_raramuri_grammar.json \
    --queries "Describe the future tense marking and its interaction with aspect."
```

### 2. IGT-Only Mode
To run a bottom-up discovery pipeline to infer the language's typological profile from raw IGT data:

```bash
python deep_main.py \
    --language "Choguita Rarámuri" \
    --igt data/choguita_raramuri_igt.json \
    --igt-only
```

---

## 🧠 Methodology: The Evidence Graph

The system doesn't just "summarize" text. It builds an **Evidence Graph** where each claim is typed:
* **GRAMMAR_STATEMENT**: Explicit author claims from prose.
* **IGT_PATTERN**: Derived from quantitative IGT analysis.
* **ABSENCE_EVIDENCE**: Confirmed absence of features from the corpus.
* **INFERENCE**: Derived by combining multiple data points.

When a `GRAMMAR_STATEMENT` (e.g., "Language X has no gender") is contradicted by an `IGT_PATTERN` (e.g., "Found 50 instances of FEM/MASC tags"), the agent triggers a **Contradiction Resolution** step to evaluate the source of the error.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
