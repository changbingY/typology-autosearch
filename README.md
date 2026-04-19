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

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `deep_main.py` | Main entry point for the pipeline. |
| `deep_agent.py` | Core logic for the Grammar + IGT Research Agent. |
| `igt_agent.py` | Logic for the Corpus-only (Bottom-up) Agent. |
| `deep_tools.py` | Advanced toolkit (DeepGrammarToolkit) for LLM tool-use. |
| `igt_analysis.py` | Quantitative statistical engine for IGT data. |
| `evidence_graph.py` | Data structure for tracking claims and contradictions. |
| `abbreviations.py` | Registry for glossing abbreviation expansion. |

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9+
* OpenAI or Qwen-compatible API Key.

### Installation
```bash
git clone [https://github.com/yourusername/DeepLanguageResearchAgent.git](https://github.com/yourusername/DeepLanguageResearchAgent.git)
cd DeepLanguageResearchAgent
pip install -r requirements.txt
```

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

### 2. IGT-Only Mode
If no reference grammar exists, run a bottom-up discovery pipeline to infer the language's typological profile from raw IGT data:

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

## 🎓 Research Background

This project was developed within the context of **Computational Linguistics** research at the **University of British Columbia (UBC)**. It draws inspiration from the award-winning work: 
> *"LingGym: How Far Are LLMs from Thinking Like Field Linguists?"* (EMNLP 2025).

**Author**: Changbing Yang

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
