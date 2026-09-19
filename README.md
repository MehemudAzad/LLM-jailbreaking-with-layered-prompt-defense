# Tool 27: LLM Jailbreak Battery & Layered Prompt Defense

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark: AdvBench](https://img.shields.io/badge/Benchmark-AdvBench--50-red.svg)](https://github.com/gonp/AdvBench)
[![Status: Evaluated](https://img.shields.io/badge/Evaluation-5%20Models%20%7C%20850%20Trials-brightgreen.svg)](results/README.md)

**CSE-406: Computer Security Project (2026)**  
*Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology (BUET)*

---

## 1. Executive Summary

Large Language Models (LLMs) aligned via Reinforcement Learning from Human Feedback (RLHF) refuse harmful user prompts (e.g., malware generation, dangerous chemistry, cyber exploitation). However, **jailbreak attacks** bypass these safety guardrails by exploiting systemic vulnerabilities in model training.

In this project, we designed, implemented, and empirically evaluated:
1. **The Attack Battery (`attacks/`)**: **17 distinct jailbreak attack techniques** implemented from scratch in pure Python (no external attack libraries like TextAttack or Foolbox), categorized into Injections, Encodings, Personas, Compound Combinations, and Model-Assisted variants.
2. **The 6-Layer Defense Pipeline (`defense/`)**: An end-to-end multi-tier defense architecture spanning **L0 (Pre-fill Guard)**, **L1 (Perplexity Filter)**, **L1.5 (Structural Cipher Filter)**, **L2 (Prompt Paraphraser)**, **L3 (System Hardening)**, and **L4 (Response Classifier / Judge)**.
3. **Cross-Model Benchmark**: Evaluated across **5 open-weight models** (**Qwen 2.5 7B-Instruct**, **Llama 3.1 8B-Instruct**, **Mistral 7B-Instruct**, **Gemma 2 9B-it**, and **Qwen 3 8B**) over 50 AdvBench harmful behaviors ($50 \times 17 = 850$ trials per model configuration).

### Key Scientific Findings:
- **Heuristic Non-Refusal vs. Actionable Harm**: Standard string-matching heuristics drastically overestimate jailbreak success. On Qwen 2.5 7B, while **64.9%** of raw attacks bypassed refusal keywords, only **27.5%** actually produced executable, actionable harm.
- **Layer 4 Output Gatekeeper**: By deploying Qwen 2.5 7B-Instruct as an output judge in Layer 4, **served actionable harm to the end user is reduced to ~0.0%** across all models.
- **Cheap Upstream Filtering**: Layers 0, 1, and 1.5 run in under 5 ms on CPU and intercept **33.5%–37.3%** of all attacks before inference, completely neutralizing multi-stage combinations (`combination_1/2/3`) and character-level encodings (`leetspeak`, `disemvowel`).

---

## 2. Theoretical Framework (Wei et al. 2023)

Jailbreak attacks exploit two fundamental failure modes in instruction-tuned language models:

```
                            ┌─────────────────────────────────────────────────────────────┐
                            │                 LLM Jailbreak Failure Modes                 │
                            └──────────────────────────────┬──────────────────────────────┘
                                                           │
                      ┌────────────────────────────────────┴────────────────────────────────────┐
                      ▼                                                                         ▼
       ┌──────────────────────────────┐                                          ┌──────────────────────────────┐
       │     Competing Objectives     │                                          │  Mismatched Generalization   │
       ├──────────────────────────────┤                                          ├──────────────────────────────┤
       │ Helpfulness vs. Safety clash │                                          │ RLHF safety trained on clean │
       │ Forces affirmative prefixes, │                                          │ English; pre-training saw    │
       │ JSON syntax, or multi-task   │                                          │ ciphers & character-level    │
       │ distractor lists.            │                                          │ obfuscations.                │
       └──────────────┬───────────────┘                                          └──────────────┬───────────────┘
                      │                                                                         │
                      ▼                                                                         ▼
       Defended by: L0, L2, L3, L4                                                Defended by: L1, L1.5
```

1. **Competing Objectives**: The model is trained to be both helpful and safe. Attacks like `prefix_injection` force the model to start with affirmative tokens (`"Absolutely! Here is "`), shifting token generation probabilities away from refusal.
2. **Mismatched Generalization**: Safety training covers natural English. Ciphers, Base64, and character-level substitutions (`leetspeak`, `disemvowel`) bypass safety filters because the model understands the semantic meaning from pre-training while its refusal heuristics fail to trigger.

---

## 3. Empirical Benchmark Results

### 3.1 Baseline Attack Benchmark (Without Defense)

> Evaluated on 50 AdvBench harmful goals across all 17 attack techniques ($850$ trials per model):

| Target Model | Undefended Non-Refusal Rate | Undefended Actionable ASR | Raw Passthrough ASR | Adaptive ASR (Any Attack Wins) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 2.5 7B-Instruct** | **64.9%** (552/850) | **27.5%** (234/850) | **2.0%** (1/50) | **100.0%** (50/50) |
| **Llama 3.1 8B-Instruct** | **54.7%** (465/850) | **18.0%** | **4.0%** (2/50) | **100.0%** (50/50) |
| **Mistral 7B-Instruct** | **92.5%** (786/850) | *Unmeasured* | **100.0%** (50/50) | **100.0%** (50/50) |
| **Gemma 2 9B-it** | **41.8%** (355/850) | **11.5%** | **10.0%** (5/50) | **100.0%** (50/50) |
| **Qwen 3 8B** | **85.3%** (725/850) | — | **92.0%** (46/50) | **100.0%** (50/50) |

- **Raw Passthrough ASR**: Direct harmful query baseline. Qwen 2.5 (2.0%) and Llama 3.1 (4.0%) strongly refuse raw harm, whereas Mistral 7B complies with **100.0%** of direct malicious prompts without requiring adversarial framing.
- **Adaptive ASR**: If an adversary tests all techniques, **100.0% of goals fall** against raw models.

---

### 3.2 Layered Defense Pipeline Benchmark (With Defense)

> Evaluated under the 6-layer defense pipeline across the same 850 trials per model:

| Target Model | Pre-Defenses Blocked (L0–L1.5) | Output Judge (L4) Blocked | Non-Refusal Rate (After Defense / Served) | Actionable ASR (Post-L4 Gate) | Defended Adaptive ASR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen 2.5 7B-Instruct** | **36.8%** (313/850) | **0.5%** (4/850) | **14.4%** (122/850) | **~0.0%** | **94.0%** (47/50) |
| **Llama 3.1 8B-Instruct** | **37.3%** (317/850) | **1.2%** (10/850) | **16.6%** (141/850) | **~0.0%** | **96.0%** (48/50) |
| **Mistral 7B-Instruct** | **37.2%** (316/850) | **6.9%** (59/850) | **28.2%** (240/850) | **~0.0%** | **100.0%** (50/50) |
| **Gemma 2 9B-it** | **33.5%** (285/850) | **4.8%** (41/850) | **13.4%** (114/850) | **~0.0%** | **96.0%** (48/50) |
| **Qwen 3 8B** | — | — | — | — | — |

- **Zero Actionable Harm**: Actionable dangerous instructions served to the end user are suppressed to **~0.0%** via Layer 4 output classification.
- **Pre-Defense Interception Uniformity**: Layers 0, 1, and 1.5 consistently intercept **33.5%–37.3%** of all attacks across every model architecture.

---

## 4. Per-Attack Comparison Across 5 Models

Full breakdown of Attack Success Rate (%) for **Raw (Undefended)** vs. **Def (Defended)**:

| Attack Technique | Category | Qwen 2.5 Raw | Qwen 2.5 Def | Llama 3.1 Raw | Llama 3.1 Def | Mistral Raw | Mistral Def | Gemma 2 Raw | Gemma 2 Def | Qwen 3 Raw |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `auto_obfuscation` | Assisted | 100.0% | 18.0% | 96.0% | 24.0% | 0.0% | 62.0% | 100.0% | 18.0% | 8.0% |
| `auto_payload_splitting` | Assisted | 96.0% | 36.0% | 58.0% | 30.0% | 100.0% | 46.0% | 50.0% | 10.0% | 100.0% |
| `combination_1` | Combination | 100.0% | **0.0%** | 100.0% | **0.0%** | 100.0% | **0.0%** | 60.0% | **0.0%** | 100.0% |
| `combination_2` | Combination | 100.0% | **0.0%** | 100.0% | **0.0%** | 96.0% | **0.0%** | 86.0% | **0.0%** | 100.0% |
| `combination_3` | Combination | 100.0% | **0.0%** | 100.0% | **0.0%** | 100.0% | **0.0%** | 98.0% | **0.0%** | 100.0% |
| `dev_mode` | Persona | 2.0% | 28.0% | 8.0% | 24.0% | 100.0% | 84.0% | 6.0% | 2.0% | 100.0% |
| `disemvowel` | Encoding | 92.0% | 2.0% | 92.0% | 6.0% | 98.0% | **0.0%** | 58.0% | 6.0% | 100.0% |
| `distractors` | Injection | 76.0% | 2.0% | 12.0% | 28.0% | 100.0% | 6.0% | 8.0% | 56.0% | 100.0% |
| `evil_confidant` | Persona | 4.0% | 10.0% | 36.0% | 10.0% | 80.0% | 2.0% | 42.0% | 12.0% | 100.0% |
| `leetspeak` | Encoding | 88.0% | **0.0%** | 34.0% | **0.0%** | 98.0% | **0.0%** | 18.0% | 12.0% | 100.0% |
| `passthrough` | Control | 2.0% | 8.0% | 4.0% | 10.0% | 100.0% | 40.0% | 10.0% | 2.0% | 92.0% |
| `prefix_injection` | Injection | 100.0% | 14.0% | 96.0% | 10.0% | 100.0% | 34.0% | 10.0% | 4.0% | 90.0% |
| `prefix_injection_hello` | Injection | 22.0% | 40.0% | 68.0% | 54.0% | 100.0% | 44.0% | 32.0% | 28.0% | 6.0% |
| `prefix_injection_textonly` | Injection | 100.0% | 14.0% | 78.0% | 10.0% | 100.0% | 34.0% | 28.0% | 4.0% | 82.0% |
| `refusal_suppression` | Injection | 14.0% | 34.0% | 34.0% | 46.0% | 100.0% | 58.0% | 94.0% | 50.0% | 100.0% |
| `style_injection_json` | Injection | 100.0% | 36.0% | 4.0% | 28.0% | 100.0% | 60.0% | 4.0% | 12.0% | 76.0% |
| `wikipedia_article` | Persona | 8.0% | 2.0% | 10.0% | 2.0% | 100.0% | 10.0% | 6.0% | 12.0% | 96.0% |
| **Mean ASR** | — | **64.9%** | **14.4%** | **54.7%** | **16.6%** | **92.5%** | **28.2%** | **41.8%** | **13.4%** | **85.3%** |

---

## 5. Defense Architecture (The 6-Layer Pipeline)

```
[ User Input Prompt ]
        │
        ▼
┌─────────────────────────────────┐
│   Layer 0: Pre-fill Guard       │ ──► Strips forged assistant-turn prefixes ("Absolutely! Here is")
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Layer 1: Perplexity Filter    │ ──► gpt2-large sliding window; blocks leetspeak/disemvowel (PPL > 424.8)
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Layer 1.5: Structural Filter  │ ──► Regex detection for Base64 / Hex / cipher delimiters (combination 1/2/3)
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Layer 2: Prompt Paraphraser   │ ──► Neutral LLM rewriting; strips distractors and adversarial wrappers
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Layer 3: System Hardening     │ ──► Instruction hierarchy enforcement prepended to target model
└───────────────┬─────────────────┘
                │
                ▼
      [ Target LLM Inference ]
                │
                ▼
┌─────────────────────────────────┐
│   Layer 4: Response Classifier  │ ──► Qwen 2.5 7B-Instruct evaluates output; blocks residual actionable harm
└───────────────┬─────────────────┘
                │
                ▼
        [ Safe Response ]
```

---

## 6. How to Run the Project

### 6.1 Interactive Terminal Live Demo (Zero Dependencies, Offline)
Runs the complete 5-part presentation walkthrough with zero GPU requirements:
```bash
python3 demo.py
```
*(Options: `--no-pause` to run continuously, or `--part {1,2,3,4,5}` to jump to specific sections.)*

### 6.2 Offline Dry Run (Harness Verification)
Tests the end-to-end evaluation harness locally using fake mock handles without downloading weights:
```bash
python run_eval.py --dry-run --limit 1
```

### 6.3 Real Evaluation Harness
To execute benchmark evaluations against local or remote HuggingFace models:
```bash
# Baseline attack evaluation (Undefended)
python run_eval.py --attack all --defense off --limit 50

# Full 6-layer defense pipeline evaluation
python run_eval.py --attack all --defense on --limit 50
```

### 6.4 Kaggle Execution Notebooks
All production runs were executed on Kaggle GPU instances (T4 / T4×2). Execution notebooks reside in `notebooks/`:
- `notebooks/qwen/qwen2-5-7b-it-jailbreak-attacks-notebook.ipynb`: Qwen 2.5 7B Undefended
- `notebooks/qwen/qwen2.5-defense-notebook.ipynb`: Qwen 2.5 7B Defended
- `notebooks/qwen/judge_actionable_asr.ipynb`: Qwen 2.5 7B Actionable ASR Judge
- `notebooks/llama/llm-jailbreaking-llama3-1-attack-notebook.ipynb`: Llama 3.1 8B Undefended
- `notebooks/llama/llama3-1-8b-defense-notebook.ipynb`: Llama 3.1 8B Defended
- `notebooks/mistral/mistral-llm-jailbreak-attacks-notebook.ipynb`: Mistral 7B Undefended
- `notebooks/mistral/mistral-defense-notebook.ipynb`: Mistral 7B Defended
- `notebooks/gemma/gemma2-defense-notebook.ipynb`: Gemma 2 9B Defended
- `notebooks/gemma/gemma-2-9b-it-attack-notebook.ipynb`: Gemma 2 9B Undefended
- `notebooks/qwen/qwen3-jailbreak-attack-notebook.ipynb`: Qwen 3 8B Undefended

---

## 7. Project Structure & File Roles

```
project-406/
├── demo.py                     # Turnkey interactive presentation / viva demo script
├── run_eval.py                 # Primary CLI harness for attack and defense evaluations
├── regrade.py                  # Offline evaluation scoring and heuristic/judge regrader
├── report.py                   # Terminal and LaTeX table generation utilities
├── benign_eval.py              # Benign dataset false positive rate (FPR) evaluator
├── conftest.py                 # Pytest root configuration
├── config.toml                 # Pinned models, random seeds (1337), thresholds, and dataset paths
├── core/
│   ├── config.py               # TOML configuration parser
│   ├── models.py               # HuggingFace & Fake model backend wrappers
│   ├── datasets.py             # AdvBench harmful & benign prompt loaders
│   ├── transcript.py           # Append-only JSONL logging engine
│   └── seed.py                 # Deterministic seed controller
├── attacks/                    # 17 pure Python jailbreak attack implementations
│   ├── base.py                 # Base AttackTechnique class and registry
│   ├── passthrough.py          # Negative control baseline
│   ├── prefix_injection.py     # Forged assistant turn affirmative continuation
│   ├── refusal_suppression.py  # Output vocabulary constraint attack
│   ├── style_injection_json.py # Structured JSON schema enforcement
│   ├── distractors.py          # Multi-task sandwich injection
│   ├── leetspeak.py            # Alphanumeric character substitution
│   ├── disemvowel.py           # Vowel-deletion consonant skeleton attack
│   ├── dev_mode.py             # Developer mode diagnostic persona
│   ├── evil_confidant.py       # Fictional narrator roleplay
│   ├── wikipedia_article.py    # Neutral encyclopedic laundering
│   ├── combination.py          # Compound stacked attacks (combination 1, 2, 3)
│   ├── auto_obfuscation.py     # Algorithmic synonym/metaphor rewriting
│   └── auto_payload_splitting.py # Token splitting and variable reassembly
├── defense/                    # 6-layer defense pipeline modules
│   ├── pipeline.py             # Pipeline orchestrator (L0-L3 pre-target -> target -> L4 post-target)
│   ├── layer0_prefill_guard.py # Forged assistant turn stripper
│   ├── layer1_perplexity_filter.py # GPT-2 sliding-window perplexity filter (threshold: 424.8)
│   ├── layer1_5_structural.py  # Regex Base64 / Hex / delimiter cipher filter
│   ├── layer2_paraphrase.py    # Independent LLM prompt rewriting filter
│   ├── layer3_system_hardening.py # Instruction-hierarchy system prompt hardening
│   └── layer4_response_classifier.py # Qwen 2.5 7B output judge response classifier
├── datasets/
│   ├── harmful_behaviors.jsonl # Frozen 50 AdvBench evaluation behaviors
│   └── benign_prompts.jsonl    # 50 benign imperative control prompts
├── results/
│   └── README.md               # Complete cross-model comparative report & analysis
├── docs/
│   ├── ATTACKS-EXPLAINED.md    # Deep technical guide for all 17 attacks & viva prep
│   ├── DEFENSE-EXPLAINED.md    # Deep technical guide for the 6-layer defense & viva prep
│   ├── report/                 # Final Report (LaTeX source, PDF) & PRESENTATION.md
│   │   ├── final-report.tex / pdf # Formal Course Final Report
│   │   └── PRESENTATION.md     # 5-step viva voce presentation walkthrough script
│   └── scope/                  # Design Report (LaTeX source, PDF) & Course Spec PDF
│       ├── design-report.tex / pdf # Formal Course Design Report
│       └── CSE406ProjectJan2026.pdf # Official course specification PDF
└── tests/
    └── test_wiring.py          # Pipeline smoke and regression tests
```

---

## 8. Academic References

1. **Wei et al. (2023)**: *Jailbroken: How Does LLM Safety Training Fail?* Advances in Neural Information Processing Systems (NeurIPS 2023).
2. **Jain et al. (2023)**: *Baseline Defenses for Adversarial Attacks Against Aligned Language Models.* arXiv:2309.00614.
3. **Alon & Kamfonas (2023)**: *Detecting Language Model Attacks with Perplexity.* arXiv:2308.14132.
4. **Zou et al. (2023)**: *Universal and Transferable Adversarial Attacks on Aligned Language Models.* (AdvBench benchmark).

---

## 9. Authors & Team Contributions

- **Mehemud Azad (2105014)**:
  - Curated and formatted the core evaluation dataset (50 diverse harmful behaviors from AdvBench across all risk categories).
  - Researched, conceptualized, and categorized the attack strategies and taxonomy across the failure modes.
  - Executed and managed the multi-model empirical evaluation runs across open-weight LLMs (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B, Gemma 2 9B, Qwen 3 8B).
  - Built the cross-model evaluation benchmarks, actionable harm analysis, and comparative performance metrics.

- **Khalid Hasan Tuhin (2105002)**:
  - Architected and implemented the multi-stage layered defense pipeline (`defense/*` L0–L4 and bonus layers).
  - Implemented the defense filter mechanics (pattern matching, perplexity evaluation, semantic intent classification, prefix defenses, and output judge gatekeeper).
  - Built the baseline evaluation harness and evaluation tooling.
