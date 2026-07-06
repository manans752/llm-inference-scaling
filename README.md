**Built a local LLM benchmarking pipeline that compares inference-time scaling strategies across prototype and benchmark reasoning datasets, tracking accuracy, token cost, latency, and failure modes to evaluate efficient small-model deployment.**

**Built a local LLM benchmarking pipeline that compares inference-time scaling strategies across prototype and benchmark reasoning datasets, tracking accuracy, token cost, latency, and failure modes to evaluate efficient small-model deployment.**
# Inference-Time Scaling for Local LLM Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/PyTorch-Local%20Inference-red?style=for-the-badge" alt="PyTorch Local Inference" />
  <img src="https://img.shields.io/badge/Transformers-Qwen%202.5%200.5B-gold?style=for-the-badge" alt="Transformers Qwen 2.5 0.5B" />
  <img src="https://img.shields.io/badge/Benchmark-GSM8K-success?style=for-the-badge" alt="Benchmark GSM8K" />
</p>

<p align="center">
  A reproducible research pipeline for testing whether extra inference-time compute actually helps small, local language models reason better.
</p>

<p align="center">
  <a href="#why-this-project-matters">Why It Matters</a> •
  <a href="#headline-results">Results</a> •
  <a href="#visuals">Visuals</a> •
  <a href="#run-it">Run It</a> •
  <a href="#research-context">Research Context</a>
</p>

---

## Why This Project Matters

Most LLM projects stop at a demo. This repo goes a step further: it asks a real research question and builds the infrastructure to answer it carefully.

**Core question:** can a very small, locally runnable LLM become more useful on reasoning tasks if we spend more compute at inference time?

This project evaluates that question across two phases:

- `v1`: a prototype benchmark with hand-authored arithmetic, logic, and word-problem tasks
- `v2`: a stronger GSM8K-based evaluation using `Qwen/Qwen2.5-0.5B-Instruct`

The result is a full benchmarking pipeline that tracks:

- answer accuracy
- token usage
- latency
- invalid outputs
- parsing failures
- strategy-level efficiency

That makes the repo useful both as a research artifact and as a portfolio project about trustworthy, privacy-preserving LLM evaluation.

## What This Repo Demonstrates

- End-to-end experiment orchestration for local LLM evaluation
- Multiple inference-time scaling strategies: baseline, adaptive escalation, budgeted prompting, and self-consistency
- Final-answer-only prompting with robust answer extraction
- Row-level result logging plus summary metrics for analysis notebooks and reports
- A realistic negative result: more inference-time compute is not automatically better

## Headline Results

The main benchmark run is `results/v2_qwen_gsm8k_50`, a 50-question GSM8K subset evaluated locally.

| Strategy | Accuracy | Avg Tokens | Avg Latency | Accuracy / 1k Tokens |
|---|---:|---:|---:|---:|
| Adaptive | `0.06` | `109.72` | `0.78s` | `0.0109` |
| Baseline | `0.06` | `181.72` | `3.06s` | `0.0066` |
| Chain-of-thought variants | `0.05` | `244.55` | `4.65s` | `0.00068` |
| Self-consistency | `0.05` | `770.88` | `13.41s` | `0.00065` |

**Takeaway:** extra inference-time compute did not meaningfully improve GSM8K accuracy for a 0.5B local model, but the pipeline made that limitation measurable and reproducible.
[V2 Strategy Overview](/analysis/readme_v2_strategy_overview.png)

```mermaid
flowchart LR
    A["Benchmark Dataset"] --> B["Prompt Strategy"]
    B --> C["Model Generation"]
    C --> D["Answer Extraction"]
    D --> E["Scoring + Error Typing"]
    E --> F["CSV / JSON Results"]
    F --> G["Notebook + Report"]
```

## Why The Findings Are Interesting

- The prototype phase looked promising, especially on easier hand-authored tasks.
- The benchmark phase exposed the real limits of a tiny model on harder multi-step arithmetic.
- Adaptive decoding matched baseline accuracy while using fewer tokens and much less latency.
- Expensive strategies like self-consistency produced worse efficiency without improving correctness.

## Research Context

This project is inspired by the broader inference-time scaling literature, but it focuses on a more constrained and practical setting: a small open-weight model, local execution, fixed evaluation artifacts, and explicit cost tracking.

Two especially relevant references:

- [OpenAI, *Trading Inference-Time Compute for Adversarial Robustness*](https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness/) explores cases where giving reasoning models more inference-time compute can improve robustness against attacks.
- [Balachandran et al., *Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead*](https://arxiv.org/abs/2504.00294) studies benefits and limits of inference-time scaling across models and tasks, including the important point that more tokens do not automatically mean better performance.

This repo complements that line of work by asking a narrower but very practical question:

> What happens when we apply inference-time scaling ideas to a tiny, locally deployable model under free and privacy-preserving constraints?
[Error Type Distribution on GSM8K-50](/analysis/readme_budget_sweep_gsm8k50.png)
## Run It

### 1. Install dependencies

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Run a quick prototype check

```bash
./venv/bin/python run_experiment.py \
  --dataset-path data/tasks.json \
  --dataset-name prototype_v1 \
  --max-tasks 5
```

### 3. Run the main local benchmark path

```bash
./venv/bin/python run_experiment.py \
  --backend local_transformers \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-path data/benchmarks/gsm8k_test_50.json \
  --dataset-name gsm8k_test_50 \
  --experiment-name v2_qwen_gsm8k_50 \
  --strategies baseline chain_of_thought adaptive self_consistency
```

### 4. Run tests

```bash
./venv/bin/python -m unittest discover -s tests
```

## Project Layout

- [run_experiment.py](run_experiment.py): experiment entrypoint
- [inference/experiments.py](inference/experiments.py): strategy runners
- [inference/common/evaluation.py](inference/common/evaluation.py): parsing, normalization, and scoring
- [inference/common/datasets.py](inference/common/datasets.py): dataset loading and benchmark prep
- [reports/research_report.md](reports/research_report.md): report draft
- [evaluation/research_report_analysis.ipynb](evaluation/research_report_analysis.ipynb): analysis notebook

--------------------------------------------------------------------------------------------------------------------------------------------
This project explores inference-time scaling strategies and reasoning control mechanisms for LLMs.
Open-source LMs are smaller, easier to train, and, importantly, don't involve your sensitive data being sent out to a closed source entity.
Consequently, these smaller models don't have the immediate performance of larger language models.
Inference-time scaling is often the most accessible, cost-effective method to improve a langauge model's reasoning and robustness.

This project is a controlled experimental framework for studying how inference-time compute affects reasoning quality

Key Research Questions: 
- How does the quality of reasoning scale with increased inference-time computation?

References/Inspiration:
[Trading Inference-Time Compute for Adversarial Robustness, Open AI](https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness/)
[Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead
](https://doi.org/10.48550/arXiv.2504.00294)
[A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods
](https://doi.org/10.48550/arXiv.2502.01618)
