# Inference-Time Scaling for Small Local LLM Reasoning

## Abstract

This project studies whether inference-time scaling improves the reasoning performance of a small, locally runnable language model. The work is organized into two phases. In `v1`, a prototype pipeline evaluates local LLM outputs on a small hand-authored arithmetic and logic benchmark. In `v2`, the same evaluation framework is applied to a stronger GSM8K-derived arithmetic benchmark using `Qwen/Qwen2.5-0.5B-Instruct` through the local Hugging Face `transformers` backend.

The main result is intentionally modest: extra inference-time compute did not reliably improve accuracy on the harder GSM8K subset. On the 50-question GSM8K run, baseline and adaptive decoding reached `0.06` accuracy, while chain-of-thought-style budget sweeps and self-consistency were near `0.05` accuracy with substantially higher token cost. The strongest contribution is therefore not raw accuracy, but a reproducible evaluation framework for comparing small-model reasoning strategies under local, free, and privacy-preserving constraints.

## 1. Research Question

The central question is:

**Can inference-time scaling strategies improve the accuracy or efficiency of a small locally deployable LLM on arithmetic reasoning tasks?**

The motivating hypotheses were:

- Increasing generation budget may improve answer accuracy, but with diminishing returns.
- Adaptive escalation may improve efficiency by spending more tokens only when the initial answer appears unreliable.
- Self-consistency may improve robustness if sampled answers are diverse and answer extraction is reliable.
- A standard benchmark will expose limitations hidden by a small hand-authored prototype dataset.

## 2. Project Phases

### v1: Prototype Evaluation

`v1` uses `data/tasks.json` and `data/benchmarks/prototype_v1.json`, a 50-question hand-authored benchmark covering arithmetic, logic, pattern reasoning, and word problems. This phase established the core pipeline: prompt construction, local generation, answer extraction, scoring, result persistence, and strategy comparison.

The `v1` results should be interpreted as a controlled prototype, not a benchmark-grade claim. They are useful because they reveal the early behavior of the pipeline and show why prompt format, parsing, and token budget matter.

### v2: Benchmark Evaluation

`v2` upgrades the evaluation to GSM8K-derived benchmark subsets in `data/benchmarks/gsm8k_test_10.json` and `data/benchmarks/gsm8k_test_50.json`. The main reported run is `results/v2_qwen_gsm8k_50`.

The final `v2` decision was to use a free local evaluation path rather than a paid or unreliable hosted endpoint. Earlier attempts to use Hugging Face routed inference failed because several models were not available through the enabled providers. Dedicated Hugging Face endpoints were rejected as the default path because they require paid infrastructure. The final setup uses local `transformers` inference with `Qwen/Qwen2.5-0.5B-Instruct`, preserving the efficient, open-weight, locally deployable story.

## 3. System Design

The evaluation pipeline follows a shared path across all strategies:

`Dataset -> Prompt Strategy -> Model Generation -> Answer Extraction -> Scoring -> Summary Analysis`

Each row-level result stores:

- `id`, `question`, `answer`, `category`, and `source`
- `strategy`, `raw_output`, `parsed_answer`, `correct`, and `error_type`
- `prompt_tokens`, `generated_tokens`, `total_tokens`, and `latency_seconds`

This shared schema is important because it makes strategy comparisons reproducible. Baseline, adaptive, budgeted prompting, and self-consistency all flow through the same parser and scorer.

## 4. Key Design Decisions

The report should explicitly include these decisions from the development process:

- `v1` is the prototype phase; `v2` is the benchmark phase.
- `v2` uses GSM8K subsets to improve external validity.
- The final free path uses local `transformers` inference rather than a paid dedicated endpoint.
- `Qwen/Qwen2.5-0.5B-Instruct` is used because it is small, open-weight, and locally runnable.
- Strategy names are preserved for comparability, but prompts now request final-answer-only outputs rather than visible reasoning.
- The answer extractor now prefers the first clean non-empty line if no `Final Answer:` marker is present, fixing cases where a model outputs an answer and then begins explaining.
- Accuracy is reported together with token cost and latency, because a more expensive strategy is not automatically better.

## 5. Methods

### Model

The main `v2` model is `Qwen/Qwen2.5-0.5B-Instruct`, run locally with Hugging Face `transformers`. This is deliberately much smaller than frontier models, making the experiment a test of efficient local reasoning rather than maximum achievable benchmark accuracy.

### Datasets

| Dataset | Size | Source | Role |
|---|---:|---|---|
| `prototype_v1` | 50 | Hand-authored | Pipeline prototype and qualitative baseline |
| `gsm8k_test_10` | 10 | GSM8K subset | Pilot sanity check |
| `gsm8k_test_50` | 50 | GSM8K subset | Main benchmark run |

### Strategies

| Strategy | Description |
|---|---|
| Baseline | One deterministic final-answer-only generation |
| Chain-of-thought budget variants | Multiple prompt and token-budget settings; strategy name retained, but visible reasoning is no longer requested |
| Adaptive | Short final-answer pass, with fallback to a larger answer-only budget when the first output looks low-confidence |
| Self-consistency | Multiple sampled answer-only generations followed by majority vote |

### Metrics

The analysis reports:

- Accuracy
- Invalid output rate
- Parse error rate
- Average total tokens
- Total tokens
- Average latency
- Accuracy per 1k tokens
- Qualitative failure modes

## 6. v1 Prototype Results

The original prototype results show that the smaller hand-authored dataset was easier and more sensitive to prompt format:

| Strategy | Accuracy | Notes |
|---|---:|---|
| Baseline | `0.48` | Fixed 50-token generation in stored prototype results |
| Adaptive | `0.58` | Best prototype-level result |
| CoT overall | `0.44` | Strongly budget-dependent |
| Self-consistency | `0.127` | Expensive and unstable |

The most striking `v1` finding is that unconstrained CoT improved with larger token budgets, reaching `0.82` at budget `200`, while constrained CoT and self-consistency underperformed. This suggested that extra generation budget could help on simpler tasks, but also exposed parsing and prompt-adherence problems.

## 7. v2 GSM8K Results

The main `v2` result is the 50-question GSM8K run in `results/v2_qwen_gsm8k_50`.

| Strategy | Examples / Rows | Accuracy | Avg Tokens | Total Tokens | Avg Latency (s) | Invalid Rate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 50 | `0.06` | `181.72` | `9,086` | `3.06` | `0.00` |
| Adaptive | 50 | `0.06` | `109.72` | `5,486` | `0.78` | `0.00` |
| Chain-of-thought variants | 300 | `0.05` | `244.55` | `73,366` | `4.65` | `0.007` |
| Self-consistency | 100 | `0.05` | `770.88` | `77,088` | `13.41` | `0.00` |

The benchmark result is clear: for this small model on GSM8K, inference-time scaling did not meaningfully improve accuracy. Adaptive matched baseline while using fewer tokens and lower latency, which makes it the most efficient strategy among the tested options. Self-consistency was the most expensive strategy and did not improve correctness.

## 8. Interpretation

The difference between `v1` and `v2` is the main research story. On the hand-authored prototype dataset, extra token budget appeared more promising. On GSM8K, the same family of strategies struggled. This suggests that the prototype task distribution was not hard enough to reveal the model's limitations.

The `v2` results should not be framed as a failure of the project. They show an important negative result: simply spending more inference-time compute does not rescue a very small model on multi-step arithmetic. That is a useful finding for efficient AI deployment because it prevents overclaiming what a small model can do without additional tools, verifiers, or stronger arithmetic support.

## 9. Failure Analysis

Several failure modes appear in the current result files:

- Wrong arithmetic operation: the model often selects an operation that is locally plausible but globally incorrect.
- Overgeneration: some outputs begin with an answer and then continue into explanations or new tasks.
- Prompt adherence failures: even answer-only prompts can produce explanatory text.
- Majority-vote collapse: self-consistency can repeatedly vote for the same wrong answer.
- Escalation limits: adaptive fallback cannot help if the fallback prompt produces the same kind of error.

One concrete example from the GSM8K run is a typing-speed average problem where the model repeatedly outputs `51` instead of the target `52`. This shows a reasoning error, not just a parsing error. Another example is a diaper-counting problem where the model initially outputs `10`, even though the final answer is `5`.

These failures motivated the final-answer-only prompt update and the first-line extraction fix. The evaluation now better handles outputs like:

```text
4
To solve this problem...
```

In that case, the parser takes `4` as the answer rather than selecting a later explanation line.

## 10. Limitations

The main limitations are:

- The main benchmark subset contains 50 examples, so the results are exploratory.
- The model is intentionally tiny, so low GSM8K accuracy is expected.
- Existing `v2_qwen_gsm8k_50` outputs were produced before the final-answer-only prompt update, so a final report should either rerun the benchmark or clearly label the prompt update as a post-run methodological correction.
- Self-consistency uses majority vote but no verifier.
- Adaptive confidence is heuristic-based and does not use model probabilities.
- Latency depends on local hardware and may not generalize across machines.

## 11. Future Work

The most useful next steps are:

1. Rerun `v2_qwen_gsm8k_50` after the final-answer-only prompt update.
2. Add a verifier or calculator-assisted scoring pass.
3. Increase GSM8K to 100 or 250 examples if runtime allows.
4. Compare Qwen 0.5B to a slightly larger local model.
5. Add confidence intervals or bootstrap estimates for accuracy.
6. Improve adaptive escalation using output validity, disagreement, or logprob-style uncertainty when available.
7. Add a verifier-style setting to test whether explicit feedback improves scaling behavior.

## 12. Conclusion

This project demonstrates a reproducible local evaluation framework for studying inference-time scaling in small LLMs. The strongest finding is not that small models perform well on GSM8K, but that the framework makes their limitations measurable. On the 50-question GSM8K subset, extra inference-time compute did not reliably improve accuracy for `Qwen/Qwen2.5-0.5B-Instruct`, and expensive strategies such as self-consistency produced poor efficiency.

For a portfolio or resume presentation, the most defensible claim is:

**Built a local LLM benchmarking pipeline that compares inference-time scaling strategies across prototype and benchmark reasoning datasets, tracking accuracy, token cost, latency, and failure modes to evaluate efficient small-model deployment.**
