**Built a local LLM benchmarking pipeline that compares inference-time scaling strategies across prototype and benchmark reasoning datasets, tracking accuracy, token cost, latency, and failure modes to evaluate efficient small-model deployment.**


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
