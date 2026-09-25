# Research trail

The starter borrows questions and measurement practices, not implementation code or private data, from these public sources:

| Source | What it contributes to this experiment |
| --- | --- |
| [Devin Fusion announcement](https://cognition.com/blog/devin-fusion) and [Fusion in Devin CLI](https://docs.devin.ai/cli/fusion) | A lead/sidekick pairing and delegation within a session motivated subtask-level decisions. These sources describe sidekick work and dynamic routing, not a personalized contextual-bandit learner. |
| [RouteLLM](https://arxiv.org/html/2406.18665) | Preference-based routing frames the quality/cost tradeoff; our small prior table and verified reward are an intentionally simpler starting point. |
| [FrugalGPT](https://arxiv.org/html/2305.05176) | Cascades and budget-aware model use motivate charging for every attempt, not just the first successful call. |
| [RouterBench](https://arxiv.org/html/2403.12031) | Routing needs explicit baselines and comparable evaluation conditions; our fixtures are too small and synthetic to substitute for a benchmark. |
| [PersonalizedRouter](https://arxiv.org/html/2511.16883) | Models user preferences from interaction data with a graph-based approach; our opt-in per-context reward average explores a smaller, different personalization question. |
| [Contextual-bandit news recommendation](https://arxiv.org/html/1003.0146) | A motivating example of learning choices from context and partial feedback; we log the probability of the action actually taken. |
| [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/html/1103.4601) and [Open Bandit Dataset and Pipeline](https://arxiv.org/html/2008.07146) | Future offline comparison should use logged propensities, sufficient support, and uncertainty checks. This starter reruns policies in a simulator instead of claiming an off-policy estimate. |
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4) | Automated quality labels can be biased; a real verifier needs domain-specific checks and a plan for unverified outcomes. |

The synthetic capability values, success offsets, and credit units in this repository were invented for the demonstration. No nonpublic ranking design, user data, credentials, provider pricing, or empirical performance claims are included.
