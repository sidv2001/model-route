# model-route

A small, runnable experiment in choosing a model for each subtask using explicit context, synthetic capability priors, and feedback a user has opted in to share. [See the decision map](docs/design.md#decision-map) or read the [research notes](docs/research.md).

## Proposal

Some subtasks deserve a careful model; others need a quick answer and a check. I want model-route to make that choice inspectable instead of hiding it behind one "best model" setting. A rollback plan and a one-line copy edit shouldn't share a default just because they arrived in the same session. This first version asks for two explicit features, domain and complexity, then scores Quick, Balanced, and Careful using published-in-the-repo synthetic capability priors. The rubric is deliberately small enough to challenge.

Devin Fusion's lead-and-sidekick workflow helped sharpen the question for me: keep consequential decisions with a strong lead, and delegate mechanical work when it makes sense. Its published description covers sidekick delegation and mid-session routing; personalization is the separate experiment here. If the same person runs many subtasks, can verified outcomes teach a router when a cheaper model is enough for *that person's* mix of work? I want the answer to depend on evidence, with each choice visible and learning easy to stop.

This starter uses a categorical contextual epsilon-greedy policy. For each subtask it estimates verified success over a small retry budget, then subtracts synthetic call credits and retry cost. Exploration keeps every model selectable; the log records the exact probability of the model actually chosen. Feedback changes estimates only for the matching user, domain, complexity, and model. History is opt-in and can be forgotten. When verification is missing, the call still has a cost, but it supplies no training label.

Everything runs locally against nine invented subtask templates and two invented user profiles. The selected model gets at most two retries, and a simulated verifier produces the success signal. The report compares the router with always-Quick, always-Careful, uniform random, and a fixed complexity rule on the same seeded workload. It shows verified success, credits, retries, and reward together so a cheap but unreliable route cannot quietly look like a win.

I see this as a measurement scaffold, not a result about real providers. The useful next steps are opt-in, independently verified outcomes; calibration checks across domains and users; and offline policy evaluation with logged propensities before changing live traffic. A real integration would also need retention rules and a safe rollback path. For now, anyone can run the experiment, inspect a choice, change the assumptions, and see exactly what moves.

## Run it

Requires Python 3.12 or later. No model API or credentials are needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m model_route --seed 7 --rounds 20 --personalize --report reports/demo.json
.venv/bin/python -m pytest
```

Omit `--personalize` to use priors without retaining per-user feedback. `--missing-every 5` withholds verification for every fifth subtask; `--max-retries 0`, `1`, or `2` changes the same-model retry cap. The console prints one row per policy. `reports/demo.json` contains the configuration, summaries, and **chosen actions only**: task context, selected model, its exact selection probability, observed success or unknown, total synthetic credits, retries, and reward or null. No user identifiers or task descriptions are written.

Complexity is assigned explicitly in the [synthetic tasks](src/model_route/data/scenarios.json): **1** is bounded and directly checkable; **2** has dependent steps or local verification; **3** is ambiguous, cross-component, or needs a careful recovery plan. Domain is `code`, `writing`, or `reasoning`. The [profile fixture](src/model_route/data/profiles.json) lists every starting probability and per-call credit. These are illustrative assumptions, not measured model performance or dollar prices.

The [design and roadmap](docs/design.md) explain the policy math, privacy boundaries, baselines, and what would need to change for real outcomes. The existing [MIT license](LICENSE) covers the original code and diagram.
