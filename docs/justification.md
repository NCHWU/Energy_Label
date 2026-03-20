# Project Justification: Energy Labels for Large Language Models

## The Problem

The adoption of Large Language Models is accelerating across industries, but their energy cost is largely invisible to end users. Unlike household appliances, which carry mandatory EU energy labels (A–G), LLMs have no standardised way to communicate energy efficiency to the people choosing which model to deploy.

This matters for three reasons:

### 1. LLM energy consumption is significant and growing

- Training GPT-3 consumed an estimated 1,287 MWh — equivalent to the annual electricity use of 120 EU households (Patterson et al., 2021).
- Inference now dominates total lifecycle energy: Meta reported that inference accounts for over 60% of their ML energy budget (Wu et al., 2022).
- Global AI electricity demand is projected to reach 85–134 TWh by 2027, comparable to the annual consumption of the Netherlands (IEA, 2024).

### 2. Users lack the information to make efficient choices

When organisations select an LLM for a task, the decision is typically driven by accuracy benchmarks alone (MMLU, HumanEval, etc.). Energy consumption is not reported on any major leaderboard. This means:

- A 70B-parameter model may be chosen when a 7B model achieves comparable accuracy at 10x less energy.
- There is no market incentive for model developers to optimise for efficiency.
- Sustainability commitments cannot be operationalised without measurement.

### 3. The EU energy label model is proven to work

The EU Energy Label Directive (2017/1369) has been one of the most effective policy instruments for reducing energy consumption. Research shows that energy labels:

- Shifted 85% of refrigerator sales to A-rated models within a decade (Schiellerup, 2002).
- Saved an estimated 230 TWh/year across the EU by 2020 (European Commission).
- Work because they are **simple** (a single letter grade), **comparative** (shown at point of selection), and **standardised** (consistent methodology).

There is no technical reason this approach cannot extend to LLMs.

## Our Solution

We propose an **Energy Per Correct Answer (EPCA)** benchmark that:

1. **Measures real energy** — not estimates or proxies, but Joules measured via nvidia-smi GPU power polling with trapezoidal integration on the actual inference hardware.
2. **Ties energy to accuracy** — EPCA = Total Energy / Correct Answers. This penalises models that are energy-hungry, inaccurate, or both.
3. **Assigns A–G labels** — directly borrowing the EU energy label format that consumers and policymakers already understand.
4. **Supports multiple domains** — coding, healthcare, general knowledge. A model rated "A" for coding may be rated "D" for medical tasks.
5. **Is open source and reproducible** — anyone can run the benchmark on their hardware and verify results.

## Why This Is Novel

| Existing work | What it lacks |
|---|---|
| ML CO2 Impact (Lacoste et al., 2019) | Estimates emissions from training only, ignores inference |
| CodeCarbon (Courty et al., 2023) | Measures energy but does not relate it to task accuracy |
| HumanEval / MBPP benchmarks | Measure accuracy but not energy |
| Green AI (Schwartz et al., 2020) | Advocates for efficiency reporting but provides no label system |
| LLM-Perf Leaderboard | Reports throughput, not energy per correct answer |

**Our contribution is the first benchmark that combines real energy measurement with task accuracy into a single, interpretable label.** This bridges the gap between sustainability research and practical model selection.

## Impact

### For practitioners
A data scientist choosing between 5 LLMs for a production deployment can look at the energy label for their domain and pick the most efficient model that meets their accuracy requirements — just as a consumer picks between A-rated washing machines.

### For model developers
Energy labels create a market incentive to optimise for efficiency, not just accuracy. A model that achieves label A will have a competitive advantage over a label D model with similar accuracy.

### For policymakers
The EU AI Act (2024) introduces transparency requirements for AI systems. An energy label provides a standardised, verifiable metric that could be incorporated into AI sustainability reporting.

### For researchers
The EPCA metric and benchmark methodology can be extended to new domains, model architectures, and hardware configurations. The open-source codebase welcomes contributions.

## Feasibility

The benchmark is fully implemented and functional:

- **Energy measurement** is validated against prior work (sustainableA1 framework with nvidia-smi polling).
- **Accuracy evaluation** uses sandboxed code execution with deterministic test assertions (pass@1).
- **Statistical validation** uses bootstrap confidence intervals to ensure observed differences are significant.
- **The web UI** provides an accessible leaderboard for comparing models across domains.

The first benchmark runs targeting 3–5 local models on 100 LeetCode-style tasks can be completed within days on a single NVIDIA GPU workstation.

## References

- Patterson, D. et al. (2021). Carbon Emissions and Large Neural Network Training.
- Wu, C. et al. (2022). Sustainable AI: Environmental Implications, Challenges and Opportunities. MLSys.
- IEA (2024). Electricity 2024 — Analysis and Forecast to 2026.
- Schwartz, R. et al. (2020). Green AI. Communications of the ACM.
- Lacoste, A. et al. (2019). Quantifying the Carbon Emissions of Machine Learning.
- Courty, B. et al. (2023). CodeCarbon: A Lightweight Software Package for Tracking Carbon Emissions.
- European Commission. Regulation (EU) 2017/1369 — Energy Labelling Framework.
