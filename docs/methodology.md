# Methodology: LLM Energy Label Benchmark

## 1. Accuracy Metric: pass@1

### Definition
pass@1 measures whether a model produces a functionally correct solution on its first (and only) attempt. The generated code is executed against a deterministic test suite; if all assertions pass, the task is scored as correct.

### Why pass@1

1. **Functional correctness over textual similarity.** Unlike BLEU or ROUGE, which measure surface-level similarity to a reference solution, pass@1 verifies that the code actually works. A solution that is syntactically different from the reference but semantically correct receives full credit.

2. **Fair energy accounting.** pass@k (k > 1) allows multiple inference attempts per task, but each attempt consumes energy. Using k = 1 ensures a one-to-one mapping between inference energy and the task outcome—there are no hidden retries inflating accuracy while masking energy cost.

3. **Determinism and reproducibility.** With temperature set to 0, pass@1 yields the same result on repeated runs for a given model, making cross-study comparisons meaningful.

4. **Established precedent.** pass@1 is the primary metric in HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and SWE-bench, making our results directly comparable to existing leaderboards.

### Why not alternatives

| Metric | Issue for energy labeling |
|--------|--------------------------|
| pass@k (k > 1) | Multiple attempts consume energy that is not reflected in the accuracy score, breaking the EPCA ratio |
| BLEU / ROUGE | Measures textual overlap, not functional correctness; penalizes valid alternative implementations |
| Human evaluation | Not scalable, not reproducible, introduces subjective variance |
| Perplexity | Measures language modeling quality, not task completion ability |

## 2. Secondary Quality Metrics

While pass@1 drives the energy label, we report two secondary metrics for each correct solution:

- **Runtime (ms):** Wall-clock execution time of the generated solution against the test suite. Lower runtime may indicate a more algorithmically efficient solution.
- **Peak memory (KB):** Maximum resident set size during execution. Lower memory suggests better space efficiency.

These are reported alongside the energy label but do **not** factor into the EPCA calculation. This keeps the label simple and interpretable while still surfacing code quality information.

## 3. Energy Per Correct Answer (EPCA)

### Definition

    EPCA = Total energy consumed (Joules) / Number of correct answers

EPCA is measured per model across the full benchmark suite. A model that uses 100 J total and correctly solves 50 out of 100 tasks has an EPCA of 2.0 J.

### Why EPCA

- **Penalizes both inefficiency and inaccuracy.** A model that is energy-efficient but inaccurate will have a high EPCA (dividing by few correct answers). A model that is accurate but energy-hungry will also have a high EPCA. Only models that are both efficient and accurate score well.
- **Intuitive units.** Joules per correct answer is easy to explain to non-technical audiences.
- **Composable.** EPCA can be computed for subsets of tasks (e.g., by difficulty) without changing the formula.

### Caveats

- EPCA is **inf** when a model solves zero tasks. Such models are assigned label G.
- Energy measurement depends on hardware; results from different machines are not directly comparable. All models in a single comparison must be benchmarked on the same machine.
- CodeCarbon estimates energy from CPU/GPU utilization and TDP; it is an approximation, not a direct power meter reading.

## 4. A–G Label Assignment

Labels map EPCA to letter grades using fixed thresholds (Joules per correct answer):

| Label | EPCA ≤ |
|-------|--------|
| A     | 5      |
| B     | 10     |
| C     | 20     |
| D     | 40     |
| E     | 80     |
| F     | 160    |
| G     | > 160  |

Thresholds are initial estimates and should be recalibrated after large-scale runs. The exponential scale (each band roughly doubles) reflects the wide variance in model energy consumption.

## 5. Statistical Validation

To assess whether observed EPCA differences between models are statistically significant, we use **bootstrap confidence intervals** (1000 resamples, 95% CI) on the EPCA difference. If the CI for (EPCA_A − EPCA_B) excludes zero, the difference is significant.

## 6. Validation Against External Leaderboards

We validate our accuracy results by comparing pass@1 scores against published HumanEval / MBPP leaderboard entries for the same models. If our pass@1 rankings diverge significantly from established benchmarks, it indicates a problem with our task suite or evaluation harness rather than a genuine model difference.
