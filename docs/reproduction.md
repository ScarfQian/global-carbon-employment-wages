# Reproduction scope

The model stages start from the original EXIOBASE 2022 PXP tables. All outputs are written under the requested result directory. Input data, numerical references, verification reports and plots are kept outside the tracked code.

## Computational coverage

The deterministic account, network and scenario stages were tested with the full 2022 input using Python 3.12.7 and the package versions listed in `requirements.txt`. Synthetic tests cover attribution conservation, zero-denominator treatment, the bivariate Gini identity, policy accounting, bilateral aggregation and network distances.

The complete current and future optimization searches, Monte Carlo ensembles and Sobol ensembles were not rerun during code preparation. Their original algorithms, seeds, parameter ranges and stopping rules are included. Current-frontier membership was recomputed from stored evaluations; this check is distinct from rerunning the optimizer. Reference tables and detailed comparisons are maintained separately.

## Execution order

1. Set the original data path and run `check-model`.
2. Run `baseline` for country-skill and trade-network results.
3. Run `current-core` and `future-deterministic` for deterministic policy results.
4. Run the uncertainty, sensitivity and optimization stages as required. Run `future-nsga3-extension` after `future-nsga3` to apply the original extension rule to any selected world that fails the convergence checks.
5. Run the matching result-summary stages and compare against independently stored numerical references.

The future pathway years are 2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055 and 2060. The future uncertainty and policy-search analyses use the key years 2030, 2050 and 2060. The production-technology reference and scenario assumptions are implemented in the source modules and should be retained when reproducing these results.
