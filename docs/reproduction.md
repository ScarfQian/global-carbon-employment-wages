# Reproduction

## Execution order

1. Set the EXIOBASE 2022 PXP data path and run `check-model`.
2. Run `baseline` for country-skill and trade-network results.
3. Run `current-core` and `future-deterministic` for deterministic policy results.
4. Run the uncertainty, sensitivity and optimization stages as required. After `future-nsga3`, run `future-nsga3-extension` for any selected pathway-year combination that fails the convergence checks.
5. Run the matching summary stages and compare the outputs with the separately stored reference tables.

The four future pathways cover 2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055 and 2060. Uncertainty and policy-search analyses use 2030, 2050 and 2060. All scenarios retain the 2022 production-technology reference.

## Tested analyses

The baseline accounts, networks, S0-S5 scenarios, deterministic future states and fixed-baseline labour-hour lock were tested with the full 2022 input. Synthetic tests cover attribution conservation, zero denominators, bivariate Ginis, policy accounting, bilateral aggregation and network distances.

Current-frontier membership and representative selection were checked against stored evaluations. Complete optimization searches, Monte Carlo ensembles and Sobol ensembles were not rerun for this release. Their algorithms and settings are included in the source modules.
