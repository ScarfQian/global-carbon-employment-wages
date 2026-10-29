# Global carbon, employment and wages

Python model and analysis code for examining how carbon, employment and wages are distributed across countries and skill groups in international trade, and how transition policies change those relationships.

The analysis uses the 2022 product-by-product tables from EXIOBASE 3.9.6: 44 countries and five rest-of-world regions, with 200 products per region. It combines country-skill accounting, cross-border network analysis and four-objective policy evaluation. The three skill groups are Low, Medium and High.

## Installation

Tested with Python 3.12.7 and the package versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

## Data

Obtain EXIOBASE 3.9.6, year 2022, product-by-product (`pxp`), from [EXIOBASE](https://www.exiobase.eu/). Keep the extracted table and satellite-account directories together. Set the input path explicitly:

```bash
export EXIOBASE_PXP_PATH=/path/to/IOT_2022_pxp
```

Data and results are stored separately. See [data requirements](docs/data.md) for the required files and units.

## Run the analyses

All stages accept `--data-path` and `--output-root`. Results default to the ignored local `results/` directory.

```bash
python scripts/run_analysis.py --stage check-model
python scripts/run_analysis.py --stage baseline
python scripts/run_analysis.py --stage current-core
python scripts/run_analysis.py --stage future-deterministic
```

| Stage | Analysis |
| --- | --- |
| `check-model` | Carbon attribution conservation, the inequality definition and full-versus-aggregated MRIO algebra |
| `baseline` | Country-skill positions, bilateral flows, network metrics, corridor comparisons and attribution/threshold comparisons |
| `current-core` | Baseline and five policy scenarios, including fixed-baseline labour-hour lock |
| `current-monte-carlo` | Policy uncertainty with 1,000 draws |
| `current-sobol` | Eight-parameter sensitivity and sample-size comparison |
| `current-nsga3` | Current four-objective policy search with three seeds |
| `future-deterministic` | Four pathways across nine years, plus the policy package and anchor-year scenarios |
| `future-monte-carlo` | Uncertainty analysis for 12 pathway-year combinations |
| `future-sobol` | Eight-parameter sensitivity for the same 12 combinations |
| `future-nsga3` | Four-objective policy frontiers and convergence checks |
| `future-nsga3-extension` | Additional generations when convergence criteria are not met |

`current-all` runs the four current-policy stages. `future-all` runs the four future-policy stages. Policy searches and uncertainty analyses take substantially longer than the deterministic stages. Scenario settings, parameter bounds, seeds and generation counts are defined in the source modules.

The objectives are carbon reduction, employment growth, growth in total wages and reduction in the country-skill inequality index. Negative improvements are retained in the objective vectors. The inequality index is the mean of the carbon-employment and carbon-wage bivariate Ginis. Labour hours are reported alongside these outcomes. S5 is a specified combination of carbon pricing, renewable transition, demand substitution and transition support.

## Summarize policy results

After the corresponding model stages have produced their input tables:

```bash
python scripts/summarize_results.py --stage current-frontier
python scripts/summarize_results.py --stage future-frontiers
python scripts/summarize_results.py --stage future-probabilities
python scripts/summarize_results.py --stage network
```

These commands calculate frontier representatives, frontier structure, policy-variable intervals, probability summaries and network-share divergences. `future-frontiers` reads the future search tables; `future-probabilities` reads the future Monte Carlo summaries. Plotting code is maintained separately.

## Numerical verification

The comparison interface accepts a separate reference directory:

```bash
python scripts/verify_results.py \
  --reference-root /path/to/reference_tables \
  --output-root results \
  --group current future hour-lock \
  --report results/numerical-comparison.csv
```

Reference tables and comparison reports are stored separately. The tests use small synthetic MRIO systems. See [reproduction instructions](docs/reproduction.md) for the execution order and tested analyses.

## Code organization

| Module | Purpose |
| --- | --- |
| `footprint.py` | Load EXIOBASE and construct carbon, employment, wage and labour-hour coefficients |
| `misalignment.py`, `positions.py` | Country-skill comparisons and position summaries |
| `network.py`, `network_metrics.py`, `corridors.py` | Bilateral flows, network structure and corridor comparisons |
| `attribution.py`, `thresholds.py` | Attribution bases and high-carbon thresholds |
| `inequality.py` | Skill-carbon attribution and bivariate inequality |
| `base_policy.py`, `policy.py` | Policy shocks, scenario accounting and multi-objective search |
| `future.py` | Future pathway parameters and scenario snapshots |
| `pipeline.py`, `baseline.py` | Analysis entry points |
| `summaries.py` | Frontier, fixed-set labour-hour and network summaries |

## Citation and license

Please cite the repository and the data source when using this code. Repository citation metadata is provided in `CITATION.cff`. The code is released under the MIT license; EXIOBASE data retain their own access and licensing conditions.
