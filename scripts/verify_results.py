#!/usr/bin/env python3
"""Compare generated numerical tables with an external reference directory."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


GROUPS = {
    "baseline": [
        ("01_Country_Skill", "01_Country_Skill/csv", name, keys)
        for name, keys in (
            ("misalignment_country_skill_production.csv", ["Region", "Skill"]),
            ("misalignment_country_skill_consumption.csv", ["Region", "Skill"]),
            ("quadrant_summary_production.csv", ["Quadrant"]),
            ("quadrant_summary_consumption.csv", ["Quadrant"]),
            ("skill_carbon_attribution_coverage.csv", ["Attribution_Method", "Accounting_Perspective"]),
            ("skill_carbon_attribution_robustness.csv", ["Attribution_Method", "Accounting_Perspective", "Region", "Skill"]),
            ("wage_account_reconciliation_summary.csv", ["Account_Name"]),
        )
    ] + [
        ("02_Corridor_Skill", "02_Corridor_Skill/csv", name, keys)
        for name, keys in (
            ("corridor_skill_misalignment.csv", ["From_Region", "To_Region", "Skill"]),
            ("corridor_skill_exclusive_type_summary.csv", ["Exclusive_Type"]),
            ("corridor_skill_exclusive_type_by_skill.csv", ["Skill", "Exclusive_Type"]),
            ("corridor_skill_exclusive_type_by_region_pair.csv", ["From_Region", "To_Region", "Exclusive_Type"]),
            ("top_corridors_exclusive_scale_weighted_v3.csv", ["From_Region", "To_Region", "Skill"]),
            ("top_region_pairs_exclusive_burden_return_v3.csv", ["From_Region", "To_Region"]),
        )
    ] + [
        ("03_Network_And_Lockin", "03_Network_And_Lockin/csv", name, keys)
        for name, keys in (
            ("embodied_network_overlap_summary.csv", ["level", "network_a", "network_b"]),
            ("embodied_network_concentration_summary.csv", ["level", "network"]),
            ("embodied_corridor_sector_divergence_summary.csv", ["demand_country", "producer_country", "pair_name"]),
            ("260324-employment_lock_ratio_by_skill.csv", ["Skill"]),
            ("260324-high_carbon_hours_share_by_skill.csv", ["Skill"]),
            ("high_carbon_total_carbon_threshold_robustness.csv", ["Threshold_Percentile", "Skill"]),
            ("high_carbon_intensity_threshold_robustness.csv", ["Threshold_Percentile", "Skill"]),
        )
    ] + [
        ("03_Network_And_Lockin", "02_Network/csv", name, ["Region"])
        for name in ("network_country_centrality.csv", "network_transmission_centrality.csv")
    ],
    "current": [
        ("05_Current_Policy", "01_Current", name, ["Scenario"])
        for name in (
            "current_core_scenarios_final.csv",
            "current_inequality_components_final.csv",
            "current_skill_lock_final.csv",
            "current_net_transfer_concentration_final.csv",
            "current_labour_hour_diagnostics_final.csv",
        )
    ],
    "future": [
        ("06_Future_Policy", "02_Future", name, ["Pathway", "Year"])
        for name in (
            "future_baseline_36states_final.csv",
            "future_core_policy_36states_final.csv",
            "future_anchor_states_20_final.csv",
        )
    ],
    "hour-lock": [
        ("05_Current_Policy", "04_Summaries", name, ["Scenario", "Skill"])
        for name in (
            "policy_hour_lock_fixed_baseline.csv",
            "policy_hour_lock_fixed_baseline_summary.csv",
        )
    ],
    "current-summaries": [
        ("05_Current_Policy", "04_Summaries", "current_frontier_representative_points.csv", ["Representative_Label"]),
        ("05_Current_Policy", "04_Summaries", "frontier_tolerance_sensitivity.csv", ["Tolerance"]),
    ],
    "future-summaries": [
        ("06_Future_Policy", "04_Summaries", "future_frontier_policy_bands_final.csv", ["Pathway", "Year", "Parameter"]),
        ("06_Future_Policy", "04_Summaries", "future_probability_summary_final.csv", ["Pathway", "Year"]),
    ],
}

# Execution metadata varies across machines and is excluded from numeric comparisons.
EXCLUDED_COLUMNS = {
    "Created_At", "Executor", "Run_Version", "Upstream_Sources", "Source_File",
    "Runtime_Seconds", "Runtime_Seconds_Cumulative", "Mask_SHA256",
}


def compare_tables(reference_path, generated_path, keys, rtol, atol):
    reference = pd.read_csv(reference_path)
    generated = pd.read_csv(generated_path)
    sort_keys = [key for key in keys if key in reference.columns and key in generated.columns]
    if not sort_keys:
        raise ValueError(f"No shared row keys for {reference_path.name}")
    reference = reference.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    generated = generated.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    if len(reference) != len(generated):
        raise ValueError(f"Row count differs for {reference_path.name}: {len(reference)} vs {len(generated)}")
    if not reference[sort_keys].equals(generated[sort_keys]):
        raise ValueError(f"Row keys differ for {reference_path.name}")
    numeric = [
        column for column in reference.select_dtypes(include=[np.number, "bool"]).columns
        if column not in EXCLUDED_COLUMNS and not column.startswith("Runtime_")
    ]
    missing = sorted(set(numeric) - set(generated.columns))
    if missing:
        raise ValueError(f"Missing numeric columns in {generated_path.name}: {missing}")
    if not numeric:
        raise ValueError(f"No numeric measurements in {reference_path.name}")
    checked, max_relative = 0, 0.0
    for column in numeric:
        expected = pd.to_numeric(reference[column], errors="raise").to_numpy(dtype=float)
        actual = pd.to_numeric(generated[column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"Numeric mismatch in {reference_path.name}: {column}")
        finite = np.isfinite(expected) & np.isfinite(actual)
        if finite.any():
            relative = np.abs(actual[finite] - expected[finite]) / np.maximum(np.abs(expected[finite]), 1.0)
            max_relative = max(max_relative, float(relative.max()))
        checked += len(expected)
    return {
        "Table": reference_path.name,
        "Rows": len(reference),
        "Numeric_Columns": len(numeric),
        "Compared_Values": checked,
        "Maximum_Relative_Difference": max_relative,
        "Passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group", choices=tuple(GROUPS), nargs="+", default=["current", "future", "hour-lock"])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args(argv)
    rows = []
    for group in args.group:
        for reference_dir, generated_dir, filename, keys in GROUPS[group]:
            reference = args.reference_root / reference_dir / filename
            generated = args.output_root / generated_dir / filename
            if not reference.is_file() or not generated.is_file():
                raise FileNotFoundError(f"Required comparison table missing: {reference} or {generated}")
            rows.append(compare_tables(reference, generated, keys, args.rtol, args.atol))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.report, index=False)
    print(f"Passed: {len(rows)} table comparisons; {sum(row['Compared_Values'] for row in rows)} numeric values")
    print(f"Maximum relative difference: {max(row['Maximum_Relative_Difference'] for row in rows):.3g}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
