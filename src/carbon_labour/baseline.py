"""Run baseline diagnostics using an already loaded footprint engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from . import attribution, corridors, misalignment, network_metrics, positions
from . import production_lock, thresholds, wage_reconciliation
from .footprint import FootprintEngine
from .network import NetworkEngine


def _write_frames(frames: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write tables with their original comparison filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False)


def _reconcile_wages(engine: FootprintEngine, output_dir: Path) -> None:
    """Use current footprint accounts, keeping residual wages separate."""
    broad = engine.results.get("wage_by_skill")
    if broad is None:
        broad = engine.compute_wage_footprint_by_skill()
    direct = engine.results.get("production_side_country_skill_triplets")
    if direct is None:
        direct = engine.compute_production_side_country_skill_triplets()
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    broad.to_csv(csv_dir / "wage_by_skill.csv")
    direct.to_csv(csv_dir / "production_side_country_skill_triplets.csv", index=False)
    wages = broad["Wage_Footprint"]
    skill_mask = broad.index.astype(str).isin(wage_reconciliation.SKILLS)
    account = wage_reconciliation.WageAccount
    accounts = [
        account(
            "descriptive footprint engine", "wage_by_skill.csv",
            "broad wage account including residual categories", "M.EUR",
            float(wages.sum()), False, "Low; Medium; High; residual categories", "Yes",
        ),
        account(
            "skill-resolved accounts", "wage_by_skill.csv",
            "Low/Medium/High wage account", "M.EUR",
            float(wages.loc[skill_mask].sum()), True, "Low; Medium; High", "No",
        ),
        account(
            "descriptive country-skill table", "production_side_country_skill_triplets.csv",
            "production-side country-skill direct wage", "M.EUR",
            float(direct["Direct_Wage"].sum()), True, "Low; Medium; High", "No",
        ),
    ]
    from .policy import FinalScenarioEngine
    policy = FinalScenarioEngine(engine)
    policy_total = float(policy.compute_baseline()["Wage_Total"])
    accounts.append(account(
        "policy engine", "policy.py", "Low/Medium/High policy wages", "M.EUR",
        policy_total, True, "Low; Medium; High", "No",
    ))
    reconciliation, audit = wage_reconciliation.compute_wage_reconciliation(accounts)
    wage_reconciliation.write_outputs(reconciliation, audit, output_dir)
    summary = wage_reconciliation.wage_account_summary(reconciliation)
    summary.to_csv(csv_dir / "wage_account_reconciliation_summary.csv", index=False)


def _run_production_lock(engine: FootprintEngine, output_dir: Path) -> None:
    """Compute the original total-direct-carbon employment and hours locks."""
    metrics = production_lock.InequalityMetrics(engine)
    production = metrics.compute_production_side_inequality()
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "inequality_production_inequality.json").write_text(
        json.dumps(production, indent=2), encoding="utf-8"
    )
    hours = engine.results.get("high_carbon_hours_share_by_skill_gender")
    if hours is None:
        hours = engine.compute_high_carbon_hours_by_skill_gender()
    csv_dir = output_dir / "csv"
    _write_frames({"high_carbon_hours_share_by_skill_gender.csv": hours}, csv_dir)
    production_lock.export_employment_lock_table(production, csv_dir)
    production_lock.export_hours_lock_table(hours, csv_dir)


def _network_from_footprint_engine(engine: FootprintEngine) -> NetworkEngine:
    """Reuse loaded matrices and the original network extension extraction."""
    network = NetworkEngine(str(engine.data_path))
    network.mrio = engine.mrio
    network.L = engine.mrio.L
    network.Y = engine.mrio.Y
    network.Z = engine.mrio.Z
    network.regions = list(engine.regions)
    network.sectors = list(engine.products)
    network.co2_emissions = np.asarray(engine.co2_emissions, dtype=float).reshape(-1)
    network._extract_extensions()
    return network


def run_baseline(footprint_engine: FootprintEngine, output_root: str | Path) -> Dict[str, Any]:
    """Write baseline tables without loading data or reading reference results.

    The engine must already contain the MRIO matrices and extensions. Network
    skill flows retain their intermediate-input definition; country-sector flows
    retain their final-demand definition. Threshold sensitivity is supplementary
    and does not replace the total-direct-carbon production locking definition.
    """
    if footprint_engine.mrio is None:
        raise ValueError("run_baseline requires an already loaded FootprintEngine")
    root = Path(output_root)
    country_dir = root / "01_Country_Skill"
    corridor_dir = root / "02_Corridor_Skill"
    network_dir = root / "02_Network"
    lock_dir = root / "03_Network_And_Lockin"
    inputs = misalignment.build_inputs_from_footprint_engine(footprint_engine)
    package = misalignment.compute_misalignment_package(inputs)
    misalignment.write_outputs(package, country_dir)

    producer = positions.prepare_country_skill_table(package["producer_skill"], "Production")
    consumer = positions.prepare_country_skill_table(package["consumer_skill"], "Consumption")
    _write_frames(
        {
            "quadrant_summary_production.csv": positions.build_quadrant_summary(producer),
            "quadrant_summary_consumption.csv": positions.build_quadrant_summary(consumer),
            "top_country_skill_cells_by_quadrant.csv": positions.build_top_country_skill_cells(
                pd.concat([producer, consumer], ignore_index=True)
            ),
            "quadrant_summary_audit.csv": positions.build_quadrant_audit(producer, consumer),
        },
        country_dir / "csv",
    )
    coverage, robustness, summary = attribution.compute_attribution_robustness(inputs)
    attribution.write_outputs(coverage, robustness, summary, root)
    _write_frames(
        thresholds.compute_threshold_package(package["producer_sector_skill"]),
        lock_dir / "csv",
    )
    _reconcile_wages(footprint_engine, country_dir)
    _run_production_lock(footprint_engine, lock_dir)

    network = _network_from_footprint_engine(footprint_engine)
    network.build_network(method="aggregated")
    network_tables = network.save_results(str(network_dir))
    transmission = network_tables["transmission"]
    sector_frames = network_tables["country_sector"]
    skill_frames = {
        metric: transmission[f"bilateral_{metric}_transfer_by_skill"]
        for metric in corridors.METRIC_VALUE_COLUMNS
    }
    limitations = [
        "Country-sector drivers use final-demand-induced production; by-skill flows use intermediate inputs.",
    ]
    missing_skill = [metric for metric, frame in skill_frames.items() if frame.empty]
    missing_sector = [metric for metric, frame in sector_frames.items() if frame.empty]
    if not missing_skill and not missing_sector:
        network_metrics.run_all(
            {metric: str(network_dir / "csv" / f"network_bilateral_{metric}_transfer_by_skill.csv")
             for metric in skill_frames},
            {metric: str(network_dir / "csv" / f"network_bilateral_{metric}_transfer_by_country_sector.csv")
             for metric in sector_frames},
            lock_dir,
        )
    else:
        limitations.append(
            f"Network overlap/concentration/divergence unavailable: empty skill accounts {missing_skill}; "
            f"empty country-sector accounts {missing_sector}."
        )
    if not missing_skill:
        corridor, audit = corridors.compute_corridor_skill_misalignment(skill_frames)
        top = corridor.loc[corridor["R_CW"].notna()].sort_values("R_CW", ascending=False).head(50).reset_index(drop=True)
        top_by_scale = corridor.sort_values(
            ["Weighted_Burden_Return_Score", "Carbon_Transfer"], ascending=[False, False]
        ).head(50).reset_index(drop=True)
        summaries = corridors.build_corridor_summaries(corridor)
        if not sector_frames["carbon"].empty:
            drivers = corridors.build_sector_drivers(network_dir / "csv", top, top_sectors=5)
        else:
            drivers = pd.DataFrame()
            limitations.append("Corridor sector drivers unavailable: no country-sector carbon table was generated.")
        corridors.write_outputs(corridor, top, top_by_scale, drivers, audit, summaries, corridor_dir)
        exclusive = positions.prepare_corridor_table(corridor)
        if not exclusive.empty:
            ranked, pairs = positions.scale_weighted_corridor_tables(exclusive)
            _write_frames({
                "top_corridors_exclusive_scale_weighted_v3.csv": ranked,
                "top_region_pairs_exclusive_burden_return_v3.csv": pairs,
                "corridor_skill_misalignment_scale_weighted.csv": exclusive,
            }, corridor_dir / "csv")
            _write_frames({"Figure2_authoritative_top_corridors_snapshot.csv": ranked}, root / "03_SourceData")
            _write_frames(
                {
                    "corridor_skill_exclusive_type_summary.csv": positions.build_corridor_exclusive_summary(exclusive),
                    "top_corridors_by_exclusive_type.csv": positions.build_top_corridors_by_exclusive_type(exclusive),
                    "corridor_skill_exclusive_type_by_skill.csv": positions.build_corridor_group_summary(
                        exclusive, ["Skill", "Exclusive_Type"]
                    ),
                    "corridor_skill_exclusive_type_by_region_pair.csv": positions.build_corridor_group_summary(
                        exclusive, ["From_Region", "To_Region", "Exclusive_Type"]
                    ),
                    "corridor_skill_exclusive_type_audit.csv": positions.build_corridor_exclusive_audit(exclusive, corridor),
                },
                corridor_dir / "csv",
            )
        else:
            limitations.append("Exclusive corridor summaries unavailable: no valid cross-border ratio cells.")
    else:
        limitations.append(f"Corridor diagnostics unavailable: empty skill accounts {missing_skill}.")
    return {
        "output_root": str(root),
        "country_skill": str(country_dir),
        "corridor_skill": str(corridor_dir),
        "network": str(network_dir),
        "network_and_lockin": str(lock_dir),
        "limitations": limitations,
    }
