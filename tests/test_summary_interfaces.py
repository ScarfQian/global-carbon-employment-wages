"""Check that summary stages require only their documented inputs."""

import pandas as pd

from carbon_labour import summaries


def test_future_frontier_summary_does_not_require_monte_carlo(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(summaries, "build_future_scope_inventory", lambda: calls.append("inventory"))
    monkeypatch.setattr(summaries, "select_future_authoritative_scopes", lambda: (None, "frontiers"))
    monkeypatch.setattr(summaries, "build_future_policy_bands", lambda frame: calls.append(frame))

    def unexpected_probability_read():
        raise AssertionError("The frontier stage must not read Monte Carlo inputs")

    monkeypatch.setattr(summaries, "build_future_probability_summary", unexpected_probability_read)
    summaries.main(["--stage", "future-frontiers", "--output-root", str(tmp_path)])
    assert calls == ["inventory", "frontiers"]


def test_failed_union_scope_is_not_labelled_converged(monkeypatch, tmp_path):
    read_root = tmp_path / "02_Future"
    read_root.mkdir()
    pd.DataFrame({
        "Pathway": ["P3"], "Year": [2050],
        "Future_NSGA_Scope": ["selected_world_union_frontier_g30"],
        "NSGA3_Generations": [30], "Policy_ID": ["synthetic"], "Seed": [42],
    }).to_csv(read_root / "future_nsga3_frontiers_final.csv", index=False)
    pd.DataFrame({"World": ["P3-2050"], "Generations": [30], "Passed": [False]}).to_csv(
        read_root / "future_nsga3_selected_world_convergence.csv", index=False
    )
    for name, value in {
        "FINAL_ROOT": tmp_path, "FUTURE_READ_DIR": read_root, "FUTURE_DIR": tmp_path / "04_Summaries"
    }.items():
        monkeypatch.setattr(summaries, name, value)
    scopes, _ = summaries.select_future_authoritative_scopes()
    assert scopes.iloc[0]["Convergence_Status"] == "failed"
    assert "converged" not in scopes.iloc[0]["Selection_Rule"]
