"""Check pathway scope, objective direction and distribution distances."""

import numpy as np
import pytest

from carbon_labour import future
from carbon_labour.summaries import jensen_shannon_divergence, nondominated_mask


def test_pathway_year_combinations_are_unique():
    worlds = future.baseline_worlds()
    keys = {(world.pathway, world.year) for world in worlds}
    assert len(keys) == len(worlds) == 36
    assert {world.year for world in worlds} == set(future.YEAR_GRID)
    assert len(future.core_worlds()) == 20
    assert len(future.key_worlds()) == 12


def test_key_world_selection_rejects_unknown_pathways():
    with pytest.raises(ValueError):
        future.key_worlds(pathways=["unknown"])


def test_frontier_retains_negative_tradeoffs_and_equal_points():
    values = np.array([
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    ])
    assert nondominated_mask(values).tolist() == [True, True, False, True]


def test_network_divergence_is_symmetric_and_scale_invariant():
    left = np.array([1.0, 2.0, 0.0])
    right = np.array([0.0, 1.0, 3.0])
    value = jensen_shannon_divergence(left, right)
    assert np.isclose(value, jensen_shannon_divergence(right, left))
    assert np.isclose(value, jensen_shannon_divergence(left * 10, right * 2))
    assert jensen_shannon_divergence(left, left) == 0.0
