import random

import numpy as np

from imageforge.fidelity import compare, diversity, frechet_distance, kernel_distance, nearest_neighbour_similarity


def test_kernel_distance_is_near_zero_for_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(200, 32)) * 3
    b = rng.normal(size=(200, 32)) * 3
    same = kernel_distance(a, b)
    shifted = kernel_distance(a, rng.normal(size=(200, 32)) * 3 + 2.0)
    assert abs(same) < 0.5
    assert shifted > same + 1.0
    # unbiased: a small sample from the same distribution still sits near zero relative to a shift
    assert abs(kernel_distance(a[:12], b)) < abs(shifted) / 4


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_frechet_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16))
    assert frechet_distance(a, a) < 1e-6
    near = frechet_distance(a, a + 0.1)
    far = frechet_distance(a, a + 2.0)
    assert 0 < near < far


def test_nearest_neighbour_detects_copies():
    rng = np.random.default_rng(1)
    real = _unit(rng.normal(size=(50, 16)))
    copies = real[:10]
    fresh = _unit(rng.normal(size=(10, 16)))
    assert nearest_neighbour_similarity(copies, real) > 0.999
    assert nearest_neighbour_similarity(fresh, real) < 0.9
    assert nearest_neighbour_similarity(real, real, exclude_self=True) < 0.999


def test_diversity_zero_when_collapsed():
    x = _unit(np.ones((5, 8)))
    assert abs(diversity(x)) < 1e-6
    rng = np.random.default_rng(2)
    assert diversity(_unit(rng.normal(size=(50, 8)))) > 0.5


def test_compare_reports_both_pairs():
    rng = np.random.default_rng(3)
    real = _unit(rng.normal(size=(40, 8)))
    syn = _unit(rng.normal(size=(20, 8)))
    out = compare(real, syn, random.Random(0))
    assert set(out) == {"synthetic_vs_real", "real_vs_real"}
    assert out["synthetic_vs_real"]["n"] == 20
    # baseline uses a real subset of the same size as the synthetic set
    assert out["real_vs_real"]["n"] == 20
    for m in out.values():
        assert all(k in m for k in ("fd", "kid", "nn", "div"))


def test_baseline_is_size_matched():
    rng = np.random.default_rng(4)
    real = _unit(rng.normal(size=(200, 8)))
    small = _unit(rng.normal(size=(5, 8)))
    big = _unit(rng.normal(size=(100, 8)))
    small_out = compare(real, small, random.Random(0))
    big_out = compare(real, big, random.Random(0))
    assert small_out["real_vs_real"]["n"] == 5
    assert big_out["real_vs_real"]["n"] == 100
    # a synthetic set larger than the real pool still gets a baseline of at most half the pool
    huge = _unit(rng.normal(size=(500, 8)))
    assert compare(real, huge, random.Random(0))["real_vs_real"]["n"] == 100
    # a tiny baseline carries the same small-sample inflation as a tiny synthetic set
    assert small_out["real_vs_real"]["fd"] > big_out["real_vs_real"]["fd"]
