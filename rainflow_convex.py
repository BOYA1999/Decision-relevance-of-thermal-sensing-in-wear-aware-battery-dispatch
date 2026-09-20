import json
from itertools import product
from pathlib import Path

import numpy as np
from vendor.rainflow import extract_cycles


def rainflow_value_gradient(soc):
    x = np.asarray(soc, dtype=float).ravel()
    gradient = np.zeros_like(x)
    cycles = [(abs(x[1] - x[0]), (x[0] + x[1]) / 2, 0.5, 0, 1)] if len(x) == 2 else extract_cycles(x)
    value = 0.0
    for amplitude, _, count, start, end in cycles:
        value += count * amplitude ** 1.7 / 6000
        slope = 1.7 * count * amplitude ** 0.7 * np.sign(x[start] - x[end]) / 6000
        gradient[start] += slope
        gradient[end] -= slope
    return float(value), gradient


def run_checks():
    from plant import battery_wear
    rng = np.random.default_rng(20260906)
    worst_finite_difference = 0.0
    worst_support_excess = -np.inf
    worst_tied_support_excess = -np.inf
    worst_local_tied_excess = -np.inf
    for _ in range(100):
        x = rng.uniform(0.05, 0.95, 17)
        value, gradient = rainflow_value_gradient(x)
        finite_difference = np.zeros_like(x)
        for index in range(len(x)):
            change = np.eye(1, len(x), index).ravel() * 1e-7
            finite_difference[index] = (rainflow_value_gradient(x + change)[0] - rainflow_value_gradient(x - change)[0]) / 2e-7
        error = float(np.max(np.abs(gradient - finite_difference)))
        worst_finite_difference = max(worst_finite_difference, error)
        assert np.allclose(gradient, finite_difference, atol=2e-11, rtol=1e-5)
        assert np.isclose(value, battery_wear(x), atol=1e-15)
    for _ in range(1000):
        length = int(rng.integers(2, 65))
        x, other = rng.uniform(0, 1, (2, length))
        value, gradient = rainflow_value_gradient(x)
        excess = value + gradient @ (other - x) - rainflow_value_gradient(other)[0]
        worst_support_excess = max(worst_support_excess, float(excess))
        assert excess <= 1e-11
        tied = rng.integers(0, 6, length) / 5
        tied_value, tied_gradient = rainflow_value_gradient(tied)
        excess = tied_value + tied_gradient @ (other - tied) - rainflow_value_gradient(other)[0]
        worst_tied_support_excess = max(worst_tied_support_excess, float(excess))
        assert excess <= 1e-11
    assert np.isclose(rainflow_value_gradient([0, 1, 0])[0], 1 / 6000)
    assert np.isclose(rainflow_value_gradient([0, 1])[0], 0.5 / 6000)
    assert np.allclose(rainflow_value_gradient([0.5, 0.5, 0.5])[1], 0)
    prefix, future = rng.uniform(0, 1, 11), rng.uniform(0, 1, 9)
    full = np.r_[prefix, future]
    value, gradient = rainflow_value_gradient(full)
    tail_gradient = gradient[len(prefix):]
    replacement = rng.uniform(0, 1, 9)
    assert value + tail_gradient @ (replacement - future) <= rainflow_value_gradient(np.r_[prefix, replacement])[0] + 1e-11
    for values in product([0.0, 0.5, 1.0], repeat=5):
        x = np.array(values)
        value, gradient = rainflow_value_gradient(x)
        for _ in range(10):
            other = x + rng.normal(0, 1e-5, 5)
            excess = value + gradient @ (other - x) - rainflow_value_gradient(other)[0]
            worst_local_tied_excess = max(worst_local_tied_excess, float(excess))
            assert excess <= 1e-12
    return {
        "status": "PASS", "generic_finite_difference_paths": 100,
        "random_supporting_hyperplane_pairs": 1000, "tied_supporting_hyperplane_pairs": 1000,
        "worst_finite_difference_absolute_error": worst_finite_difference,
        "maximum_random_support_excess": worst_support_excess,
        "maximum_tied_support_excess": worst_tied_support_excess,
        "exhaustive_five_point_tied_patterns": 243,
        "local_tied_perturbations": 2430,
        "maximum_local_tied_support_excess": worst_local_tied_excess,
        "support_tolerance": 1e-11,
        "scope": "Convex depth-only battery wear with full-cycle weight 1 and residue-half-cycle weight 0.5; not converter mean-temperature-weighted wear",
    }


if __name__ == "__main__":
    report = run_checks()
    Path(__file__).with_name("rainflow_convex_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
