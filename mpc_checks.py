import json
from pathlib import Path

import numpy as np
from mpc import solve_mpc
from plant import simulate_interval


def run_checks():
    reports = {}
    horizon = 8
    flat = solve_mpc(np.full((1, horizon), 40.0), np.full(horizon, 25.0),
                     np.full(horizon, 0.15), np.full(horizon, 0.05),
                     100.0, [35.0, 30.0], 0.3, [0.5], {})
    assert flat["status"] == "optimal"
    assert np.allclose(flat["plan"]["ch"], 0, atol=1e-7)
    assert np.allclose(flat["plan"]["dis"], 0, atol=1e-7)
    assert np.isclose(flat["objective_upper"], 40.2 * horizon * 0.25 * 0.15)
    reports["flat_price_no_arbitrage"] = flat["objective_upper"]
    horizon = 12
    buy = np.r_[np.full(4, 0.06), np.full(4, 0.4), np.full(4, 0.1)]
    sell = np.full(horizon, 0.03)
    net = np.full((3, horizon), 40.0)
    net[1, 1:] += 10
    net[2, 1:] -= 8
    ambient, prefix = np.full(horizon, 30.0), [0.5, 0.7, 0.4, 0.5]
    nominal = solve_mpc(net, ambient, buy, sell, 100, [40, 35], 0.4, prefix, {"rho": 0.1})
    assert nominal["status"] == "optimal", nominal
    assert nominal["rainflow_gap"] <= 0.005
    assert nominal["objective_gap"] >= -1e-6 and nominal["objective_gap"] <= 0.005 + 1e-6
    assert max(nominal["constraint_residuals"].values()) < 2e-6
    assert nominal["simultaneous_charge_discharge_kW"] < 2e-6
    assert nominal["nonanticipativity_error"] < 2e-6
    plan = nominal["plan"]
    assert np.allclose(plan["energy"][:, 0], 100)
    assert np.allclose(plan["energy"][:, -1], 100)
    assert np.all((plan["energy"] >= 20 - 1e-6) & (plan["energy"] <= 180 + 1e-6))
    assert np.all(plan["Tj"][:, 1:] <= 105 + 1e-6)
    assert np.all(plan["Ts"][:, :-1] + 6 * plan["q"] <= 105 + 1e-6)
    assert np.allclose(plan["gimp"] - plan["gexp"], net + plan["ch"] - plan["dis"] + plan["q"])
    assert np.allclose(np.diff(plan["energy"], axis=1), 0.25 * (0.98 * plan["ch"] - plan["dis"] / 0.98))
    through = plan["ch"] + plan["dis"]
    knots = np.linspace(0, 100, 5)
    pwl = 0.2 + 0.009 * through + 0.00018 * (1 + 0.4 ** 2) * np.interp(through, knots, knots ** 2)
    assert np.all(plan["q"] >= pwl - 1e-7)
    reference_peak = -np.inf
    for scene in range(3):
        state = [40, 35]
        for k in range(horizon):
            physical = simulate_interval(state, plan["dis"][scene, k] - plan["ch"][scene, k], 30, 0.4)
            reference_peak = max(reference_peak, physical["peak_Tj"])
            state = physical["state"]
    assert reference_peak <= 105 + 1e-6
    reports["nominal"] = {key: nominal[key] for key in ["status", "iterations", "rainflow_gap", "solve_seconds", "objective_gap", "constraint_residuals"]}
    reports["nominal_reference_1s_peak"] = reference_peak
    hot = solve_mpc(np.full((1, 8), 90), np.full(8, 45), np.r_[np.full(4, 0.01), np.full(4, 0.8)],
                    np.full(8, 0.005), 100, [70, 65], 0.8, [0.5], {"margin": 3, "temperature_weight": 0})
    assert hot["status"] == "optimal"
    assert np.isclose(hot["plan"]["Tj"][:, 1:].max(), 102, atol=1e-6)
    state, hot_peak = [70, 65], -np.inf
    for p in (hot["plan"]["dis"] - hot["plan"]["ch"])[0]:
        physical = simulate_interval(state, p, 45, 0.8)
        state, hot_peak = physical["state"], max(hot_peak, physical["peak_Tj"])
    assert hot_peak <= 102 + 1e-6
    reports["binding_thermal_constraint"] = {"predicted_limit": 102, "reference_1s_peak": hot_peak}
    horizon = 8
    weights = np.array([0.2, 0.3, 0.5])
    flat_net = np.array([np.full(horizon, 30), np.full(horizon, 40), np.full(horizon, 50)])
    expected = solve_mpc(flat_net, np.full(horizon, 25), np.full(horizon, 0.15),
                         np.full(horizon, 0.05), 100, [35, 30], 0.3, [0.5],
                         {"rho": 0, "scenario_weights": weights, "battery_capex": 0, "health_constraints": False})
    exact_expected = 0.25 * 0.15 * float(weights @ (flat_net + 0.2).sum(axis=1))
    assert expected["status"] == "optimal"
    assert np.isclose(expected["objective_upper"], exact_expected, atol=1e-6)
    same = solve_mpc(np.full((3, horizon), 40), np.full(horizon, 25), np.full(horizon, 0.15),
                     np.full(horizon, 0.05), 100, [35, 30], 0.3, [0.5], {"rho": 0.1})
    assert same["status"] == "optimal"
    assert np.isclose(same["objective_upper"], flat["objective_upper"], atol=1e-6)
    assert np.isclose(same["power"], flat["power"], atol=1e-6)
    reports["rho_zero_weighted_expected_cost"] = expected["objective_upper"]
    reports["identical_support_matches_single_scenario"] = same["objective_upper"]
    cut_limit = solve_mpc(net, ambient, buy, sell, 100, [40, 35], 0.4, prefix, {"max_cuts": 1})
    assert cut_limit["status"] == "cut_limit" and cut_limit["rainflow_gap"] > 0.005
    reports["unconverged_cut_limit_is_exposed"] = cut_limit["rainflow_gap"]
    failure = solve_mpc(np.full((1, 8), 40), np.full(8, 25), np.full(8, 0.15),
                        np.full(8, 0.05), 100, [120, 120], 0.3, [0.5], {})
    assert failure["status"] == "solver_failed" and failure["power"] is None
    reports["infeasible_hot_initial_state_is_exposed"] = failure["solver_status"]
    reports["status"] = "PASS"
    return reports


if __name__ == "__main__":
    report = run_checks()
    Path(__file__).with_name("mpc_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
