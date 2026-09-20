import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from plant import (battery_wear, converter_wear, power_loss, rainflow_cycles,
                   simulate_interval, thermal_discretization, wear_cost)


def run_checks():
    reports = {}
    for cooling in [0.7, 1.0, 1.4]:
        q = float(power_loss(80, 0.3))
        steady = np.array([30 + q * (6 + 20 * cooling), 30 + q * 20 * cooling])
        result = simulate_interval(steady, 80, 30, 0.3, cooling)
        assert np.allclose(result["state"], steady, atol=1e-10)
        state = np.array([40.0, 35.0])
        def ode(t, x):
            exchange = (x[0] - x[1]) / 6
            return [(q - exchange) / 0.3, (exchange - (x[1] - 30) / (20 * cooling)) / 60]
        independent = solve_ivp(ode, [0, 900], state, rtol=1e-10, atol=1e-10).y[:, -1]
        result = simulate_interval(state, 80, 30, 0.3, cooling)
        assert np.allclose(result["state"], independent, atol=2e-7)
        f, g = thermal_discretization(cooling, 1)
        assert np.allclose([result["Tj"][1], result["Ts"][1]], f @ state + g @ [q, 30])
        reports[f"ode_error_cooling_{cooling}"] = float(np.max(np.abs(result["state"] - independent)))
    results = [simulate_interval([40, 35], 80, 30, 0.3, 1, dt) for dt in [0.5, 1, 2]]
    assert all(np.allclose(x["state"], results[0]["state"], atol=1e-8) for x in results)
    assert all(abs(x["delivered_energy_kWh"] - 20) < 1e-12 for x in results)
    assert len(rainflow_cycles([0, 1, 0])) == 2
    assert np.isclose(battery_wear([0, 1, 0]), 1 / 6000)
    assert np.isclose(battery_wear([0, 1]), 0.5 / 6000)
    assert np.isclose(converter_wear([60, 100, 60]), 1e-6)
    nested = rainflow_cycles([0, 3, 1, 2, 0])
    assert np.isclose(sum(n for r, _, n, _, _ in nested if r == 1), 1)
    assert np.isclose(sum(n for r, _, n, _, _ in nested if r == 3), 1)
    assert battery_wear([0.5, 0.5, 0.5]) == converter_wear([80, 80, 80]) == 0
    protected = simulate_interval([110, 104], 100, 40, 1, protection=True)
    assert protected["tripped"] and 0 < protected["active_seconds"] < 900
    assert np.isclose(protected["delivered_energy_kWh"], 100 * protected["active_seconds"] / 3600)
    hit = int(protected["active_seconds"])
    assert np.all(protected["power_kW"][hit:] == 0)
    initial_trip = simulate_interval([116, 100], 100, 40, protection=True)
    assert initial_trip["tripped"] and initial_trip["delivered_energy_kWh"] == 0
    charge = simulate_interval([30, 30], -40, 30)
    assert np.isclose(charge["delivered_energy_kWh"], -10)
    assert wear_cost(0.01, 0.02, 100, 200) == 5
    reports.update(status="PASS", nested_cycles=nested, basic_cycles=rainflow_cycles([0, 1, 0]),
                   trip_active_seconds=protected["active_seconds"], trip_peak=protected["peak_Tj"],
                   limitations="Trip timing is evaluated on the dt grid; synthetic life-consumption coefficients are uncalibrated")
    return reports


if __name__ == "__main__":
    report = run_checks()
    Path(__file__).with_name("plant_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
