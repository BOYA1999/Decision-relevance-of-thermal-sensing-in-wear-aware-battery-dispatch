import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from plant import battery_wear, converter_wear, power_loss, simulate_interval

ROOT = Path(__file__).resolve().parent


def replay(folder, write_traces=False):
    folder = Path(folder).resolve()
    folder = folder.parent if folder.suffix.lower() == '.csv' else folder
    table = pd.read_csv(folder / 'intervals.csv')
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    settings = {**cfg['mpc'], **metrics.get('overrides', {})}
    capacity, cooling = settings['energy_capacity'], settings.get('cooling', 1.0)
    assert len(table) == 96 and table.interval.tolist() == list(range(96))
    state = np.array([table.ambient_C.iloc[0] + 5.2, table.ambient_C.iloc[0] + 4])
    energy, junction, soc, latched, residuals = capacity / 2, [state[0]], [.5], False, {}
    operating, shortfall_total, exceedance = 0.0, 0.0, 0

    def check(name, actual, expected, tolerance=1e-7):
        error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
        residuals[name] = max(residuals.get(name, 0), error)
        assert np.isfinite(error) and error <= tolerance, (folder.name, name, error, tolerance)

    for row in table.itertuples():
        if latched:
            check('latched_action_kW', row.action_kW, 0)
        check('Tj_start_C', row.Tj_start_C, state[0])
        check('Ts_start_C', row.Ts_start_C, state[1])
        event = simulate_interval(state, row.action_kW, row.ambient_C, row.ripple,
                                  cooling=cooling, protection=True)
        power = event['power_kW']
        loss = power_loss(power, row.ripple)
        net = row.load_kW - row.PV_kW
        grid = net - power + loss
        buy = .12 if row.hour_utc < 6 else .32 if 17 <= row.hour_utc < 21 else .20
        limit = .6 * max(net + .2, 0) if metrics['regime'] == 'heatwave_dr' and 17 <= row.hour_utc < 18.5 else 1e4
        cost = float((buy * np.maximum(grid, 0) - .04 * np.maximum(-grid, 0)).sum() / 3600)
        shortfall = float(np.maximum(grid - limit, 0).sum() / 3600)
        energy += float((np.maximum(-power, 0) * .98 - np.maximum(power, 0) / .98).sum() / 3600)
        over = int(np.sum(event['Tj'][1:] > 110))
        expected = {'executed_mean_kW': power.mean(), 'energy_kWh': energy,
                    'operating_cost_USD': cost, 'DR_shortfall_kWh': shortfall,
                    'grid_energy_kWh': grid.sum() / 3600, 'loss_energy_kWh': loss.sum() / 3600,
                    'Tj_peak_C': event['peak_Tj'], 'thermal_exceedance_seconds': over,
                    'trip': int(event['tripped'])}
        for name, value in expected.items():
            check(name, getattr(row, name), value)
        state = event['state']
        junction.extend(event['Tj'][1:])
        soc.append(energy / capacity)
        latched = latched or event['tripped']
        operating += cost
        shortfall_total += shortfall
        exceedance += over
    junction, soc = np.asarray(junction), np.asarray(soc)
    bw, cw = battery_wear(soc), converter_wear(junction)
    wear_cost = 250 * capacity * bw + cfg['converter_capex'] * cw
    deficit = capacity / 2 - energy
    terminal = max(deficit, 0) / .98 * .20 - max(-deficit, 0) * .98 * .04
    dr_cost = shortfall_total * settings['dr_penalty']
    total = operating + dr_cost + wear_cost
    expected = {'operating_cost_USD': operating, 'DR_cost_USD': dr_cost, 'battery_wear': bw,
                'converter_wear': cw, 'wear_cost_USD': wear_cost, 'terminal_correction_USD': terminal,
                'total_cost_USD': total, 'energy_adjusted_cost_USD': total + terminal,
                'terminal_SOC': energy / capacity, 'terminal_SOC_error': energy / capacity - .5,
                'trip': int(latched), 'thermal_exceedance_hours': exceedance / 3600,
                'DR_shortfall_kWh': shortfall_total, 'Tj_peak_C': junction.max(),
                'FEC': np.abs(np.diff(soc)).sum() / 2,
                'Tj_RMSE_C': np.sqrt(np.mean((table.Tj_estimate_C.to_numpy() - junction[::900][:-1]) ** 2))}
    for name, value in expected.items():
        check('metric_' + name, metrics[name], value, 1e-10 if name.endswith('_wear') else 1e-7)
    assert metrics['terminal_service_met'] == bool(abs(soc[-1] - .5) < 1e-6)
    assert Counter(table.solver_status) == Counter(metrics['solver_status_counts'])
    if (folder / 'solver_checks.json').exists():
        stored = json.loads((folder / 'solver_checks.json').read_text(encoding='utf-8'))
        assert [r['status'] for r in stored] == table.solver_status.tolist()
    if write_traces:
        np.savez_compressed(folder / 'physical_traces.npz', Tj=junction, SOC=soc)
    return {'episode': folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else folder.name,
            'physics_and_ledger_status': 'PASS', 'Tj_samples': len(junction), 'SOC_samples': len(soc),
            'original_numerical_gate': metrics['numerical_gate'],
            'original_solver_status_counts': metrics['solver_status_counts'],
            'traces_written': write_traces, 'absolute_residuals': residuals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--write-traces', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    folders = sorted(p.parent for p in args.episode.rglob('intervals.csv')) if args.recursive else [args.episode]
    assert folders, 'No episode interval files found'
    reports = [replay(folder, args.write_traces) for folder in folders]
    report = {'status': 'PASS', 'episodes': len(reports), 'MPC_rerun': False,
              'claim': 'Replay checks the recorded actions against the frozen synthetic plant and wear ledger; original optimizer gates are unchanged.',
              'records': reports}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'episodes': len(reports), 'original_numerical_gates': dict(Counter(r['original_numerical_gate'] for r in reports)),
                      'maximum_absolute_residual': max(max(r['absolute_residuals'].values()) for r in reports)}, indent=2))
