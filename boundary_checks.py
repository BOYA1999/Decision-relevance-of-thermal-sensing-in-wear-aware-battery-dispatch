import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from plant import battery_wear, converter_wear
from study import ROOT


def main():
    output = ROOT / 'runs/boundary'
    contract = json.loads((output / 'boundary_contract.json').read_text())
    cfg = json.loads((ROOT / 'config.json').read_text())
    summary = pd.read_csv(output / 'summary.csv')
    expected = {(c, d, m) for c in contract['external_plant_cooling'] for d in contract['dates'] for m in contract['modes']}
    observed = set(zip(summary.cooling, summary.date, summary['mode']))
    assert len(summary) == contract['episodes'] and expected == observed
    assert not summary.duplicated(['cooling', 'date', 'mode']).any()
    checked_files = 0
    for row in summary.itertuples():
        folder = output / f'cooling_{round(row.cooling * 100)}' / f'{row.date}_{row.regime}_{row.mode}_{row.seed}'
        table = pd.read_csv(folder / 'intervals.csv')
        metrics = json.loads((folder / 'metrics.json').read_text())
        solver = json.loads((folder / 'solver_checks.json').read_text())
        with np.load(folder / 'physical_traces.npz') as traces:
            junction, soc = traces['Tj'], traces['SOC']
        assert len(table) == len(solver) == 96 and len(junction) == 86401 and len(soc) == 97
        assert table.interval.tolist() == list(range(96))
        assert np.isfinite(junction).all() and np.isfinite(soc).all()
        assert np.allclose(soc[1:] * cfg['mpc']['energy_capacity'], table.energy_kWh, atol=1e-9)
        peaks = [junction[i * 900:(i + 1) * 900 + 1].max() for i in range(96)]
        assert np.allclose(peaks, table.Tj_peak_C, atol=1e-9)
        assert abs(junction.max() - row.Tj_peak_C) < 1e-9
        assert abs(soc[-1] - row.terminal_SOC) < 1e-12
        assert abs(soc[-1] - .5 - row.terminal_SOC_error) < 1e-12
        assert bool(abs(soc[-1] - .5) < 1e-6) == bool(row.terminal_service_met)
        assert int(table.trip.any()) == row.trip
        assert Counter(table.solver_status) == Counter(x['status'] for x in solver) == Counter(metrics['solver_status_counts'])
        statuses = Counter(table.solver_status)
        fallback = sum(n for status, n in statuses.items() if status not in ['optimal', 'cut_limit', 'latched_trip'])
        assert fallback == row.solver_fallback_intervals
        assert statuses.get('cut_limit', 0) == row.cut_limit_intervals
        assert abs(table.operating_cost_USD.sum() - row.operating_cost_USD) < 1e-9
        assert abs(table.DR_shortfall_kWh.sum() - row.DR_shortfall_kWh) < 1e-9
        assert abs(np.sum(junction[1:] > 110) / 3600 - row.thermal_exceedance_hours) < 1e-9
        bw, cw = battery_wear(soc), converter_wear(junction)
        wear = 250 * cfg['mpc']['energy_capacity'] * bw + cfg['converter_capex'] * cw
        assert abs(bw - row.battery_wear) < 1e-12 and abs(cw - row.converter_wear) < 1e-12
        assert abs(wear - row.wear_cost_USD) < 1e-9
        assert abs(row.total_cost_USD - row.operating_cost_USD - row.DR_cost_USD - wear) < 1e-9
        assert abs(row.energy_adjusted_cost_USD - row.total_cost_USD - row.terminal_correction_USD) < 1e-9
        assert metrics['overrides'] == {'cooling': row.cooling}
        checked_files += 4
    aggregate = pd.read_csv(ROOT / 'results/boundary_aggregate.csv').set_index(['cooling', 'mode'])
    for group, g in summary.groupby(['cooling', 'mode']):
        a = aggregate.loc[group]
        checks = {'episodes': len(g), 'trip_episodes': g.trip.sum(),
                  'solver_fallback_intervals': g.solver_fallback_intervals.sum(),
                  'episodes_with_solver_fallback': g.solver_fallback_intervals.gt(0).sum(),
                  'cut_limit_intervals': g.cut_limit_intervals.sum(), 'latched_trip_intervals': g.latched_trip_intervals.sum(),
                  'terminal_service_failed_episodes': (~g.terminal_service_met).sum(),
                  'terminal_SOC_error_abs_mean': g.terminal_SOC_error.abs().mean(),
                  'terminal_SOC_error_abs_max': g.terminal_SOC_error.abs().max(),
                  'DR_shortfall_kWh_total': g.DR_shortfall_kWh.sum(), 'DR_shortfall_kWh_mean': g.DR_shortfall_kWh.mean(),
                  'operating_cost_USD_mean': g.operating_cost_USD.mean(), 'wear_cost_USD_mean': g.wear_cost_USD.mean(),
                  'wear_adjusted_cost_USD_mean': g.total_cost_USD.mean(),
                  'energy_adjusted_cost_USD_mean': g.energy_adjusted_cost_USD.mean(),
                  'Tj_peak_C_max': g.Tj_peak_C.max(), 'Tj_peak_C_mean': g.Tj_peak_C.mean(),
                  'thermal_exceedance_hours_total': g.thermal_exceedance_hours.sum()}
        assert all(abs(float(a[k]) - float(v)) < 1e-9 for k, v in checks.items())
    primary = pd.read_csv(ROOT / 'runs/main/summary.csv')
    replay = summary.loc[summary.cooling.eq(1)].merge(primary, on=['date', 'mode', 'regime', 'seed'], suffixes=('_boundary', '_primary'))
    for name in ['operating_cost_USD', 'DR_shortfall_kWh', 'Tj_peak_C', 'energy_adjusted_cost_USD', 'terminal_SOC_error']:
        assert (replay[name + '_boundary'] - replay[name + '_primary']).abs().max() < 1e-9
    for name, expected_hash in contract['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name
    report = {'status': 'PASS', 'episodes_checked': len(summary), 'episode_files_checked': checked_files,
              'Tj_samples_per_episode': 86401, 'SOC_samples_per_episode': 97,
              'aggregate_rows_checked': len(aggregate), 'baseline_primary_replay_episodes': len(replay),
              'frozen_hash_mismatches': 0, 'trip_episodes': int(summary.trip.sum()),
              'solver_fallback_intervals': int(summary.solver_fallback_intervals.sum()),
              'terminal_service_failed_episodes': int((~summary.terminal_service_met).sum()),
              'maximum_Tj_C': float(summary.Tj_peak_C.max()),
              'scope': 'Exploratory post-primary synthetic perturbation, not physical threshold or safety validation.'}
    (ROOT / 'results/boundary_qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
