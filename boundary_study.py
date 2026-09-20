import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from study import ROOT, episode

DATES = ['2017-01-01', '2017-04-01', '2017-07-01', '2017-10-01']
MODES = ['economic', 'rainflow', 'scalar', 'waveform', 'full_mean', 'full']
COOLING = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
OUTPUT = ROOT / 'runs/boundary'


def verify_frozen():
    frozen = json.loads((ROOT / 'frozen_main_manifest.json').read_text())['hashes']
    for name, expected in frozen.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    return frozen


def main():
    frozen = verify_frozen()
    primary_path = ROOT / 'runs/main/summary.csv'
    primary = pd.read_csv(primary_path)
    assert len(primary) == 384
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = {
        'status': 'EXPLORATORY_POSTPRIMARY; designed after inspecting the completed primary results',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'design_trigger': {'primary_episodes': len(primary), 'primary_trip_episodes': int(primary.trip.sum()),
                           'primary_max_Tj_C': float(primary.Tj_peak_C.max()),
                           'primary_summary_sha256': hashlib.sha256(primary_path.read_bytes()).hexdigest()},
        'dates': DATES, 'modes': MODES, 'regime': 'heatwave_dr', 'seed': 41,
        'external_plant_cooling': COOLING, 'episodes': 144, 'workers': 3,
        'controller_cooling': 1.0, 'other_parameters': 'All frozen main configuration and implementations retained',
        'perturbation': 'Purely synthetic positive multiplier on external-plant Rsa, interpretable only as an assumed fan/cooling-resistance perturbation; no measured fan failure or physical failure threshold is inferred.',
        'failure_policy': 'Retain thermal trips, solver-failure zero-action fallbacks, cut-limit returns, terminal-service errors, DR shortfall, raw trajectories and solver checks. No retuning or deletion of unfavorable cells.',
        'cost_definitions': {'wear_adjusted_cost': 'operating_cost_USD + DR_cost_USD + wear_cost_USD',
                             'energy_adjusted_cost': 'wear_adjusted_cost + terminal_correction_USD; correction does not restore unmet terminal service'},
        'source_hashes': frozen,
        'boundary_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / 'boundary_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    (ROOT / 'boundary_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    summaries = []
    jobs = [(cooling, date, mode) for cooling in COOLING for date in DATES for mode in MODES]
    with (OUTPUT / 'boundary.log').open('w', encoding='utf-8') as log, ProcessPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(episode, date, mode, 'heatwave_dr', 41,
                               str(OUTPUT / f'cooling_{round(cooling * 100)}'), {'cooling': cooling}):
                   (cooling, date, mode) for cooling, date, mode in jobs}
        for future in as_completed(pending):
            cooling, date, mode = pending[future]
            result = future.result()
            counts = result['solver_status_counts']
            result.update(cooling=cooling,
                          solver_fallback_intervals=sum(v for k, v in counts.items() if k not in ['optimal', 'cut_limit', 'latched_trip']),
                          cut_limit_intervals=counts.get('cut_limit', 0),
                          latched_trip_intervals=counts.get('latched_trip', 0))
            summaries.append(result)
            pd.DataFrame(summaries).sort_values(['cooling', 'date', 'mode']).to_csv(OUTPUT / 'summary.csv', index=False)
            record = {'completed': len(summaries), 'cooling': cooling, 'date': date, 'mode': mode,
                      'trip': result['trip'], 'fallback_intervals': result['solver_fallback_intervals'],
                      'terminal_SOC_error': result['terminal_SOC_error'],
                      'solver_status_counts': counts, 'elapsed_seconds': result['elapsed_seconds']}
            line = json.dumps(record)
            log.write(line + '\n')
            log.flush()
            if len(summaries) % 12 == 0:
                print(line, flush=True)
    frame = pd.DataFrame(summaries)
    assert len(frame) == 144 and not frame.duplicated(['cooling', 'date', 'mode']).any()
    rows = []
    for (cooling, mode), g in frame.groupby(['cooling', 'mode'], sort=True):
        rows.append({'cooling': cooling, 'mode': mode, 'episodes': len(g),
                     'trip_episodes': int(g.trip.sum()), 'solver_fallback_intervals': int(g.solver_fallback_intervals.sum()),
                     'episodes_with_solver_fallback': int(g.solver_fallback_intervals.gt(0).sum()),
                     'cut_limit_intervals': int(g.cut_limit_intervals.sum()),
                     'latched_trip_intervals': int(g.latched_trip_intervals.sum()),
                     'terminal_service_failed_episodes': int((~g.terminal_service_met).sum()),
                     'terminal_SOC_error_abs_mean': float(g.terminal_SOC_error.abs().mean()),
                     'terminal_SOC_error_abs_max': float(g.terminal_SOC_error.abs().max()),
                     'DR_shortfall_kWh_total': float(g.DR_shortfall_kWh.sum()),
                     'DR_shortfall_kWh_mean': float(g.DR_shortfall_kWh.mean()),
                     'operating_cost_USD_mean': float(g.operating_cost_USD.mean()),
                     'wear_cost_USD_mean': float(g.wear_cost_USD.mean()),
                     'wear_adjusted_cost_USD_mean': float(g.total_cost_USD.mean()),
                     'energy_adjusted_cost_USD_mean': float(g.energy_adjusted_cost_USD.mean()),
                     'Tj_peak_C_max': float(g.Tj_peak_C.max()), 'Tj_peak_C_mean': float(g.Tj_peak_C.mean()),
                     'thermal_exceedance_hours_total': float(g.thermal_exceedance_hours.sum())})
    aggregate = pd.DataFrame(rows)
    (ROOT / 'results').mkdir(exist_ok=True)
    aggregate.to_csv(ROOT / 'results/boundary_aggregate.csv', index=False)
    verify_frozen()
    print('BOUNDARY_COMPLETE', len(frame), 'TRIP_EPISODES', int(frame.trip.sum()),
          'FALLBACK_INTERVALS', int(frame.solver_fallback_intervals.sum()), flush=True)
    print(aggregate[['cooling', 'mode', 'trip_episodes', 'solver_fallback_intervals',
                     'terminal_service_failed_episodes', 'DR_shortfall_kWh_total', 'Tj_peak_C_max']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
