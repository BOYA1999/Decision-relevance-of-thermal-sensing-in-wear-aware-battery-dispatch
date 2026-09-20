from pathlib import Path
import datetime
import hashlib
import json
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
BASE = 'https://data.open-power-system-data.org/household_data/2020-04-15/'
URLS = {n: BASE + n for n in ['datapackage.json', 'README.md', 'household_data_60min_singleindex.csv']}
URLS.update({
    'processing_main.ipynb': 'https://raw.githubusercontent.com/isc-konstanz/household_data/2020-04-15/main.ipynb',
    'processing.ipynb': 'https://raw.githubusercontent.com/isc-konstanz/household_data/2020-04-15/processing.ipynb',
})
records_path = RAW / 'download_records.json'
previous = {r['filename']: r for r in json.loads(records_path.read_text())} if records_path.exists() else {}
records = []
for name, url in URLS.items():
    path = RAW / name
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        previous[name] = {'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    content = path.read_bytes()
    records.append({'filename': name, 'url': url, 'bytes': len(content),
                    'sha256': hashlib.sha256(content).hexdigest(),
                    'retrieved_utc': previous.get(name, {}).get('retrieved_utc')})
records_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
metadata = json.loads((RAW / 'datapackage.json').read_text())
source = pd.read_csv(RAW / 'household_data_60min_singleindex.csv')
source.index = pd.to_datetime(source.utc_timestamp, utc=True)
assert source.index.is_unique and source.index.is_monotonic_increasing
assert source.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
fields = ['DE_KN_residential4_' + s for s in ['grid_import', 'grid_export', 'pv']]
delta = source[fields].diff()
source_flag = source.interpolated.fillna('').apply(lambda s: any(f in s for f in fields))
out = pd.DataFrame(index=source.index)
out['grid_import_kWh'] = delta[fields[0]]
out['grid_export_kWh'] = delta[fields[1]]
out['PV_kWh'] = delta[fields[2]]
out['load_kWh'] = out.grid_import_kWh + out.PV_kWh - out.grid_export_kWh
out['missing_counter_endpoint'] = delta.isna().any(axis=1)
out['negative_counter_increment'] = delta.lt(0).any(axis=1)
out['negative_load_balance'] = out.load_kWh.lt(0)
out['source_interpolated_current_hour'] = source_flag
out['source_interpolated_either_endpoint_hour'] = source_flag | source_flag.shift(1, fill_value=False)
out['strict_usable_hour'] = ~out[['missing_counter_endpoint', 'negative_counter_increment',
                                'negative_load_balance', 'source_interpolated_either_endpoint_hour']].any(axis=1)
out = out.loc['2016-01-01':'2017-12-31'].copy()
day_groups = out.groupby(out.index.floor('D'))
days = day_groups.agg(hours=('load_kWh', 'size'), usable_hours=('strict_usable_hour', 'sum'),
                      source_interpolation_hours=('source_interpolated_either_endpoint_hour', 'sum'),
                      negative_load_hours=('negative_load_balance', 'sum'),
                      missing_endpoint_hours=('missing_counter_endpoint', 'sum'))
days['strict_usable_day'] = days.hours.eq(24) & days.usable_hours.eq(24)
out['strict_usable_day'] = out.index.floor('D').map(days.strict_usable_day)
out['period_role'] = ['training_calibration' if t.year == 2016 else 'test_candidate' for t in out.index]
out.index.name = 'utc_timestamp'
out.to_csv(DATA / 'household4_hourly_2016_2017.csv', float_format='%.6f', date_format='%Y-%m-%dT%H:%M:%SZ')
days.index.name = 'date_utc'
days.to_csv(DATA / 'daily_quality.csv', date_format='%Y-%m-%d')
days.loc[days.strict_usable_day].to_csv(DATA / 'strict_valid_days.csv', date_format='%Y-%m-%d')
selected = []
valid_days = days.index[days.strict_usable_day]
for month in range(1, 13):
    available = valid_days[(valid_days.year == 2017) & (valid_days.month == month)]
    if len(available):
        selected.append({'candidate': 'monthly_first_valid_day', 'month_or_quarter': month,
                         'start_date_utc': str(available[0].date()), 'end_date_utc': str(available[0].date()), 'days': 1})
for quarter, month in enumerate([1, 4, 7, 10], 1):
    for start in valid_days[(valid_days.year == 2017) & (valid_days.month >= month) & (valid_days.month <= month + 2)]:
        week = pd.date_range(start, periods=7, freq='D')
        if week[-1].month <= month + 2 and days.strict_usable_day.reindex(week, fill_value=False).all():
            selected.append({'candidate': 'quarter_first_valid_seven_day_block', 'month_or_quarter': quarter,
                             'start_date_utc': str(start.date()), 'end_date_utc': str(week[-1].date()), 'days': 7})
            break
pd.DataFrame(selected).to_csv(DATA / 'evaluation_candidates.csv', index=False)
summary = {}
for year in [2016, 2017]:
    h = out.loc[str(year)]
    d = days.loc[str(year)]
    summary[str(year)] = {'hours': len(h), 'strict_usable_hours': int(h.strict_usable_hour.sum()),
                         'strict_usable_days': int(d.strict_usable_day.sum()),
                         'negative_load_hours': int(h.negative_load_balance.sum()),
                         'missing_endpoint_hours': int(h.missing_counter_endpoint.sum()),
                         'negative_counter_increment_hours': int(h.negative_counter_increment.sum()),
                         'source_interpolation_endpoint_hours': int(h.source_interpolated_either_endpoint_hour.sum()),
                         'raw_load_mean_kWh': float(h.load_kWh.mean()), 'raw_PV_mean_kWh': float(h.PV_kWh.mean())}
manifest = {'package_title': metadata['title'], 'package_version': metadata['version'],
            'landing_page': BASE, 'source_organization': 'Open Power System Data; primary data: CoSSMic',
            'licenses': metadata['licenses'], 'files': records,
            'source_site': 'DE_KN_residential4, one urban residential building in Konstanz, Germany',
            'source_fields': [f for f in metadata['schemas']['60min']['fields'] if f['name'] in fields],
            'source_resolution': '60-minute sampled cumulative energy in kWh; upstream uses hourly .last()',
            'derived_period': ['2016-01-01T00:00:00Z', '2017-12-31T23:00:00Z'],
            'transformation': 'Adjacent cumulative counter difference; load_kWh = delta(grid_import) + delta(pv) - delta(grid_export). No added imputation, clipping or scaling.',
            'interpretation': 'Electrical balance load proxy for a stylized simulation. Not verified gross demand, a measured park, independent health labels, real tariffs or measured temperature.',
            'quality_rule': 'A strict day has 24 usable hours, no missing counter endpoints, negative increments, negative load balance, or selected-field upstream interpolation flag in either differenced endpoint hour.',
            'candidate_selection': 'First strict day per 2017 UTC month and first complete strict 7-day block within each 2017 calendar quarter; selected solely by time and quality, not controller outcomes.',
            'summary': summary,
            'attribution': 'Open Power System Data (2020). Data Package Household Data, version 2020-04-15. https://data.open-power-system-data.org/household_data/2020-04-15/. Primary data: CoSSMic. CC BY 4.0. Derived by hourly differencing and electrical balance; quality flags added.'}
(DATA / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
print(pd.DataFrame(selected).to_string(index=False))
