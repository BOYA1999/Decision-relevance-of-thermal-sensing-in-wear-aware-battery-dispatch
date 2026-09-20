# Data provenance

The packaged public input is:

Open Power System Data (2020), *Data Package Household Data*, version 2020-04-15. Primary data: CoSSMic. Landing page: https://data.open-power-system-data.org/household_data/2020-04-15/. License: CC BY 4.0.

The study uses the `DE_KN_residential4` grid-import, grid-export, and PV cumulative counters. The derived hourly table is produced by adjacent counter differences and the electrical-balance proxy

`load_kWh = delta(grid_import) + delta(PV) - delta(grid_export)`.

Quality flags are retained. The derived load is a stylized electrical input for simulation; it is not a verified gross-demand channel, site measurement, or health record. The package contains no author-controlled data or personal information.
