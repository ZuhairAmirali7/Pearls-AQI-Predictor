# `data/` directory

| Path | Committed? | Contents |
| ---- | ---------- | -------- |
| `data/sample/` | ✅ yes | Small, **synthetic** sample observations for offline demos/tests. |
| `data/local/` | ❌ gitignored | Local Parquet **feature store** (feature groups written by the pipelines). |
| `data/raw/` | ❌ gitignored | Optional raw provider payloads (when `store_raw_payloads: true`). |
| `data/processed/` | ❌ gitignored | Scratch space for ad-hoc processed exports. |

## ⚠️ `data/sample/` is synthetic

`data/sample/karachi_sample_observations.csv` (~168 hourly rows) is produced by
`pearls_aqi.data.sample.generate_sample_observations`. It is **deterministic
synthetic data** with realistic diurnal/seasonal structure, tagged
`data_source = sample`.

**It is NOT real air-quality data and must never be presented as real
observations.** Its only purpose is to make the project runnable, testable, and
demonstrable without any API key, account, or network access.

Regenerate / extend it with:

```bash
python scripts/seed_sample_data.py --city Karachi --days 180
```

## Real data

For real observations the pipelines use the free, keyless **Open-Meteo** APIs by
default (or OpenWeather with a key). Backfill real history with:

```bash
python scripts/backfill_historical_data.py --city Karachi \
  --start-date 2024-01-01 --end-date 2026-01-01
```

The local feature store lives under `data/local/feature_store/*.parquet`
(one file per feature group). See [`docs/data_dictionary.md`](../docs/data_dictionary.md).
