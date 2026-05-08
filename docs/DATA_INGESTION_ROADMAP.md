# Data Ingestion Roadmap

Prioritized list of data sources to ingest for model accuracy improvement.
See also: memory file `reference_future_data_sources.md` for detailed notes.

## Status Key
- ✅ Done
- 🔧 In DB, needs wiring to features
- ❌ Not ingested

## Tier 1: High Impact

| # | Source | Status | Impact | Notes |
|---|--------|--------|--------|-------|
| 1 | Seasonal Rosters (age, height, weight) | ❌ | Age features everywhere | `nfl.import_seasonal_rosters()` |
| 2 | Weekly PFR (drops, pressures, bad throws) | ❌ | Strongest QB signal | `nfl.import_weekly_pfr('pass')` |
| 3 | Contracts (APY, guaranteed, contract year) | 🔧 | Contract year boost | `contracts` table, 51K rows |
| 4 | Depth Charts (official WR1/RB1 designation) | 🔧 | Preseason role clarity | `depth_charts` table, 591K rows |
| 5 | FTN Charting (formation, motion, RPO) | ❌ | Scheme tendencies | `nfl.import_ftn_data()`, 48K plays/yr |

## Tier 2: Medium Impact

| # | Source | Status | Impact | Notes |
|---|--------|--------|--------|-------|
| 6 | Seasonal PFR (season-level advanced) | ❌ | Preseason summary stats | `nfl.import_seasonal_pfr()` |
| 7 | Combine (40, bench, vertical) | 🔧 | Rookie athleticism | `combine_data_v2` table, 8.9K rows |
| 8 | Win Totals (Vegas preseason) | ❌ | Team quality proxy | Source currently broken |
| 9 | Officials (referee tendencies) | ❌ | Low signal | `nfl.import_officials()` |

## Tier 3: External (needs new pipelines)

| # | Source | Status | Impact | Notes |
|---|--------|--------|--------|-------|
| 10 | Training camp reports | ❌ | Fills ADP gap | NLP pipeline needed |
| 11 | College stats | ❌ | Better rookie model | Different ID systems |
| 12 | Coaching staff / OC history | ❌ | Scheme change signal | Manual curation |
| 13 | Social media sentiment | ❌ | Noisy, low priority | API access needed |

## Already Ingested

| Source | Coverage | Feature Version |
|--------|----------|-----------------|
| Weekly player stats | 2006-2025 | v1+ |
| PBP EPA/WPA/success | 2006-2025 (91%) | v14 |
| NGS (separation, CPOE, RYOE) | 2018-2025 | v9+ |
| Vegas lines | 2006-2025 | v9+ |
| ADP/ECR | 2024-2026 | v9+ |
| Draft picks | 1980-2026 | Draft advisor |
| Snap counts | 2013-2025 | v12+ |
| Injuries | 2018-2025 | v9+ |
| Schedule | 2006-2025 | v9+ |
