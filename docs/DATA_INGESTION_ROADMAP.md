# Data Ingestion Roadmap

Prioritized list of data sources to ingest for model accuracy improvement.
See also: memory file `reference_future_data_sources.md` for detailed notes.

## Status Key
- ✅ Ingested + wired to features
- 📦 Ingested, not yet in features
- ❌ Not ingested

## Tier 1: High Impact

| # | Source | Status | Features Added | Notes |
|---|--------|--------|----------------|-------|
| 1 | Seasonal Rosters (age, height, weight) | ✅ | `age_curve` (from birth_date) | 2,624 players, 91% coverage |
| 2 | Weekly PFR (drops, pressures, bad throws) | 📦 | In `weekly_pfr` table (59K rows) | Ready for feature promotion |
| 3 | Contracts (APY, guaranteed, contract year) | ✅ | `is_contract_year`, `contract_apy_rank` | 51K rows, 91% GSIS match |
| 4 | Depth Charts (official WR1/RB1 designation) | ✅ | `depth_chart_rank` | 2020-2024, 84K rows |
| 5 | FTN Charting (formation, motion, RPO) | ✅ | `team_motion_rate`, `team_play_action_rate` | 2022-2025, 2.3K team-weeks |

## Tier 2: Medium Impact

| # | Source | Status | Features Added | Notes |
|---|--------|--------|----------------|-------|
| 6 | Seasonal PFR (season-level advanced) | 📦 | In `seasonal_pfr` table (5.9K rows) | Ready for feature promotion |
| 7 | Combine (40, bench, vertical) | ✅ | `speed_score` | 1,607 matched via parquet, 58% rate |
| 8 | Draft Values (trade value chart) | 📦 | In `draft_values` table (262 rows) | Could weight draft capital |
| 9 | Officials (referee tendencies) | 📦 | In `officials` table (15.6K rows) | 3.4 pt range — too low signal for feature |

## Tier 3: External (needs new pipelines)

| # | Source | Status | Impact | Notes |
|---|--------|--------|--------|-------|
| 10 | Training camp reports | ❌ | Fills ADP gap | NLP pipeline needed |
| 11 | College stats | ❌ | Better rookie model | Different ID systems |
| 12 | Coaching staff / OC history | ❌ | Scheme change signal | Manual curation |
| 13 | Social media sentiment | ❌ | Noisy, low priority | API access needed |

## Already Ingested (core pipeline)

| Source | Coverage | Feature Version |
|--------|----------|-----------------|
| Weekly player stats | 2006-2025 (110K rows) | v1+ |
| PBP EPA/WPA/success | 2006-2025 (91%) | v14 |
| NGS (separation, CPOE, RYOE) | 2018-2025 | v9+ |
| Vegas lines | 2006-2025 | v9+ |
| ADP/ECR | 2024-2026 | v9+ |
| Draft picks | 1980-2026 | Draft advisor |
| Snap counts | 2013-2025 | v12+ |
| Injuries | 2018-2025 | v9+ |
| Schedule | 2006-2025 | v9+ |

## Feature Version History

- **v14** (current): 28 features/position — PBP EPA for RB/WR/TE, late-season momentum, age_curve from birth_date, team_changed, availability_3yr, career_year_flag, bayesian_prior_ppg, is_contract_year, contract_apy_rank, depth_chart_rank, speed_score, team_motion_rate, team_play_action_rate, fp_late6_vs_season
- **v13**: Added age_curve, team_changed, availability_3yr, career_year_flag, bayesian_prior_ppg
- **v12**: QB HistGBR with NaN-aware PBP features
- **v9**: snap_share_pct restored, NGS features added
