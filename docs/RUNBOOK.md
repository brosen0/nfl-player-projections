# Operational Runbook

Per Agent Directive V7, Section 16 — operational procedures for common tasks,
failure scenarios, and recovery steps.

---

## Weekly Operations

### Standard Weekly Workflow

Run this sequence every Tuesday–Saturday to prepare predictions for Sunday:

```bash
# 1. Refresh data (Tuesday, after stats are published)
python run_app.py --refresh --skip-data=false

# 2. Generate predictions for next week
python run_app.py --with-predictions

# 3. Start the web server for dashboard access
python run_app.py
```

Alternatively, run all steps at once:

```bash
python run_app.py --refresh --with-predictions
```

### Data Refresh Only

```bash
python -m src.scrapers.run_scrapers --refresh
```

### Predictions Only (skip data refresh)

```bash
python run_app.py --with-predictions --skip-data
```

### Force Regenerate Predictions

Use when predictions seem stale or after model retraining:

```bash
python run_app.py --force-predictions
```

---

## Model Training

### Retrain All Position Models

```bash
python -m src.models.train --positions QB RB WR TE
```

### Retrain a Single Position

```bash
python -m src.models.train --positions QB
```

### Where Models Are Stored

- Trained models: `data/models/*.joblib`
- Model metadata: `data/models/model_metadata.json`
- Version history: `data/models/model_version_history.json`

---

## Monitoring

### Check Model Health

Review monitoring alerts:

```bash
cat data/monitoring/alerts.jsonl | tail -20
```

Review monitoring metrics:

```bash
cat data/monitoring/metrics.jsonl | tail -20
```

### Key Indicators to Watch

| Indicator | Warning Threshold | Action |
|-----------|-------------------|--------|
| RMSE increase vs baseline | > 20% degradation | Investigate data quality, retrain |
| Prediction drift (mean shift) | > 2 std devs from historical | Check for data source changes |
| Feature drift (KS test) | p-value < 0.01 | Verify feature pipeline, check for schema changes |
| Confidence interval coverage | < 85% at 90% nominal | Recalibrate conformal residuals |
| Missing features in input | > 5% of features | Check data sources, review imputation |

### Experiment Tracking

Review recent experiments:

```bash
cat data/experiments/experiment_log.jsonl | tail -5
```

---

## Failure Scenarios and Recovery

### Scenario: Data Source Unavailable

**Symptoms:** `nfl-data-py` raises connection errors or returns empty data.

**Diagnosis:**
```bash
python -c "import nfl_data_py as nfl; print(nfl.import_weekly_data([2025]))"
```

**Recovery:**
1. Check if nflverse servers are down (check https://github.com/nflverse/nflverse-data)
2. If transient, retry after waiting 30 minutes
3. If prolonged, use cached data in `data/nfl_data.db` — predictions will use last available data
4. Do not retrain models on stale data

### Scenario: Model Files Missing or Corrupted

**Symptoms:** `FileNotFoundError` or `joblib.load` errors when predicting.

**Diagnosis:**
```bash
ls -la data/models/*.joblib
python -c "import joblib; joblib.load('data/models/<model_file>.joblib')"
```

**Recovery:**
1. Check `data/models/model_version_history.json` for the last successful training run
2. If backup models exist, restore from version history
3. If no backups, retrain: `python -m src.models.train --positions QB RB WR TE`
4. After retraining, verify predictions: `python -m src.predict --weeks 1`

### Scenario: API Server Won't Start

**Symptoms:** FastAPI/Uvicorn fails to bind or crashes on startup.

**Diagnosis:**
```bash
# Check if port is in use
lsof -i :8000

# Try starting manually with verbose logging
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

**Recovery:**
1. Kill any existing process on port 8000
2. Check that `data/nfl_data.db` exists and is not corrupted
3. Verify model files exist in `data/models/`
4. Check `requirements.txt` dependencies are installed

### Scenario: Predictions Look Wrong

**Symptoms:** Predictions are unreasonably high/low, all zeros, or identical across players.

**Diagnosis:**
1. Check if data was refreshed: `SELECT MAX(week) FROM player_weekly_stats` in `data/nfl_data.db`
2. Check model metadata: `cat data/models/model_metadata.json`
3. Run a quick sanity check: `python -m src.predict --player "Patrick Mahomes" --weeks 1`

**Recovery:**
1. Force data refresh: `python run_app.py --refresh`
2. Force prediction regeneration: `python run_app.py --force-predictions`
3. If still wrong, check monitoring alerts for drift: `cat data/monitoring/alerts.jsonl | tail -10`
4. If model has degraded, retrain: `python -m src.models.train --positions QB RB WR TE`

### Scenario: CI Tests Failing

**Symptoms:** GitHub Actions workflow fails.

**Diagnosis:**
```bash
# Run the specific failing test locally
python -m pytest tests/<failing_test>.py -v

# Run the rubric compliance check
python scripts/check_rubric_compliance.py
```

**Recovery:**
1. Read the test failure output carefully
2. If data-dependent test: ensure test fixtures are up to date
3. If leakage test: investigate recent feature changes for temporal contamination
4. If ML audit test: check model performance thresholds in `tests/test_ml_audit.py`
5. Fix the issue and verify locally before pushing

### Scenario: Database Locked

**Symptoms:** `sqlite3.OperationalError: database is locked`

**Diagnosis:**
```bash
# Check for processes holding the database
fuser data/nfl_data.db
```

**Recovery:**
1. Stop any running instances of the app or data loaders
2. If no processes found, the lock file may be stale — rename `data/nfl_data.db-journal` if it exists
3. Restart the application

---

## Docker Operations

### Build and Run

```bash
docker-compose up --build
```

### Run in Background

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Off-Season Operations

During the NFL off-season (typically February–August):

1. **Do not generate predictions** — the system shows OOS (Out-Of-Season) validation metrics instead
2. **Retrain models** before the new season starts using all available historical data
3. **Update data sources** — check if nfl-data-py has added new fields for the upcoming season
4. **Review and update** `config/settings.py` if scoring rules or position handling changes

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Main entry point | `run_app.py` |
| Configuration | `config/settings.py` |
| Database | `data/nfl_data.db` |
| Trained models | `data/models/*.joblib` |
| Model metadata | `data/models/model_metadata.json` |
| Experiment log | `data/experiments/experiment_log.jsonl` |
| Monitoring alerts | `data/monitoring/alerts.jsonl` |
| API server | `api/main.py` |
| CI workflow | `.github/workflows/rubric-compliance.yml` |
| Test suite | `tests/` |
