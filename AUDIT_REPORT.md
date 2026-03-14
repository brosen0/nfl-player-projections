# NFL Player Projections — Codebase Audit Report

**Date:** 2026-03-14
**Scope:** Full codebase audit covering architecture, code quality, security, testing, and deployment

---

## Executive Summary

The NFL Player Projections system is a Python-based fantasy football prediction platform with ~64,400 lines of Python across 84 source modules and 49 test files. It uses machine learning (XGBoost, LightGBM, PyTorch) to project NFL player fantasy performance, served via a FastAPI backend with a static HTML/JS frontend.

The project has a solid foundation with well-structured modules, meaningful tests, and thoughtful ML engineering (data leakage guards, uncertainty quantification, backtesting). However, the audit identified **3 critical**, **4 high**, **6 medium**, and **5 low** severity findings that should be addressed before production deployment.

| Category | Grade | Summary |
|----------|-------|---------|
| Architecture | B | Good module separation; some oversized files |
| Security | C | SQL injection patterns, permissive CORS, weak defaults |
| Code Quality | C+ | Heavy print usage, generic exception handling |
| Testing | B- | 40% module coverage; existing tests are high quality |
| CI/CD | B | Secret scanning + rubric compliance; no automated test gate enforcement |
| Deployment | B- | Docker + health check; weak credential defaults |

---

## 1. Architecture

### 1.1 Project Structure

```
src/
├── data/          # Data loading, validation, entity resolution (14 modules)
├── models/        # ML training, ensemble, position-specific models (14 modules)
├── features/      # Feature engineering, utilization scoring (12 modules)
├── evaluation/    # Metrics, backtesting, A/B testing, monitoring (13 modules)
├── integrations/  # ESPN, Sleeper, Yahoo fantasy platform connectors (3 modules)
├── governance/    # Approval gates, conflict resolution (2 modules)
├── deployment/    # Staged deployment logic (1 module)
├── optimization/  # Lineup optimizer (1 module)
├── scrapers/      # Schedule scraper (3 modules)
├── research/      # Auto-experiment framework (1 module)
├── utils/         # Database, helpers, retry, leakage guards (7 modules)
├── pipeline.py    # Orchestration
├── predict.py     # Prediction entry point
└── app_data.py    # Application data layer
```

**Strengths:**
- Clear separation of concerns across 12 subdirectories
- Dedicated data quality modules (schema_validator, quality_gates, lineage)
- Governance layer with approval gates and conflict resolution
- Entity resolver for cross-source data normalization

**Concerns:**
- Several oversized modules (see §3.2)
- `app.js` is a 1,243-line monolith handling all frontend logic
- 13 documentation/status files in the project root create clutter

### 1.2 Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost 2.0, LightGBM 4.3, PyTorch ≥2.1, scikit-learn 1.4 |
| Hyperparameter Tuning | Optuna 3.5 |
| Data | pandas 2.1, NumPy 1.26, nfl-data-py |
| API | FastAPI 0.109, Uvicorn 0.27, Gunicorn 21.2 |
| Database | SQLite (default), PostgreSQL (optional) |
| Frontend | Vanilla HTML/CSS/JS, Streamlit (alternate) |
| Deployment | Docker, docker-compose |
| Testing | pytest 7.4, pytest-cov |

### 1.3 Data Flow

```
nfl-data-py API → nfl_data_loader.py → schema_validator → quality_gates
    → entity_resolver → feature_engineering → train.py → ensemble.py
        → production_model.py → predict.py → FastAPI → Frontend
```

Data lineage tracking is implemented via `src/data/lineage.py` with bronze/silver/gold artifact tiers.

---

## 2. Security Findings

### CRITICAL

#### S-1: SQL Injection via String Interpolation
- **Location:** `src/utils/database.py:130, 190`
- **Issue:** Column names and DDL types are interpolated directly into ALTER TABLE statements using f-strings.
- **Risk:** If column/type values ever originate from external input or configuration, arbitrary SQL can be injected. SQLite does not support parameterized DDL, but a whitelist check is absent.
- **Recommendation:** Validate column names against an explicit allowlist before interpolation.

#### S-2: SQL Injection in Performance Tracker
- **Location:** `scripts/performance_tracker.py:32, 102-114`
- **Issue:** Table names are interpolated into CREATE TABLE and SELECT statements via f-strings with no validation.
- **Recommendation:** Use a constant or validate against allowed table names.

#### S-3: Overly Permissive CORS
- **Location:** `api/main.py:72-78`
- **Issue:** CORS allows all origins (`*`) with `allow_credentials=True`, all methods, and all headers.
- **Risk:** Any website can make authenticated cross-origin requests to the API, enabling CSRF-style attacks.
- **Recommendation:** Restrict `allow_origins` to known frontend domains. If credentials are not needed, set `allow_credentials=False`.

### HIGH

#### S-4: Regex DoS (ReDoS) in API Search
- **Location:** `api/main.py:504`
- **Issue:** The `name` query parameter is passed to `str.contains()` which performs regex matching. A malicious regex pattern could cause catastrophic backtracking.
- **Recommendation:** Use `regex=False` in `str.contains()` or escape special characters with `re.escape()`.

#### S-5: Weak Default Database Password
- **Location:** `docker-compose.yml:21`
- **Issue:** PostgreSQL password defaults to `changeme` when `DB_PASSWORD` is not set.
- **Recommendation:** Remove the fallback value; require explicit `DB_PASSWORD` configuration.

#### S-6: Unsafe Deserialization (Model Loading)
- **Location:** `src/models/ensemble.py:36, 149, 168` and other model loading sites
- **Issue:** `joblib.load()` uses pickle under the hood. Deserialization warnings are suppressed rather than addressed.
- **Risk:** If model files are tampered with or sourced from untrusted locations, arbitrary code execution is possible.
- **Recommendation:** Add SHA256 integrity checks for model files. Consider ONNX for safer serialization.

#### S-7: Missing API Rate Limiting
- **Location:** `api/main.py` (entire file)
- **Issue:** No rate limiting middleware is configured on any endpoint.
- **Recommendation:** Add `slowapi` or similar rate limiting to prevent abuse.

### MEDIUM

#### S-8: No API Input Range Validation
- **Location:** `api/main.py:328, 355, 365`
- **Issue:** Season path parameters (`int`) have no range validation (e.g., 2006–current year).

#### S-9: Fantasy Platform Credential Handling
- **Location:** `src/integrations/yahoo_fantasy.py:26-27`
- **Issue:** Constructor accepts credentials as parameters with empty-string defaults instead of enforcing environment variable loading.

#### S-10: Docker Image Copies Entire Directory
- **Location:** `Dockerfile:19`
- **Issue:** `COPY . .` copies all files including `.env.example`, documentation, and test files into the production image.
- **Recommendation:** Use `.dockerignore` to exclude non-production files and multi-stage builds.

### POSITIVE FINDINGS
- No `os.system()`, `subprocess.shell=True`, `eval()`, or `exec()` usage detected
- No `.env` files committed; proper `.gitignore` rules in place
- Secret scanning CI workflow active (`scripts/scan_secrets.py`)
- SSL certificate handling configured for data fetching

---

## 3. Code Quality

### 3.1 Logging vs Print Statements

| Metric | Count |
|--------|-------|
| `print()` statements | **1,446** across 73 files |
| `logging` imports | 165 across 29 files |

**Top offenders:**
| File | Print Count |
|------|-------------|
| `src/models/train.py` | ~120 |
| `src/data/nfl_data_loader.py` | ~60 |
| `run_app.py` | ~38 |

**Impact:** Production debugging is severely hampered. All output goes to stdout with no log levels, rotation, or structured formatting.

**Recommendation:** Replace `print()` with `logging.info()`/`warning()` and configure centralized logging with rotation in `run_app.py`.

### 3.2 Module Size

| File | Lines | Recommendation |
|------|-------|----------------|
| `src/models/train.py` | 2,299 | Split into train_core, train_pipeline, feature_prep |
| `src/features/feature_engineering.py` | 2,054 | Split into base, rolling, interaction features |
| `src/features/advanced_analytics.py` | 1,495 | Extract sentiment, coaching, suspension modules |
| `src/features/advanced_rookie_injury.py` | 1,432 | Extract rookie and injury into separate modules |
| `src/data/nfl_data_loader.py` | 1,208 | Extract PBP fallback logic |
| `app.js` | 1,243 | Extract state management and rendering |

### 3.3 Exception Handling

**181 generic exception handlers** (`except Exception:` or bare `except:`) found across the codebase. Many silently return `None` or empty defaults, masking real errors.

**Example (src/evaluation/explainability.py):**
```python
try:
    shap_values = explainer(X)
except Exception:
    return None  # Silent failure — no logging, no context
```

**Recommendation:** Catch specific exceptions (ValueError, KeyError, ImportError). Log the exception before returning defaults.

### 3.4 Positive Code Quality Patterns
- Data leakage guards (`src/utils/leakage.py`) with temporal validation
- Retry decorator with exponential backoff (`src/utils/retry.py`)
- Feature stability analysis before model training
- Policy-based missing data handling (`src/features/feature_policy_registry.py`)
- Entity resolution for cross-source joins (`src/data/entity_resolver.py`)
- Uncertainty quantification in predictions (floor/ceiling bounds)
- Configuration centralized in `config/settings.py`

---

## 4. Testing

### 4.1 Coverage Overview

| Metric | Value |
|--------|-------|
| Test files | 49 |
| Test methods | ~603 |
| Total assertions | ~1,111 |
| Assertions per test | 1.84 (average) |
| Module coverage | **40%** (48 of ~120 modules have tests) |
| Skipped tests | 6+ (requires trained models or network) |

### 4.2 Well-Tested Areas
- **Data leakage prevention** — 15 tests with concrete value checks (gold standard)
- **Injury modeling** — 29 tests across 6 test classes
- **Database operations** — 24+ tests with CRUD and edge cases
- **Feature engineering** — 12 tests covering rolling/lag/position features
- **ML audit** — 42+ tests covering 7 audit phases
- **Schema validation** — Comprehensive validation tests

### 4.3 Critical Test Gaps

| Module | Status | Risk |
|--------|--------|------|
| `models/ensemble.py` | **No tests** | Core prediction aggregation |
| `models/train.py` | **No tests** | Training pipeline |
| `models/position_models.py` | **No tests** | Position-specific modeling |
| `models/horizon_models.py` | **No tests** | Multi-horizon predictions |
| `evaluation/metrics.py` | **No tests** | Evaluation methodology |
| `evaluation/backtester.py` | **Minimal** | Backtesting accuracy |
| `data/nfl_data_loader.py` | **Wrappers only** | Data ingestion pipeline |
| `data/pbp_stats_aggregator.py` | **Partial** | Play-by-play transformation |
| `integrations/*` | **No tests** | ESPN, Sleeper, Yahoo connectors |
| `optimization/lineup_optimizer.py` | **No tests** | Lineup optimization |
| `scrapers/*` | **No tests** | Data collection |

### 4.4 Test Quality Assessment
- **Strengths:** Tests contain genuine business logic assertions, not stubs. Synthetic data generation with controlled randomness (seed=42). Proper tempfile usage for database tests.
- **Weaknesses:** Only 33% of test files use pytest fixtures. Many tests create data inline, reducing reusability. Several tests conditionally skip when data is unavailable.

---

## 5. CI/CD & Deployment

### 5.1 GitHub Actions Workflows

| Workflow | Purpose | Issues |
|----------|---------|--------|
| `secret-scan.yml` | Scans for leaked secrets on push/PR | Runs custom scanner, not a standard tool (e.g., trufflehog) |
| `rubric-compliance.yml` | Runs compliance checker + tests | Test failures silenced with `2>/dev/null \|\| echo` — failures don't block merges |
| `refresh-static-api.yml` | Refreshes static API data | N/A |
| `jekyll-gh-pages.yml` | Deploys GitHub Pages | N/A |

### 5.2 CI Issues

- **Test failures are swallowed:** Both the temporal integrity tests and coverage gate steps redirect stderr to `/dev/null` and use `|| echo` to prevent pipeline failure. This means broken tests never block PRs.
- **Coverage threshold:** Set at 50% (`--cov-fail-under=50`) but the gate is non-blocking due to `|| echo` fallback.
- **No linting:** No flake8, ruff, mypy, or black/isort checks in CI.
- **No dependency vulnerability scanning:** No Dependabot, Snyk, or pip-audit configured.

### 5.3 Docker

- **Strengths:** Health check endpoint configured; slim base image; proper `PORT` env handling.
- **Issues:**
  - No `.dockerignore` file — copies tests, docs, `.git` into image
  - No multi-stage build — dev dependencies (pytest, matplotlib) included in production image
  - No non-root user — container runs as root by default

---

## 6. Dependency Analysis

### 6.1 Version Pinning
Most dependencies are pinned to exact versions (good for reproducibility). Some use `>=` (torch, shap, statsmodels, scipy, streamlit, plotly, nfl-data-py) which could introduce breaking changes.

### 6.2 Notable Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| torch | ≥2.1.0 | Large dependency (~2GB); used for LSTM+ARIMA hybrid models |
| xgboost | 2.0.3 | Primary model framework |
| lightgbm | 4.3.0 | Secondary model framework |
| shap | ≥0.43.0 | Explainability; version unpinned |
| nfl-data-py | ≥0.3.0 | Primary data source; version unpinned |

### 6.3 Missing from Dependencies
- No dependency vulnerability scanner (pip-audit, safety)
- No lock file (pip-compile, poetry.lock) for deterministic installs

---

## 7. Recommendations by Priority

### Critical (Address Immediately)

1. **Fix SQL injection patterns** in `src/utils/database.py` and `scripts/performance_tracker.py` — add column/table name whitelists
2. **Restrict CORS** in `api/main.py` — replace `allow_origins=["*"]` with specific domains; set `allow_credentials=False` if not needed
3. **Make CI test gates blocking** — remove `|| echo` fallbacks from rubric-compliance workflow

### High (Address Before Production)

4. **Replace print statements with logging** — start with `train.py`, `nfl_data_loader.py`, `run_app.py`
5. **Add rate limiting** to the FastAPI API
6. **Fix ReDoS vulnerability** — use `regex=False` in `str.contains()` for user-supplied search terms
7. **Add `.dockerignore`** and switch to non-root user in Dockerfile
8. **Remove default database password** from `docker-compose.yml`

### Medium (Improve Reliability)

9. **Add tests for critical untested modules** — ensemble.py, train.py, position_models.py, metrics.py
10. **Replace generic exception handlers** with specific exception types and logging
11. **Add linting to CI** — ruff or flake8 + mypy for type checking
12. **Add dependency vulnerability scanning** — pip-audit or Dependabot
13. **Pin all dependency versions** — replace `>=` with exact versions; add lock file

### Low (Technical Debt)

14. **Break apart large modules** — train.py (2,299 lines), feature_engineering.py (2,054 lines)
15. **Consolidate pytest fixtures** into conftest.py for better test reusability
16. **Clean up root directory** — move status/implementation docs to `docs/`
17. **Add structured logging** (JSON format) for production log aggregation
18. **Resolve skipped tests** — implement missing dependencies or remove dead tests

---

## 8. File Inventory

| Category | Count |
|----------|-------|
| Python source modules (`src/`) | 84 |
| Test files (`tests/`) | 49 |
| CI/CD workflows | 4 |
| Configuration files | 6 (pyproject.toml, pytest.ini, requirements.txt, etc.) |
| Documentation files (root) | 13 |
| Total Python LOC | ~64,400 |
| Frontend JS LOC | ~1,243 |

---

*Report generated by automated codebase audit on 2026-03-14.*
