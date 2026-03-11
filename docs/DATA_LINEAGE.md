# Data Lineage and Provenance

This project uses a formal **Bronze → Silver → Gold** artifact flow so model outputs can be traced back to exact source pulls.

## Artifact Root

All persisted artifacts are written under:

- `data/artifacts/bronze/`
- `data/artifacts/silver/`
- `data/artifacts/gold/`
- `data/artifacts/manifest.jsonl` (append-only index)

## File Naming Convention

Each artifact writes two files:

- `<artifact_id>.parquet`
- `<artifact_id>.metadata.json`

Where:

- `artifact_id = {layer}_{table}_{run_id}_{dataset_hash12}`
- `layer ∈ {bronze, silver, gold}`
- `dataset_hash12` is the first 12 hex characters of the DataFrame SHA-256 content hash.

## Bronze Layer (raw source pulls)

Bronze persists untouched source pulls per run.

### Required metadata fields

- `artifact_id`
- `layer` = `bronze`
- `table`
- `run_id`
- `source`
- `seasons` (list[int])
- `week_window` (`{"start": int|null, "end": int|null}`)
- `pulled_at` (UTC ISO-8601)
- `created_at` (UTC ISO-8601)
- `row_count`
- `column_count`
- `schema_hash` (SHA-256 over ordered `column:dtype`)
- `dataset_hash` (SHA-256 dataframe fingerprint)
- `parent_artifact_ids` (usually empty for Bronze)
- `parquet_path`

## Silver Layer (canonical/normalized tables)

Silver persists standardized canonical tables after cleaning/normalization.

### Required metadata fields

All common fields above, plus:

- `normalization` (human-readable normalization summary)
- `parent_artifact_ids` MUST include upstream Bronze artifact IDs.

## Gold Layer (model training matrix)

Gold persists the model training matrix used for fitting.

### Required metadata fields

All common fields above, plus:

- `feature_version` (from `config.settings.FEATURE_VERSION`)
- `target_definition` (explicit semantics for 1w/4w/18w targets)
- `parent_artifact_ids` MUST include Silver + originating Bronze IDs.

## Dataset Hash Logging Extension

`ExperimentTracker.log_dataset_hash()` now accepts `parent_artifact_ids`.

For each training run, the tracker logs:

- `training_matrix_gold`
- `holdout_matrix_gold`

with `parent_artifact_ids` populated so each gold hash can be traced to exact upstream Bronze files.

## Module Ownership

- `src/data/lineage.py`
  - artifact/hash helpers
  - parquet + metadata persistence
  - manifest append
  - artifact ID propagation helpers (`set_artifact_id`, `get_artifact_id`)
- `src/data/nfl_data_loader.py`
  - Bronze + Silver persistence for source pulls and canonical weekly tables
- `src/models/train.py`
  - Silver persistence for training features
  - Gold persistence for training matrix
  - dataset hash logging with lineage parents
