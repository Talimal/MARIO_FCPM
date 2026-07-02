# CPM Framework — Fully Continuous Prediction Model (FCPM) Pipeline

A distributed, SLURM-based grid-search pipeline for **temporal pattern mining and early-event prediction** using Temporal Interval-based Relational Patterns (TIRPs) and the Fully Continuous Prediction Model (FCPM).

---
## Setup Environment
To easily run the project, a Conda environment configuration file `environment.yml` is provided.

## Create the conda environment by running:
conda env create -f environment.yml

Once the environment is created, activate it with:
conda activate FCPM_env


## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Stage-by-Stage Breakdown](#4-stage-by-stage-breakdown)
   - [Stage 0 — Data Splitting](#stage-0--data-splitting)
   - [Stage 1 — Temporal Abstraction](#stage-1--temporal-abstraction)
   - [Stage 2 — TIRP Mining & Selection](#stage-2--tirp-mining--selection)
   - [Stage 3 — FCPM Model Building](#stage-3--fcpm-model-building)
   - [Stage 3.5 — Validation](#stage-35--validation)
   - [Stage 4 — Continuous Prediction](#stage-4--continuous-prediction)
   - [Stage 5 — Aggregation & Evaluation](#stage-5--aggregation--evaluation)
   - [Cleanup](#cleanup)
5. [Output Directory Structure](#5-output-directory-structure)
6. [How to Configure and Run](#6-how-to-configure-and-run)
   - [Step 1 — Edit config.py](#step-1--edit-configpy)
   - [Step 2 — Edit experiment_params.py](#step-2--edit-experiment_paramspy)
   - [Step 3 — Run the Main Experiment](#step-3--run-the-main-experiment)
   - [Step 4 — (Optional) Run the Monitor](#step-4--optional-run-the-monitor)
7. [Parameter Reference](#7-parameter-reference)
8. [Input Data Format](#8-input-data-format)
9. [Fault Tolerance and Resume](#9-fault-tolerance-and-resume)
10. [Results Format](#10-results-format)

---

## 1. Project Overview

This framework implements an end-to-end pipeline for **continuous event prediction** from multivariate time-series data. It is domain-agnostic and applicable to any setting where multiple variables are measured over time and a future discrete event must be predicted. The core idea is:

1. Convert raw continuous time-series variables into **Symbolic Time-Intervals (STIs)**: each variable's numeric trajectory is first discretized into a symbolic time-series (each measurement receives a state label), and consecutive identical states are then merged into intervals described by `(start, end, stateID, variableID)` — this is the STI representation.
2. Mine frequent **Temporal Interval-based Relational Patterns** (TIRPs) — recurring combinations of STIs and their Allen temporal relations — that end in the **event of interest** and are therefore candidates for prediction.
3. For each discovered TIRP, build a **Fully Continuous Prediction Model (FCPM)**: a probabilistic model that learns how TIRP prefix durations relate to the probability and timing of the target event.
4. At inference time, scan each entity's data continuously in time, compute a live probability score from all applicable TIRPs, and raise an **early warning** when the score exceeds a learned threshold for a sustained window.

**Academic foundation:** The methodology is based on the FCPM and TIRP frameworks developed in the group's research. TIRPs are mined using the KarmaLego algorithm. The FCPM models statistical distributions over TIRP prefix durations to yield a continuous, calibrated probability score.

**Key technologies:**
- **SLURM** — all computation is distributed across a cluster via `sbatch` job arrays
- **Python 3** — orchestration, modeling, and evaluation
- **KarmaLego** (`New_KarmaLego_Framework`) — temporal pattern mining
- **Hugobot2** — temporal abstraction / discretization library
- **scikit-learn** — cross-validation splits and supporting ML utilities

---

## 2. Repository Layout

```
CPM_Framework/
│
├── run_experiment.py           # MAIN ENTRY POINT — orchestrates all stages
├── experiment_monitor.py       # Optional parallel monitor (OOM detection + email)
├── config.py                   # All SLURM resource settings and algorithm constants
├── experiment_params.py        # Dataset definitions and parameter grids
├── Tirp_selection.py           # TIRP scoring and selection logic (7 methods)
│
├── run_stage0_split_folds.py   # Stage 0: K-fold CV data splitting
├── run_stage1_abstraction.py   # Stage 1: Temporal abstraction (continuous → Symbolic Time-Intervals / STIs)
├── run_stage2_mining.py        # Stage 2: TIRP mining (KarmaLego) + TIRP selection
├── run_stage3_build_model.py   # Stage 3: FCPM + TTE model building per TIRP
├── run_stage3_5_validation.py  # Stage 3.5: Validation-based TIRP selection (val_AUC, val_F1, etc.)
├── run_stage4_predict_entities.py  # Stage 4: Continuous inference on test entities
├── run_stage5_aggregation_eval.py  # Stage 5: Aggregation + early-warning evaluation
├── run_cleanup_fold.py         # Cleanup: delete fold directories on success
│
├── Hugobot2/                   # Temporal abstraction library
│   └── ta_package/             #   Core engine + discretization method implementations
│
├── New_KarmaLego_Framework/    # TIRP mining (KARMA indexing + LEGO pattern generation)
│   ├── RunKarmaLego.py         #   Main mining entry point
│   ├── Karma_new.py            #   KARMA algorithm
│   ├── Lego.py                 #   LEGO pattern extension
│   ├── Tirp_new.py             #   TIRP object representation
│   ├── TirpMatrix.py           #   Relation matrix
│   └── RelationHandler.py      #   Allen relation mapping
│
├── SKL/                        # TIRP detection utilities (used in Stage 2 scoring)
│   ├── Tirp_detection.py
│   ├── Tirp_new.py
│   ├── Karma_new.py
│   └── RelationHandler.py
│
├── CPM_Feature_Matrix/         # Feature matrix extraction (prefix durations → tables)
│   ├── Create_feature_matrix.py
│   ├── Run_create_feature_matrix.py
│   └── TIRP_prefixs_evolving_TIRPs.py
│
└── FCPM_Package/               # FCPM / CPML modeling, prediction, and evaluation
    ├── FCPM.py                 #   Distribution fitting over TIRP prefix durations
    ├── TTE.py                  #   Time-to-Event regression (GammaRegressor)
    ├── CPML.py                 #   Continuous Prediction Machine Learning (CPML) model
    ├── build_and_evaluate.py   #   Orchestrates Stage 3 model training
    ├── Continuous_Prediction.py #  Generates continuous probability scores (Stage 4)
    ├── Evaluate.py             #   ROC, AUPRC, threshold scanning, early-warning eval
    └── aggregation_class.py    #   Aggregates predictions across TIRPs (Stage 5)
```

---

## 3. Pipeline Architecture

### Overall Flow

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         run_experiment.py                                │
 │  (Monitoring loop: submits SLURM jobs, waits for .done files)            │
 └──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
 │  Stage 0    │────▶│  Stage 1    │────▶│  Stage 2    │────▶│  Stage 3    │
 │ Split Folds │     │ Abstraction │     │  Mining &   │     │ Build FCPM  │
 │             │     │(continuous  │     │  Selection  │     │  Models     │
 │ one job per │     │→ STIs)      │     │ (KarmaLego, │     │ (batched:   │
 │ dataset     │     │             │     │ event-only) │     │ 100/job)    │
 └─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                                                           ┌─────────▼──────┐
                                                           │  Stage 3.5     │
                                                           │  Validation-   │
                                                           │  based TIRP    │
                                                           │  Selection     │
                                                           └─────────┬──────┘
                                                                     │
 ┌─────────────┐     ┌─────────────┐                      ┌─────────▼──────┐
 │  Cleanup    │◀────│  Stage 5    │◀─────────────────────│  Stage 4       │
 │ (optional)  │     │ Aggregation │                       │  Predict       │
 │             │     │     +       │                       │  Entities      │
 │             │     │ Evaluation  │                       │ (batched:      │
 │             │     │             │                       │ 100/job)       │
 └─────────────┘     └─────────────┘                      └────────────────┘
```

### Key Orchestration Concepts

#### Done-file dependency gating
Every SLURM job writes a `.done` file upon successful completion. The monitoring loop in `run_experiment.py` polls for these files before advancing to the next stage. This makes the pipeline **fully resumable** — restarting the orchestrator after an interruption will skip all already-completed jobs.

#### Batching
Large workloads are split into parallel SLURM array jobs:
- **Stage 3 / 3.5**: TIRPs are divided into batches of `TIRP_BATCH_SIZE_STAGE3` (default 100) and `TIRP_BATCH_SIZE_STAGE3_5` (default 10) per job.
- **Stage 4**: Test entities are divided into batches of `ENTITY_BATCH_SIZE_FOR_PREDICTION` (default 100) per job.

#### Parameter grid search
The framework performs a **full Cartesian product** over all abstraction parameters × mining parameters × aggregation parameters × TIRP selection methods. Each unique combination is an independent experiment path and is evaluated independently in Stage 5.

#### Memory-based job throttling
Instead of limiting by job count, the orchestrator sums the allocated memory of all running/pending SLURM jobs and waits when the total exceeds `MAX_MEMORY_GB` (default 4000 GB). This prevents cluster overload.

---

## 4. Stage-by-Stage Breakdown

---

### Stage 0 — Data Splitting

| Item | Value |
|---|---|
| **Script** | `run_stage0_split_folds.py` |
| **Packages used** | `sklearn` (GroupKFold, GroupShuffleSplit) |
| **SLURM resources** | 1 CPU, 2 GB RAM, 30 min |
| **Done file** | `{dataset_base_dir}/stage0_split.done` |

**What it does:**
Splits the raw dataset into K independent cross-validation folds, keeping entity groups together (no data leakage across train/test at the group level). When `n_folds == 1`, a single 80/20 stratified split is created using `GroupShuffleSplit`. When `n_folds > 1`, `GroupKFold` is used.

**Inputs:**
| Argument | Description |
|---|---|
| `--dataset_name` | Dataset identifier string |
| `--data_path` | Path to raw data CSV |
| `--matching_path` | Path to matching CSV (EntityID → GroupID mapping) |
| `--n_folds` | Number of folds (use 1 for single 80/20 split) |
| `--dataset_base_dir` | Output directory where fold subdirectories are created |

**Outputs:**
```
{dataset_base_dir}/
├── fold_1/
│   ├── train.csv
│   └── test.csv
├── fold_2/
│   ├── train.csv
│   └── test.csv
...
└── fold_N/
    ├── train.csv
    └── test.csv
```

---

### Stage 1 — Temporal Abstraction

| Item | Value |
|---|---|
| **Script** | `run_stage1_abstraction.py` |
| **Packages used** | `Hugobot2` (TemporalAbstraction, utils) |
| **SLURM resources** | 1 CPU, 4 GB RAM, ~7 days max |
| **Done file** | `{fold_dir}/{abs_param_string}/stage1_abstraction.done` |

**What it does:**
Converts continuous multivariate time-series into **Symbolic Time-Intervals (STIs)** through a two-step process:

1. **Discretization (symbolic time-series):** Each numeric variable is discretized into a finite set of symbolic states using the chosen `d_method`. Every measurement at every time point receives a state label (e.g., state A = low, B = medium, C = high).
2. **Interval formation (STI):** Consecutive time points with the same state label are merged into a single symbolic time-interval `(start, end, stateID, variableID)`. The result for each entity is a set of non-overlapping STIs per variable — this is the STI representation that KarmaLego consumes.

The discretization boundaries are learned from the **training set** and then applied to the **test set** using the same thresholds. This prevents leakage.

If `split_event_class=True` and `event_window` is set, training data is partitioned into:
- **Class 1 (cases)**: time windows immediately preceding the target event (within `event_window` time units)
- **Class 0 (controls)**: remaining time windows

The output files are written in **KL format** (Karma-Lego format) — a text format encoding the full STI representation for each entity, consumed by KarmaLego in Stage 2.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--train_data_file` | str | Path to fold's train.csv |
| `--test_data_file` | str | Path to fold's test.csv |
| `--d_method` | str | Discretization method (see [Parameter Reference](#temporal-abstraction-methods-d_method)) |
| `--num_of_bins` | int | Number of symbolic states (bins) per variable |
| `--interpolation_gap` | int | Maximum allowed gap (time units) for interpolation between measurements |
| `--abstraction_output_dir` | str | Output directory (Train/ and Test/ subdirs created here) |
| `--split_event_class` | str | `"True"` or `"False"` — enable event-based class splitting |
| `--event_window` | str | Float (percentage of series length) or int (absolute time units) for the pre-event window |

**Outputs:**
```
{abstraction_output_dir}/
├── Train/
│   ├── KL-class-0.0.txt        # STI representation for control entities (class 0)
│   ├── KL-class-1.0.txt        # STI representation for case entities (class 1)
│   └── entity-class-relations.csv  # EntityID → class label mapping
└── Test/
    ├── KL-class-0.0.txt
    └── KL-class-1.0.txt
```

**KL File Format (STI encoding):**
Each line after the entity header encodes all symbolic time-intervals for that entity. Each STI is a 4-tuple: `start,end,stateID,variableID`.
```
startToncepts
numberOfEntities,N
EntityID_1;
start,end,stateID,variableID;start,end,stateID,variableID;...;
EntityID_2;
...
```

---

### Stage 2 — TIRP Mining & Selection

| Item | Value |
|---|---|
| **Script** | `run_stage2_mining.py` |
| **Packages used** | `New_KarmaLego_Framework` (RunKarmaLego), `Tirp_selection.py` (TirpSelection) |
| **SLURM resources** | 4 CPUs, 10 GB RAM, ~6.5 days max |
| **Done file** | `{mining_run_dir}/stage2_mining.done` |

**What it does:**
Mines **frequent TIRPs** from the STI training data using the **KarmaLego** algorithm. KarmaLego has two phases:
1. **KARMA**: builds an index of co-occurring STI pairs for each entity, capturing all pairwise Allen temporal relations.
2. **LEGO**: extends shorter patterns into longer ones, pruning by minimum vertical support at each step.

**Critical filter — event-ending TIRPs only:**
After mining, a fundamental design constraint is applied: **only TIRPs whose last STI is the event of interest** (i.e., last symbol == `event_symbol`) are retained. This is the defining property of the predictive patterns in this framework — a TIRP is only useful for prediction if it culminates in the target event. TIRPs that do not end in the event symbol are discarded entirely and never used in any downstream stage.

In practice this means:
- size > 2 (the pattern must have at least 2 STIs, so there is at least one non-event prefix to model)
- last symbol == `event_symbol` (the final STI in the pattern is the event of interest)

**TIRP Scoring and Selection:**
Each event-ending TIRP is then **scored** using multiple statistical methods that measure how discriminative the pattern is between class 1 (cases) and class 0 (controls). For each scoring method and each top-K limit (from `MAX_TIRPS_FOR_SELECTION`), the top-K TIRPs are flagged with `Binary_{method}#{K}` columns in the scores CSV. Only flagged TIRPs are saved as pickle files and passed to Stage 3 for model building.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--abstraction_output_dir` | str | Stage 1 output directory (contains Train/ subdirectory) |
| `--mining_run_dir` | str | Output directory for mining results |
| `--tirp_objects_output_dir` | str | Where to save selected TIRP `.pkl` files |
| `--mvs` | float | Minimum Vertical Support (0.0–1.0): fraction of training entities that must contain the TIRP |
| `--max_gap` | int | Maximum allowed time gap between consecutive intervals in a TIRP |
| `--relations` | int | Number of Allen temporal relations to consider: 3 (before, meets, overlaps) or 7 (all) |
| `--skip_followers` | bool | If True, skip temporal relations between intervals of the same variable |
| `--epsilon` | int | Temporal tolerance: intervals within ε time units of exact boundary are considered coincident |
| `--event_symbol` | str | Numeric symbol representing the target event (default: 999) |

**Outputs:**
```
{mining_run_dir}/
├── tirps/
│   ├── tirp_0.pkl              # Serialized TIRP objects (one per selected TIRP)
│   ├── tirp_1.pkl
│   └── ...
├── tirp_selection_scores.csv   # All TIRPs with scores and Binary_* selection flags
└── KL_data/
    ├── KL-class-0.0.txt        # Copy of training data (split at window boundary)
    └── KL-class-1.0.txt
```

**TIRP Scoring Methods (columns in tirp_selection_scores.csv):**
| Score | Description |
|---|---|
| `Vertical_Support` | Fraction of entities containing the full TIRP |
| `diff_vertical_support` | Difference in vertical support between class 1 and class 0 |
| `diff_horizontal_support` | Difference in horizontal support (average occurrences per entity) |
| `diff_mean_duration` | Difference in mean duration of the last prefix between classes |
| `mean_squared_mmd` | Mean squared MMD (Mean-Mean Duration) difference across all TIRP prefixes |
| `mean_squared_vs` | Mean squared vertical support difference across all TIRP prefixes |
| `random` | Random score (used as a baseline) |

For each score × top-K limit (from `MAX_TIRPS_FOR_SELECTION`), a binary flag column `Binary_{method}#{K}` is added indicating whether the TIRP is selected.

---

### Stage 3 — FCPM Model Building

| Item | Value |
|---|---|
| **Script** | `run_stage3_build_model.py` |
| **Packages used** | `CPM_Feature_Matrix` (create_feature_matrix_for_CPM), `FCPM_Package` (FCPM, TTE, CPML, build_and_evaluate) |
| **SLURM resources** | 1 CPU, 5 GB RAM, ~7 days max |
| **Batch size** | `TIRP_BATCH_SIZE_STAGE3 = 100` TIRPs per SLURM job |
| **Done files** | `{mining_run_dir}/feature_matrix/batch_status/stage3_batch_{N:04d}.done` |

**What it does:**
For each selected TIRP, this stage builds the predictive models that will be used for inference. The process consists of:

**1. Prefix Table Extraction (CPM_Feature_Matrix)**
For each TIRP, all sub-patterns (prefixes) are enumerated. For each prefix, the system scans the training data to find every occurrence and records:
- `EntityID` — which entity
- `TFS` (Time First Seen) — when this prefix first appeared
- Durations between each consecutive TIEP (Time-Interval End Point, i.e., the start or end of an STI)

This yields a "prefix detection table" for each TIRP prefix.

**2. Feature Matrix Creation**
- **Training feature matrix** (`create_feature_matrix_for_TTE`): Flattens time — each time unit in each entity's trajectory becomes one row, with binary indicators (`_Binary`) for when interval boundaries fire and continuous duration trackers (`_Duration`). The target variable `TTE` (Time-To-Event) is computed as the remaining time until the outcome event.
- **Test feature matrix** (`durations_merged_df`): Absolute-time matrix starting from each TIRP's `TFS`, used for inference.
- **Inference tables** are also saved for Stage 3.5 validation.

**3. Model Fitting**
- **FCPM model** (`FCPM.py`): Fits a statistical distribution (exponential, Weibull, lognormal, Pareto, halfnormal, or exponential-Weibull) to the observed prefix durations. Uses SSE or MLE fitting with optional half-Cauchy or Student-t smoothing. This model outputs a calibrated probability that the target event will occur.
- **TTE model** (`TTE.py`): Trains a `GammaRegressor` on the feature matrix to predict the expected remaining time until the event. Input features are scaled before fitting.
- **CPML model** (`CPML.py`, optional): An additional **Continuous Prediction Machine Learning (CPML)** model trained if `BUILD_CPML=True` in config.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--abstraction_output_dir` | str | Stage 1 output directory |
| `--tirp_model_run_dir` | str | Base directory for model outputs (`feature_matrix/`) |
| `--tirp_list_file` | str | Text file listing TIRP `.pkl` paths (one per line) — generated by orchestrator |
| `--max_gap` | int | Max gap (same as Stage 2 — needed for window size computation) |
| `--num_relations` | int | Number of relations (same as Stage 2) |
| `--epsilon` | int | Epsilon (same as Stage 2) |
| `--event_symbol` | str | Event symbol (same as Stage 2) |
| `--window_size_ratio` | float | Multiplier for computing the split window size (default from `config.WINDOW_SIZE_RATIO = 1.2`) |
| `--build_cpml` | flag | If present, also build CPML models |

**Outputs (per TIRP):**
```
{mining_run_dir}/feature_matrix/tirp_{id}/
├── class0/                     # Feature matrices for control entities
├── class1/                     # Feature matrices for case entities
├── test/                       # Test feature matrices for inference
├── train_inference/
│   ├── durations_merged_df.csv # Merged prefix duration table (used in Stage 3.5)
│   └── event_time_dict.pkl     # Entity → event time mapping
└── models/
    ├── {id}-FCPM.pkl           # Fitted FCPM distribution model
    ├── {id}-TTE.pkl            # Fitted TTE regression model
    └── {id}-CPML.pkl           # CPML model (if --build_cpml)
```

---

### Stage 3.5 — Validation-Based TIRP Selection

| Item | Value |
|---|---|
| **Script** | `run_stage3_5_validation.py` |
| **Packages used** | `FCPM_Package` (Continuous_Prediction, Evaluate, aggregation_class) |
| **SLURM resources** | 1 CPU, 5 GB RAM, ~7 days max |
| **Batch size** | `TIRP_BATCH_SIZE_STAGE3_5 = 10` TIRPs per SLURM job |
| **Enabled by** | `RUN_STAGE3_5_VALIDATION = True` in `config.py` |
| **Done files** | `{mining_run_dir}/feature_matrix/batch_status_val/stage3_5_batch_{N:04d}.done` |

**What it does:**
Stage 3.5 provides an additional family of **data-driven TIRP selection criteria** — `val_AUC`, `val_AUPRC`, `val_F1`, and `val_Precision` — that complement the statistical scoring methods computed in Stage 2. Rather than selecting TIRPs based on support or duration differences between classes, these criteria select TIRPs based on how well each TIRP's FCPM model actually predicts the event on the training data.

Concretely, for each TIRP, the trained FCPM model from Stage 3 is applied to the **training data** (using the inference tables saved in Stage 3) to simulate the full early-warning system at training time. The system evaluates whether the continuous predictions, when thresholded, correctly identify the target event within a tolerance window. This yields per-TIRP performance metrics (AUC, AUPRC, F1, Precision) that represent the individual predictive power of each TIRP.

These metrics are then written back into `tirp_selection_scores.csv`, adding `val_AUC`, `val_AUPRC`, `val_F1`, and `val_Precision` columns alongside the Stage 2 statistical scores. In Stage 5, when `tirp_selection_method` is set to one of the `val_*` values, TIRPs are ranked and filtered by their training-set performance rather than by structural properties alone.

If `SUPPORTING_ENTITIES_ONLY = True` in config, the evaluation is limited to entities where the TIRP actually appears (supporting entities), excluding entities where the pattern never occurs.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--tirp_model_run_dir` | str | Stage 3 output base directory (contains `feature_matrix/`) |
| `--tirp_list_file` | str | Text file listing TIRP `.pkl` paths |
| `--abstraction_output_dir` | str | Stage 1 output (used to load `entity-class-relations.csv`) |
| `--epsilon` | float | Temporal tolerance (same as earlier stages) |
| `--ew_window_size` | float | Scalar multiplier for early-warning TTE window size |
| `--ew_early_warning_value` | int | Number of **consecutive** time units above threshold required to trigger a warning |

**Outputs (per TIRP):**
```
{mining_run_dir}/feature_matrix/tirp_{id}/
├── train_predictions_FCPM/     # Per-entity prediction CSV files
├── train_metrics_all_thresholds.csv  # Metrics at every threshold 0.0–1.0
├── train_summary_metrics.csv   # Best-threshold summary: AUC, AUPRC, F1, Precision
└── stage3_val_{id}.done
```

The orchestrator reads `train_summary_metrics.csv` to update the TIRP selection scores CSV.

---

### Stage 4 — Continuous Prediction

| Item | Value |
|---|---|
| **Script** | `run_stage4_predict_entities.py` |
| **Packages used** | `FCPM_Package` (Continuous_Prediction.predict_continuous) |
| **SLURM resources** | 1 CPU, 4 GB RAM, ~7 days max |
| **Batch size** | `ENTITY_BATCH_SIZE_FOR_PREDICTION = 100` entities per SLURM job |
| **Done files** | `{mining_run_dir}/predictions/batch_status_stage4/stage4_predict_batch_{N:04d}.done` |

**What it does:**
Runs **batched inference** on all test entities. For each test entity and each time unit in its trajectory, the stage:
1. Checks which TIRPs are currently "active" (their prefix has been observed so far).
2. Loads the corresponding FCPM and TTE models.
3. Computes an FCPM probability score and a TTE prediction at that time unit.
4. Appends the result to the entity's prediction file.

The output is one compressed CSV per entity containing the full time-series of predictions.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--entity_list_file` | str | Text file with one entity ID per line |
| `--built_models_base_dir` | str | Stage 3 output directory (contains `feature_matrix/`) |
| `--prediction_output_dir` | str | Directory where per-entity prediction CSVs are written |

**Outputs:**
```
{mining_run_dir}/predictions/
├── entity_{id}_FCPM.csv.gz     # Per-entity predictions (gzipped)
│   # Columns: EntityID, TFS, outcome_class, FCPM_Prediction, TTE_prediction
├── entity_{id}_CPML.csv.gz     # (if CPML was built)
└── batch_status_stage4/
    ├── stage4_predict_batch_0001.done
    └── ...
```

---

### Stage 5 — Aggregation & Evaluation

| Item | Value |
|---|---|
| **Script** | `run_stage5_aggregation_eval.py` |
| **Packages used** | `FCPM_Package` (aggregation_class, Evaluate) |
| **SLURM resources** | 1 CPU, 4 GB RAM, ~7 days max |
| **Done files** | `{mining_run_dir}/results/{model}_{params}_{method}_{ew}.done` |

**What it does:**
This is the final evaluation stage. It has two sub-processes:

**1. Aggregation (`aggregation_class.py`)**
Reads all per-entity, per-TIRP predictions from Stage 4 and combines them into a single prediction per `(EntityID, TFS, outcome_class)` tuple. Before aggregating, it filters TIRPs based on the selected `tirp_selection_method` and `num_tirps_for_selection` using the scores in `tirp_selection_scores.csv`. The remaining predictions are combined using the chosen `aggregation_method` (avg, max, min, etc.).

**2. Evaluation (`Evaluate.py`)**
Simulates an **Early Warning system** by scanning a range of decision thresholds (0.0 to 1.0 in steps of 0.01). At each threshold:
- A warning is triggered only when predictions **stay above the threshold** for `e_w` consecutive time units (`early_warning_value`).
- Upon triggering, the system checks whether the predicted `TTE_prediction` falls within a `TTE_W`-sized tolerance window around the true event time.
- This yields TP, FP, TN, FN counts → Precision, Recall, F1, Accuracy, AUC, AUPRC.

A vectorized implementation (`_classify_entity_at_threshold_vectorized`) is used for efficiency.

**Arguments:**
| Argument | Type | Description |
|---|---|---|
| `--dataset_name` | str | Dataset name |
| `--fold_num` | int | Fold number |
| `--abs_d_method` | str | Abstraction method used |
| `--abs_b` | int | Number of bins used |
| `--abs_ig` | int | Interpolation gap used |
| `--abs_split_event_class` | str | Whether event-class splitting was used |
| `--abs_event_window` | str | Event window size (if split was used) |
| `--mine_mvs` | float | Mining minimum vertical support |
| `--mine_mg` | int | Mining maximum gap |
| `--mine_rel` | int | Mining number of relations |
| `--mine_sf` | bool | Mining skip followers flag |
| `--mine_e` | int | Mining epsilon |
| `--agg_aggregation_method` | str | How to aggregate multi-TIRP predictions |
| `--agg_num_tirps_for_selection` | int | How many TIRPs to include after filtering |
| `--tirp_selection_method` | str | Criterion for TIRP filtering |
| `--TTE_W_list` | str | Comma-separated list of TTE tolerance window sizes to evaluate |
| `--e_w_list` | str | Comma-separated list of early-warning consecutive-hit values |
| `--model_type` | str | `"FCPM"` or `"CPML"` |
| `--prediction_base_dir` | str | Stage 4 predictions directory |
| `--output_csv_path` | str | Directory for per-threshold ROC metric CSVs |
| `--results_dir` | str | Directory for summary CSV (appended with one row per TTE_W/e_w combo) |

**Outputs:**
```
{mining_run_dir}/results/
├── FCPM_ts_{method}_agg_{agg}.csv       # Full ROC metrics at every threshold
├── FCPM_{params}_{method}_{ew}.done     # Done file
└── ...

{dataset}/results/
└── summery_result_all.csv               # Merged summary (all folds × all params × TTE_W × e_w)
```

---

### Cleanup

| Item | Value |
|---|---|
| **Script** | `run_cleanup_fold.py` |
| **Condition** | Only runs if `DELETE_FOLDS_ON_SUCCESS = True` in `config.py` AND results merged successfully |

**What it does:**
Deletes fold directories (including all Stage 1–4 intermediate files) to free disk space after a fold completes successfully. Uses parallel deletion (`ThreadPoolExecutor`) for speed. As a safety measure, it **refuses to delete any directory path containing the string `"results"`**.

---

## 5. Output Directory Structure

```
{BASE_OUTPUT_DIR}/
│
├── {dataset_name}/
│   │
│   ├── fold_1/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   │
│   │   ├── abs_d_method=equal_frequency_b=3_ig=5/   ← one dir per abs combo
│   │   │   ├── Train/
│   │   │   │   ├── KL-class-0.0.txt
│   │   │   │   ├── KL-class-1.0.txt
│   │   │   │   └── entity-class-relations.csv
│   │   │   ├── Test/
│   │   │   │   ├── KL-class-0.0.txt
│   │   │   │   └── KL-class-1.0.txt
│   │   │   ├── stage1_abstraction.done
│   │   │   │
│   │   │   └── mine_mvs=0.15_mg=20_rel=7_sf=True_e=0/   ← one dir per mine combo
│   │   │       ├── KL_data/
│   │   │       │   ├── KL-class-0.0.txt
│   │   │       │   └── KL-class-1.0.txt
│   │   │       ├── tirps/
│   │   │       │   ├── tirp_0.pkl
│   │   │       │   ├── tirp_1.pkl
│   │   │       │   └── ...
│   │   │       ├── tirp_selection_scores.csv
│   │   │       ├── stage2_mining.done
│   │   │       │
│   │   │       ├── feature_matrix/
│   │   │       │   ├── tirp_0/
│   │   │       │   │   ├── class0/
│   │   │       │   │   ├── class1/
│   │   │       │   │   ├── test/
│   │   │       │   │   ├── train_inference/
│   │   │       │   │   │   ├── durations_merged_df.csv
│   │   │       │   │   │   └── event_time_dict.pkl
│   │   │       │   │   ├── models/
│   │   │       │   │   │   ├── tirp_0-FCPM.pkl
│   │   │       │   │   │   ├── tirp_0-TTE.pkl
│   │   │       │   │   │   └── tirp_0-CPML.pkl
│   │   │       │   │   ├── train_predictions_FCPM/
│   │   │       │   │   ├── train_metrics_all_thresholds.csv
│   │   │       │   │   ├── train_summary_metrics.csv
│   │   │       │   │   └── stage3_val_tirp_0.done
│   │   │       │   ├── tirp_1/ ...
│   │   │       │   ├── batch_status/
│   │   │       │   │   ├── stage3_batch_0001.done
│   │   │       │   │   └── ...
│   │   │       │   └── batch_status_val/
│   │   │       │       ├── stage3_5_batch_0001.done
│   │   │       │       └── ...
│   │   │       │
│   │   │       ├── predictions/
│   │   │       │   ├── entity_001_FCPM.csv.gz
│   │   │       │   ├── entity_002_FCPM.csv.gz
│   │   │       │   └── batch_status_stage4/
│   │   │       │       ├── stage4_predict_batch_0001.done
│   │   │       │       └── ...
│   │   │       │
│   │   │       └── results/
│   │   │           ├── FCPM_agg=avg_num_tirps=10_val_AUC_e_w=3.done
│   │   │           ├── FCPM_ts_val_AUC_agg_avg.csv
│   │   │           └── ...
│   │
│   ├── fold_2/ ...
│   ├── fold_N/ ...
│   │
│   └── results/
│       └── summery_result_all.csv   ← merged results from all folds + all params
│
├── logs/                            ← SLURM .out and .err files
│   ├── stage1_fold1_abs_xxx.out
│   ├── stage1_fold1_abs_xxx.err
│   └── ...
├── sbatch_scripts/                  ← auto-generated SLURM submission scripts
├── temp_files/                      ← temporary TIRP/entity batch list files
├── status/
│   └── experiment_status.json       ← live JSON progress tracker
└── OOM_resubmit_scripts/            ← auto-generated scripts for OOM resubmissions
```

---

## 6. How to Configure and Run

### Step 1 — Edit `config.py`

`config.py` contains all global settings for the experiment. Edit this file before launching.

#### Path Settings
| Setting | Type | Description |
|---|---|---|
| `CODE_DIR` | `str` | Absolute path to the CPM_Framework directory (this repo) |
| `BASE_OUTPUT_DIR` | `str` | Root directory where all experiment outputs will be written |

#### Cross-Validation
| Setting | Type | Default | Description |
|---|---|---|---|
| `N_FOLDS` | `int` | `5` | Number of cross-validation folds. Set to `1` for a single 80/20 split |
| `SEED` | `int` | `345` | Random seed for reproducible fold splits |

#### Model Configuration
| Setting | Type | Default | Description |
|---|---|---|---|
| `BUILD_CPML` | `bool` | `True` | If True, trains a Continuous Prediction Machine Learning (CPML) model in addition to the FCPM model for each TIRP |
| `RUN_STAGE3_5_VALIDATION` | `bool` | `True` | If True, runs Stage 3.5 validation to compute per-TIRP val metrics |
| `SUPPORTING_ENTITIES_ONLY` | `bool` | `True` | If True, Stage 3.5 validation evaluates only entities where the TIRP appears |
| `MAX_TIRPS_FOR_SELECTION` | `list[int]` | `[5, 10, 15, 25]` | Top-K values to test in TIRP selection. For each value K, the top K TIRPs by each scoring method are flagged |
| `EPSILON_FCPM` | `float` | `1.2` | Temporal tolerance parameter passed to the FCPM prediction step |
| `WINDOW_SIZE_RATIO` | `float` | `1.2` | Multiplier applied to `max_gap` to define the pre-event window boundary in Stage 3 |

#### Temporal Abstraction (Stage 1) — Knowledge-Based Method
| Setting | Type | Default | Description |
|---|---|---|---|
| `GRADIENT_WINDOW_SIZE` | `int` | `5` | Window size for gradient-based temporal abstraction |
| `KB_STATES_PATH` | `str` | `""` | Path to a CSV file defining knowledge-based state boundaries (required when using `knowledge` or `knowledge_and_kb_gradient` method) |
| `GRADIENT_CUTOFFS_PATH` | `str` | `""` | Path to a JSON file defining gradient cutoff thresholds (required for gradient-based methods) |

#### Batch Sizes
| Setting | Type | Default | Description |
|---|---|---|---|
| `TIRP_BATCH_SIZE_STAGE3` | `int` | `100` | Number of TIRPs processed per SLURM job in Stage 3 |
| `TIRP_BATCH_SIZE_STAGE3_5` | `int` | `10` | Number of TIRPs processed per SLURM job in Stage 3.5 (smaller because validation is more memory-intensive) |
| `ENTITY_BATCH_SIZE_FOR_PREDICTION` | `int` | `100` | Number of test entities predicted per SLURM job in Stage 4 |

#### SLURM Job Control
| Setting | Type | Default | Description |
|---|---|---|---|
| `MAX_MEMORY_GB` | `int` | `4000` | Maximum total allocated memory (GB) across all running/pending SLURM jobs. Orchestrator pauses submission when this limit is reached |
| `JOB_SUBMISSION_WAIT_INTERVAL_SECONDS` | `int` | `30` | Seconds to wait before retrying submission when memory limit is reached |
| `MAX_WAIT_STAGE0_SECONDS` | `int` | `1800` | Maximum seconds to wait for Stage 0 to complete (30 minutes) |
| `MAX_MONITORING_CYCLES` | `int` | `2880` | Maximum total monitoring loop iterations (~48 hours at 60s/cycle) |
| `MAX_IDLE_CYCLES` | `int` | `30` | Maximum consecutive cycles with no new jobs submitted before logging a warning |
| `DEFAULT_WAIT_TIME_SECONDS` | `int` | `60` | Check interval when jobs are running normally |
| `ACTIVE_WAIT_TIME_SECONDS` | `int` | `20` | Check interval immediately after submitting new jobs |
| `IDLE_WAIT_TIME_SECONDS` | `int` | `120` | Check interval when no jobs are pending |
| `DELETE_FOLDS_ON_SUCCESS` | `bool` | `False` | If True, fold directories are deleted after results are successfully merged |

#### SLURM Resources per Stage
Defined as a dict `SBATCH_RESOURCES` mapping stage number → `{cpus, mem, time_limit}`:

| Stage | CPUs | Memory | Time Limit |
|---|---|---|---|
| 0 (Split Folds) | 1 | 2 GB | 30 minutes |
| 1 (Abstraction) | 1 | 4 GB | ~7 days |
| 2 (Mining) | 4 | 10 GB | ~6.5 days |
| 3 (Build Model) | 1 | 5 GB | ~7 days |
| 3.5 (Validation) | 1 | 5 GB | ~7 days |
| 4 (Predict) | 1 | 4 GB | ~7 days |
| 5 (Aggregation) | 1 | 4 GB | ~7 days |
| cleanup | 4 | 8 GB | ~7 days |

These can be increased individually if jobs fail with OOM. Alternatively, `experiment_monitor.py` handles OOM failures automatically.

#### Conda Environment
| Setting | Type | Default | Description |
|---|---|---|---|
| `CONDA_ENV_NAME` | `str` | `"FCPM_env"` | Name of the conda environment to activate in SLURM jobs |
| `CONDA_LOAD_COMMAND` | `str` | `"module load anaconda"` | Module load command before conda activation |

---

### Step 2 — Edit `experiment_params.py`

`experiment_params.py` defines the datasets to process and the parameter grids to search over.

#### Dataset Configuration

Each dataset is a Python dictionary added to the `datasets_to_run` list:

```python
datasets_to_run = [
    {
        "dataset_name": "my_dataset",           # Unique name — used in directory structure
        "data_path": "/path/to/data.csv",       # Raw temporal data file
        "matching_path": "/path/to/matching.csv", # EntityID → GroupID mapping file
        "event_symbol": 999,                    # Numeric code for the target event
        # Stage 1 parameters:
        "d_method": ["equal_frequency", "td4c_kullback-leibler"],
        "b": [3, 5],                            # Number of bins (states)
        "ig": [5],                              # Interpolation gap
        "split_event_class": [False],           # Event-based class splitting
        "event_window": [None],                 # Window for event split
        # Stage 2 parameters:
        "mvs": [0.1, 0.15],                    # Minimum vertical support
        "mg": [20],                             # Maximum gap
        "rel": [7],                             # Number of Allen relations
        "sf": [True],                           # Skip followers
        "e": [0],                               # Epsilon
        # Stage 5 parameters:
        "aggregation_method": ["avg", "max"],   # Aggregation methods
        "num_tirps_for_selection": [5, 10, 15, 25],  # Top-K TIRP selection
        "TTE_W": [7, 14],                       # TTE tolerance windows (time units)
        "e_w": [1, 3],                          # Early warning consecutive hits
    }
]
```

#### Dataset Fields
| Field | Type | Description |
|---|---|---|
| `dataset_name` | `str` | Unique identifier — used for directory naming |
| `data_path` | `str` | Absolute path to the input temporal data CSV |
| `matching_path` | `str` | Absolute path to the entity-group matching CSV |
| `event_symbol` | `int` | Integer code appended to represent the target event during abstraction |

#### Stage 1 Parameters (Abstraction)
| Parameter | Type | Description |
|---|---|---|
| `d_method` | `list[str]` | Discretization method(s) — see full table in [Parameter Reference](#temporal-abstraction-methods-d_method) |
| `b` | `list[int]` | Number of bins (symbolic states) per variable. Typical range: 3–7 |
| `ig` | `list[int]` | Interpolation gap: maximum number of time units between measurements that will be linearly interpolated. Larger values fill more gaps but may introduce noise |
| `split_event_class` | `list[bool]` | Whether to split training data into pre-event (Class 1) and control (Class 0) windows. Only valid for supervised methods (`td4c_*`) |
| `event_window` | `list[float or int or None]` | Size of the pre-event window. Float = fraction of total series length. Int = absolute time units. `None` = disabled |

#### Stage 2 Parameters (Mining)
| Parameter | Type | Description |
|---|---|---|
| `mvs` | `list[float]` | Minimum Vertical Support (0.0–1.0): the minimum fraction of training entities that must contain the TIRP for it to be considered frequent. Lower values mine more (but noisier) patterns |
| `mg` | `list[int]` | Maximum Gap: the maximum allowed time gap (in time units) between consecutive temporal intervals in a TIRP. Controls how "loose" the temporal patterns can be |
| `rel` | `list[int]` | Number of Allen temporal relations: `3` = {before, meets, overlaps}; `7` = all 7 Allen relations. Using 7 mines more specific patterns but dramatically increases the search space |
| `sf` | `list[bool]` | Skip Followers: if `True`, temporal relations between intervals of the **same variable** are ignored. Reduces noise from autocorrelation within a single time series |
| `e` | `list[int]` | Epsilon: temporal constraint tolerance. Intervals within ε time units of an exact relation boundary are treated as satisfying that relation. Larger ε finds more matches at the cost of precision |

#### Stage 5 Parameters (Aggregation & Evaluation)
| Parameter | Type | Description |
|---|---|---|
| `aggregation_method` | `list[str]` | Method to combine predictions from multiple TIRPs — see [Aggregation Methods](#aggregation-methods-aggregation_method) |
| `num_tirps_for_selection` | `list[int]` | Number of top TIRPs to include after filtering. Must be values that also appear in `config.MAX_TIRPS_FOR_SELECTION` |
| `TTE_W` | `list[int]` | Time-To-Event tolerance window (in time units). A TTE prediction is considered correct if it falls within ±`TTE_W` of the true event time |
| `e_w` | `list[int]` | Early Warning consecutive-hit value: number of consecutive time units the prediction must stay above the decision threshold before triggering an alert. Higher values reduce false alarms at the cost of sensitivity |

#### TIRP Selection Methods
Defined in `Tirp_selection_methods` list. The orchestrator evaluates **every combination** of aggregation parameters × TIRP selection method. See full description in [Parameter Reference](#tirp-selection-methods-tirp_selection_methods).

---

### Step 3 — Run the Main Experiment

```bash
cd /path/to/CPM_Framework
python run_experiment.py
```

There are **no command-line arguments** — all configuration is done via `config.py` and `experiment_params.py`.

The orchestrator will:
1. Create the output directory tree under `BASE_OUTPUT_DIR`.
2. Set up logging at `{BASE_OUTPUT_DIR}/logs/` and `{BASE_OUTPUT_DIR}/status/experiment_status.json`.
3. For each dataset: submit Stage 0, wait for completion, then enter the monitoring loop.
4. The monitoring loop polls every `DEFAULT_WAIT_TIME_SECONDS` seconds, checks done files, and submits the next wave of jobs as prerequisites complete.
5. After all stages complete for a dataset, results are merged into `summery_result_all.csv`.

**To run in the background (recommended for long experiments):**
```bash
nohup python run_experiment.py > orchestrator.log 2>&1 &
echo "Orchestrator PID: $!"
```

**To resume after an interruption:** simply re-run `python run_experiment.py`. The orchestrator checks for existing `.done` files and skips completed jobs automatically.

---

### Step 4 — (Optional) Run the Monitor

`experiment_monitor.py` is an **independent monitoring process** that can run in parallel with the main orchestrator. It provides:
- Email progress reports at configurable intervals
- Automatic detection of OOM (Out-of-Memory) SLURM failures
- Automatic resubmission of failed jobs with +20 GB additional memory
- Copying of non-OOM error logs to `{BASE_OUTPUT_DIR}/Other_error_logs/` for manual review

```bash
python experiment_monitor.py \
  --base_dir /path/to/BASE_OUTPUT_DIR \
  --logs_dir /path/to/BASE_OUTPUT_DIR/logs \
  --email your.email@institution.edu \
  --interval 2 \
  --failure_lookback 24 \
  --smtp_server localhost \
  --smtp_port 25
```

**Monitor Arguments:**
| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_dir` | `str` | required | Root experiment output directory (same as `BASE_OUTPUT_DIR` in config) |
| `--logs_dir` | `str` | required | SLURM logs directory (typically `{base_dir}/logs`) |
| `--email` | `str` | required | Email address to receive progress reports |
| `--interval` | `float` | `2` | Hours between email notifications |
| `--failure_lookback` | `int` | `24` | How many hours back to scan log files for failures |
| `--smtp_server` | `str` | `"localhost"` | SMTP server for sending emails |
| `--smtp_port` | `int` | `25` | SMTP port |
| `--sender_email` | `str` | `None` | From address (defaults to system user) |
| `--run_once` | flag | — | If set, run a single scan cycle and exit (useful for cron jobs) |

**To run in the background:**
```bash
nohup python experiment_monitor.py \
  --base_dir /path/to/output \
  --logs_dir /path/to/output/logs \
  --email user@example.com \
  --interval 4 > monitor.log 2>&1 &
```

---

## 7. Parameter Reference

### Temporal Abstraction Methods (`d_method`)

| Method | Type | Description |
|---|---|---|
| `equal_frequency` | Unsupervised | Quantile-based binning: each bin contains approximately equal numbers of measurements. Robust to outliers |
| `equal_width` | Unsupervised | Fixed-width binning: divides the value range into equal-sized intervals. Simple but sensitive to outliers |
| `sax` | Unsupervised | Symbolic Aggregate approXimation: uses Gaussian distribution to assign breakpoints so each bin has equal probability mass under a normal distribution |
| `td4c_kullback-leibler` | **Supervised** | Finds bin boundaries that maximize KL divergence between the value distributions of the two classes. Most information-theoretically motivated |
| `td4c_cosine` | **Supervised** | Finds bin boundaries that maximize the cosine distance between class value distributions |
| `td4c_diffsum` | **Supervised** | Finds bin boundaries that maximize the sum of absolute differences between class histograms |
| `td4c_diffmax` | **Supervised** | Finds bin boundaries that maximize the maximum absolute difference between class histograms |
| `knowledge_and_kb_gradient` | Hybrid | Combines domain knowledge (fixed thresholds from `KB_STATES_PATH`) with gradient-detected change points. Requires `KB_STATES_PATH` and `GRADIENT_CUTOFFS_PATH` to be set in `config.py` |
| `persist` | Unsupervised | Persistence-based: a new state is created only when the value changes by more than a persistence threshold |

> **Note:** Supervised methods (`td4c_*`) support `split_event_class` and `event_window` parameters. Unsupervised methods ignore these parameters.

### TIRP Selection Methods (`Tirp_selection_methods`)

There are two families of TIRP selection methods:

**Family 1 — Statistical / structural methods (computed in Stage 2):**
These rank TIRPs based on properties of the STI patterns themselves — how differently they appear in cases vs. controls.

| Method | Description |
|---|---|
| `diff_vertical_support` | Ranks TIRPs by the difference in vertical support (fraction of entities containing the TIRP) between class 1 and class 0. High values indicate TIRPs that appear much more often in cases than controls |
| `diff_horizontal_support` | Ranks by the difference in horizontal support (average number of occurrences per entity) between classes. Captures TIRPs that recur more frequently in cases |
| `diff_mean_duration` | Ranks by the difference in mean duration of the last TIRP prefix between classes. Captures temporal differences in how STI patterns unfold toward the event |
| `mean_squared_mmd` | Averages the squared Mean-Mean Duration difference across **all prefixes** of the TIRP. Considers the entire temporal evolution of the STI pattern |
| `mean_squared_vs` | Averages the squared vertical support difference across all prefixes. Considers discriminative power at all stages of the pattern |
| `all` | Union of all statistical methods: a TIRP is selected if it ranks in the top-K by **any** of the above criteria |
| `random` | Random baseline: TIRPs are randomly selected (seeded). Used for ablation studies |

**Family 2 — Validation-based methods (computed in Stage 3.5):**
These rank TIRPs based on the empirical predictive performance of each TIRP's FCPM model on the training data. Requires `RUN_STAGE3_5_VALIDATION = True` in `config.py`.

| Method | Description |
|---|---|
| `val_AUC` | Selects TIRPs with highest **AUC** measured when running the TIRP's FCPM model on training data |
| `val_AUPRC` | Selects TIRPs with highest **AUPRC** (Area Under Precision-Recall Curve) on training data |
| `val_Precision` | Selects TIRPs with highest **Precision** at the best training-data threshold |
| `val_F1` | Selects TIRPs with highest **F1 score** on training data |

### Aggregation Methods (`aggregation_method`)

| Method | Description |
|---|---|
| `avg` | Simple arithmetic mean of all active TIRP predictions at each `(EntityID, TFS)` time point |
| `max` | Maximum prediction across all active TIRPs. Aggressive — raises score if any single TIRP fires strongly |
| `min` | Minimum prediction. Conservative — only scores high if all TIRPs agree |
| `avg_top_percentage` | Average of the top-scoring fraction of active TIRPs. Balances robustness and sensitivity |
| `weighted_avg` | Weighted mean where each TIRP's prediction is weighted by its vertical support score. Gives more weight to TIRPs that appeared in more training entities |

### Number of Allen Relations (`rel`)

| Value | Relations Included | Description |
|---|---|---|
| `3` | before, meets, overlaps | Basic temporal ordering. Faster mining, less specific patterns |
| `7` | before, meets, overlaps, starts, during, finishes, equals | All Allen relations. Mines finer-grained temporal structure but exponentially increases search space |

---

## 8. Input Data Format

### Temporal Data CSV (`data_path`)
Must contain at minimum:
- `EntityID` — unique entity identifier (e.g., subject ID, session ID)
- `Time` — integer or float timestamp (time units)
- One or more variable columns with numeric measurements

Example:
```
EntityID,Time,Var1,Var2,Var3
entity_001,0,3.2,14.1,0.8
entity_001,1,3.5,14.4,0.9
entity_001,2,,13.9,1.0
entity_002,0,4.1,12.0,0.5
...
```

Missing values are handled via interpolation (controlled by `ig` / interpolation gap).

### Matching CSV (`matching_path`)
Maps each entity to a group for stratified cross-validation splitting:
- `EntityID` — must match the data CSV
- `GroupID` — group identifier (e.g., subject group, trial cohort, site). Entities in the same group are always in the same fold, preventing leakage between related entities.

Example:
```
EntityID,GroupID
entity_001,group_A
entity_002,group_A
entity_003,group_B
...
```

### Event Symbol
The `event_symbol` (default `999`) is a synthetic numeric state code appended to each entity's timeline to mark the target event. During Stage 1 abstraction, this symbol is inserted at the event time, and during Stage 2 mining, only TIRPs whose last symbol is the event symbol are retained.

---

## 9. Fault Tolerance and Resume

### Done-file gating
Every SLURM job writes a `.done` file to a specific path upon **successful** completion. The orchestrator only advances to the next stage once the corresponding `.done` file exists. This means:
- If the cluster goes down or a job times out, re-running `python run_experiment.py` will automatically resume from where execution stopped.
- No job will be submitted twice (the orchestrator tracks submitted jobs in memory and checks `.done` files before each submission).

### Automatic OOM Recovery
When `experiment_monitor.py` is running, it scans SLURM `.err` files every cycle for OOM (Out-of-Memory) indicators:
- `"OOM Killed"`, `"Exceeded memory limit"`, `"out-of-memory"`, `"Out of memory"`, `"Memory limit exceeded"`

For each detected OOM failure:
1. The monitor parses the original `.out` file to reconstruct all job parameters.
2. It generates a new SBATCH script with `original_memory + 20 GB` allocated.
3. It submits the resubmission script automatically via `sbatch`.
4. The resubmission is tracked in `OOM_resubmit_scripts/resubmission_tracker.json` to prevent re-processing the same failure.

### Experiment Status File
The orchestrator maintains a live JSON status file at `{BASE_OUTPUT_DIR}/status/experiment_status.json`. It records per-dataset progress metrics and the last update timestamp, useful for external monitoring dashboards.

---

## 10. Results Format

### Per-Stage Done Files
Each stage creates a `.done` file upon success. The path encodes the full parameter combination, making each experiment uniquely identifiable.

### Final Summary CSV
After all folds and parameter combinations complete, the orchestrator calls `merge_results_files()` which collects all per-fold result CSVs and produces:

```
{BASE_OUTPUT_DIR}/{dataset_name}/results/summery_result_all.csv
```

Each row represents one complete experiment configuration (one fold × one abstraction combination × one mining combination × one aggregation configuration × one TTE_W value × one e_w value) and contains:

| Column Group | Columns |
|---|---|
| Dataset info | `dataset_name`, `fold_num` |
| Abstraction params | `abs_d_method`, `abs_b`, `abs_ig`, `abs_split_event_class`, `abs_event_window` |
| Mining params | `mine_mvs`, `mine_mg`, `mine_rel`, `mine_sf`, `mine_e` |
| Aggregation params | `agg_aggregation_method`, `agg_num_tirps_for_selection` |
| Selection info | `tirp_selection_method` |
| Evaluation params | `TTE_W`, `e_w` |
| Model type | `model_type` |
| **Metrics** | `AUC`, `AUPRC`, `F1`, `Accuracy`, `Precision`, `Recall` |

This file is the primary output for downstream analysis and comparison across hyperparameter configurations.
