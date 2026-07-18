# MARIO Framework
A forecasting model, which is based on CPM models (based on the logic of FCPM/CPML).

## Core Idea
CPM models are event-prediction models, that were created in order to predict the completion of evolving TIRPs, focusing an event of interest. 
These models were tested on classification datasets, that have two classes, case and controls.
Stages in the models training process:
1. The models splitted the data into folds, and using cross fold validation, made sure there is no leakage from test to train when training the models.
2. The abstraction phase - choosing a temporal abstraction (or multiple) to run on train, learn bins boundaries and apply the learned bins on test. This stage results in temporal intervals, on both the train and test sets that represent the original time stamped raw data, just in intervals representation.
3. KarmaLego - temporal patterns mining, which uses the intervals from the previous stage and mined frequent patterns, given predefined parameters (such as minimum vertical support, max_gap, epsilon, number of relations etc.).
4. Given the resulting frequent patterns (TIRPS), the next stage is to learn per each TIRP, that ends with a predefined event of interest, a prediction model. This model learns, for every possible prefix of this TIRP, the probability that the TIRP will be completed (the last STI in the TIRP will appear).
5. Then, in inference time, every timestamp that is revealed, runs all the relevant models (given the evolving TIRPs, for every active prefix, every model returns a probability to completion) and aggregated their predictions into one single probability.


The core idea in MARIO is to use the same logic of learning a prediction model for each TIRP, learn all the evolving prefixes and aggregate these prediction, but instead of predicting whether or not some event of interest will occur, the models forecast the symbol in t + horizon (given the current timestamp in inference time is t).
Hence, every model is trained to predict the occuring symbol in t + horizon, per timestamp t in the training data.
The core logic of training the models relying on the durations between every consecutive prefixes, stays the same.

All definitions that do not appear in this document, can be found with additional explanation in FCPM_README.md.

---

## Build Plan — Adapting FCPM to Forecasting

This section is the working document for turning the FCPM event-prediction pipeline into MARIO's forecasting pipeline. It is filled in stage by stage as each stage's design is agreed. Stages not yet designed are left as TODO.

Global change that motivates everything below: the prediction unit moves from **one label per entity** (case/control classification) to **one target per timestamp per entity** — the symbol occurring at `t + HORIZON`, forecast from data available at `t`.

### Global new parameters

| Param | Default | Meaning |
|---|---|---|
| `HORIZON` | (per-dataset, required) | Forecast lead time. A model trained/evaluated at time `t` predicts the symbol at `t + HORIZON`. Also used as the split embargo size. |
| `HOLDOUT_ENTITY_FRACTION` | `0.2` | Fraction of entities held out entirely from training (never seen). Set to `0.0` for pure within-entity forecasting. |
| `CHRONO_SPLIT_RATIO` | `0.8` | Per-entity chronological cut point, expressed as a fraction of the entity's time span. |
| `SEED` | `2026` | Seeds the random entity holdout partition. |
| `target_variable` | (per-dataset, required) | The `TemporalPropertyID` whose future abstracted state is forecast. "The symbol at `t + HORIZON`" means the state of this variable covering absolute time `t + HORIZON`. |

Removed relative to FCPM: `N_FOLDS`, class stratification, GroupKFold/GroupShuffleSplit, the requirement for `matching.csv`, the `event_symbol`, supervised abstraction methods (`td4c_*`, `tid3`), and `split_event_class` / `event_window`.

### Forecast target label (global definition)

At absolute time `t`, the label MARIO learns/evaluates is:

> the abstracted **state of `target_variable`** whose STI covers time `t + HORIZON`.

Derived from the target variable's STIs (find the STI with `start <= t+HORIZON <= end`; its `stateID` is the label). If no STI covers `t + HORIZON` (gap / missing data), that `(entity, t)` example has no label and is dropped. `HORIZON` is in `TimeStamp` units. This label is constructed in Stage 3; Stage 1 only needs to produce the STIs, which it already does.

---

### Stage 0 — Data Splitting (Forecasting) — DESIGN AGREED

Replaces `run_stage0_split_folds.py`'s K-fold cross-validation with a **hybrid chronological + entity holdout** split. There is no classification, so there are no folds and no stratification.

**Two-axis partition**

1. **Entity axis (new-entity generalization):** a seeded random `HOLDOUT_ENTITY_FRACTION` of entities are held out entirely and never appear in training. They form the `new_entity` test regime.
2. **Time axis (temporal forecasting):** the remaining *train entities* are cut chronologically at `CHRONO_SPLIT_RATIO` of their time span. Their early portion is training; their late portion forms the `seen_future` test regime.

**Embargo (correctness-critical):** for a train entity, the last `HORIZON` time units before the cut are dropped from training. A training example at time `t` has target `t + HORIZON`, so without this gap the targets near the cut would leak into the test region. Holdout entities need no embargo (they are fully disjoint from training).

**Seen-entity context requirement (forward reference to Stage 1/4):** to score a seen entity's future slice at time `t`, MARIO still needs that entity's pre-cut history to know which TIRP prefixes are active at `t`. Therefore `test.csv` carries each seen entity's **full** timeline, with `cut_time` in the manifest marking where scoring begins. Holdout entities are scored across their whole timeline after a warm-up. The exact inference mechanics are handled in Stage 1/4.

**Design defaults chosen:** cut by **time span** (`min→max` of `TimeStamp`), and a **single tagged `test.csv`** (not separate files) so Stage 4/5 stay near-unchanged and can report metrics overall and per regime.

**Input format (verified against real data):** the pipeline uses **Hugobot long format** — `EntityID, TemporalPropertyID, TimeStamp, TemporalPropertyValue` — *not* the wide `EntityID, Time, Var…` shape. Each `TemporalPropertyID` is a measurement variable over `TimeStamp`. Rows with **`TemporalPropertyID == -1`** are per-entity **classification labels** (`TimeStamp = 0`, value ∈ {0,1}), which Hugobot extracts for supervised methods and then drops. In MARIO these class rows are irrelevant: Stage 0 **excludes them from the chronological cut** (cut is computed over measurement rows only) but **carries them through untouched** so unsupervised abstraction is unaffected. Because they sit at `TimeStamp = 0` they naturally land in the train slice. Consequence for Stage 1: **supervised abstraction methods (`td4c_*`, `tid3`) and `split_event_class` depend on these now-meaningless class labels and must be disabled/redesigned.**

**Inputs**

| Argument | Description |
|---|---|
| `--dataset_name` | Dataset identifier string |
| `--data_path` | Path to raw data CSV |
| `--horizon` | Forecast horizon / embargo size |
| `--holdout_entity_fraction` | Fraction of entities fully held out |
| `--chrono_split_ratio` | Per-entity past/future cut as a fraction of time span |
| `--seed` | Random seed for the entity holdout partition |
| `--split_dir` | Output directory for the split (reuses the former `fold_1/` slot) |

`--matching_path` and `--n_folds` are removed.

**Outputs**

```
{split_dir}/
├── train.csv          # train-entities only, pre-cut rows minus the HORIZON embargo
├── test.csv           # seen entities (full timeline, scored after cut_time) + holdout entities (full timeline)
├── split_manifest.csv # EntityID, role={train|holdout}, test_regime={seen_future|new_entity}, cut_time
└── stage0_split.done
```

Notes:
- A seen entity intentionally appears in both `train.csv` (early rows) and `test.csv` (full timeline for context); only `t > cut_time` is scored in evaluation.
- To minimize downstream path churn, the single split directory reuses the existing `fold_1/` directory slot instead of renaming path conventions across stages.

**Resolved**
- **Entity independence:** each `EntityID` is an independent entity, so the random entity holdout needs no group-awareness. `matching.csv` / `GroupID` are removed from MARIO entirely.

**Open / TODO**
- Choose the cut convention if `TimeStamp` is irregularly sampled (time-span cut can yield very uneven row counts across entities).

---

### Stage 1 — Temporal Abstraction (Forecasting) — IMPLEMENTED

Same core job as FCPM — learn discretization bins on the train slice, apply them to test, merge each variable's symbolic series into STIs, write KL files — but stripped of all classification machinery.

**Implemented changes (`run_stage1_abstraction.py`)**
- **Unsupervised methods only.** `is_supervised_method()` rejects `td4c_*`, `tid3*`, and `mdlp` (they bin to separate classes, which no longer exist) with a clear error before any work. Allowed: `equal_frequency`, `equal_width`, `sax`, `persist`, `gradient`, `knowledge`/`kb_gradient`.
- **Removed the event/class split path** entirely: `prepare_event_based_split`, the `split_event_class` branch, and the `--split_event_class` / `--event_window` CLI args. Abstraction always runs in standard mode (learn on train, apply to test).
- **No `event_symbol` insertion** (there never was any in Stage 1; it lived in Stage 2's now-deleted `split_data`).
- **Target variable** is abstracted like any other variable; its STIs are what Stage 3 reads to build the forecast label.

**Single KL file — how it actually works.** Hugobot *always* writes `KL.txt` containing **all** entities (`core.py` / `utils.save_results`), and only additionally writes per-class `KL-class-*.txt` when class labels are present. We do **not** strip the `-1` class rows (that would empty `entity-class-relations.csv`, which other tooling may read), so those extra per-class files are still produced — but they are **ignored**. Stage 2 consumes `Train/KL.txt`. So "single KL file" is satisfied in practice: `KL.txt` is the one file that matters.

**Unchanged and still correct**
- Learn-bins-on-train / apply-to-test (bins from train slices, applied to the seen-future and holdout timelines in `test.csv`).
- A seen entity's early portion legitimately appears in both the Train KL (learned+applied) and Test KL (applied) — different files, different purposes.

**Arguments (revised):** `--train_data_file`, `--test_data_file`, `--d_method` (unsupervised only), `--num_of_bins`, `--interpolation_gap`, `--abstraction_output_dir`.

**Outputs**
```
{abstraction_output_dir}/
├── Train/KL.txt          # all train-entity STIs  (+ ignored KL-class-*.txt, states.csv, symbolic_time_series.csv)
└── Test/KL.txt           # all test-entity STIs (seen full timelines + holdout)  (+ ignored extras)
```

**Verified:** real diabetes subsample runs Stage 0 → Stage 1 → Stage 2 end to end. Stage 1 emits `Train/KL.txt` (`numberOfEntities` = 64 train entities); Stage 2 mines it and computes VS fractions against those 64. Supervised-method guard rejects `td4c_cosine`.
### Stage 2 — TIRP Mining (Forecasting) — IMPLEMENTED

**Decisions**
- **Diabetes settings:** `target_variable = 1` (TPID 1), `horizon = 5`.
- **Keep ALL TIRPs.** The event-of-interest filter is deleted — no "last symbol == event_symbol", no `size > 2` requirement. Every mined frequent TIRP will get a Stage 3 model that forecasts the target symbol at `t + horizon`.
- **Autoregressive patterns allowed:** the target variable's own *past* STIs may appear inside mined TIRPs. Not leakage (only pre-`t` STIs used) and essential for the univariate case (where TPID 1 is the only variable, so every TIRP is built from the target's own history).
- **Selection = vertical-support range.** Replace the class-based `diff_*` / top-K scoring with a simple filter: keep TIRPs whose vertical support **as a fraction of training entities** falls in `[TIRP_VS_MIN, TIRP_VS_MAX]` (config; default `[0.0, 1.0]` = keep all). The "what makes a TIRP good" question (likely correlation-based) is deferred.

**Implementation (`run_stage2_mining.py`)**
- Removed `split_data` (window splitting + synthetic `999` event insertion) and the entire `TirpSelection` class-diff scoring path.
- `load_training_kl` reads a single `Train/KL.txt` (MARIO Stage 1 output) or, for backward compatibility until Stage 1 is reworked, concatenates any `Train/KL-class-*.txt` (together = all training entities) into one KL file under `KL_data/KL.txt`.
- Runs KarmaLego on that single file, then filters `frequent_tirps` by VS fraction and saves the kept TIRP `.pkl`s.
- `--event_symbol` is still accepted but ignored (orchestrator compatibility). New args: `--vs_min`, `--vs_max`.

**Outputs**
```
{mining_run_dir}/
├── KL_data/KL.txt                 # single combined training KL used for mining
├── tirps/<to_string()>.pkl        # one pickle per kept TIRP
└── tirp_selection_scores.csv      # TIRP_Representation, Size, Vertical_Support,
                                    #   Vertical_Support_Fraction, Selected
```

**Verified:** mining + VS-range filtering (inclusion and exclusion), pkl saving, and both the single-file and legacy two-class-file inputs, on synthetic KL data.

**Open / TODO**
- Design the real "good TIRP" criterion (correlation between the TIRP prefix and the future target symbol) to replace/augment the VS-range filter.
- Once Stage 1 emits `KL.txt`, the legacy `KL-class-*.txt` concatenation branch can be removed.
- Stage 5's selection consumes `tirp_selection_scores.csv` via `Binary_{method}#{K}` columns — that contract will change when Stage 5 is reworked.
### Stage 3 — Per-TIRP Forecast Model Building — IMPLEMENTED

Same skeleton as FCPM's model-building stage (batch over a `tirp_list_file`, one output dir per TIRP, `.done` markers for resumability), but the per-TIRP unit changes from a set of prefix TTE/classification models to **one multiclass forecast model** predicting the `target_variable`'s symbol at `t + HORIZON`.

**Decisions**
- **One model per TIRP.** FCPM's case/control split, synthetic event insertion, and the per-prefix TTE models are all gone. Each mined TIRP gets a single multiclass `CPML` (XGBoost) model whose label is the future target symbol.
- **Label source.** The forecast label comes from the `target_variable`'s STIs (loaded once per batch from Stage 1's `Train/KL.txt`), not from any class file — consistent with the global forecast-target definition above.
- **Event-free durations.** The TIRP's evolving prefixes and their per-instance TIEP durations are detected on the training KL with no event-of-interest logic.

**Implementation (`run_stage3_build_model.py`)**
- `prepare_batch_data(abstraction_output_dir, target_variable)` (once per batch): parses `Train/KL.txt` (`txt_2_csv`) and returns `{train_kl_path, target_stis}`, where `target_stis` is the target variable's STIs `[EntityID, StartTime, EndTime, StateID]` used to label every row.
- `process_single_tirp(...)` (per TIRP):
  1. `build_forecast_durations(...)` (event-free) → per-instance durations table (`durations_merged_df.csv`), one row per detected TIRP instance with the consecutive-TIEP durations plus `TFS` (absolute time of the first TIEP);
  2. `build_forecast_training_arrays(durations_df, target_stis, horizon)` expands to one row per `(entity, absolute t)` and labels each with the target STI covering `t + HORIZON`, **dropping rows whose `t + HORIZON` no STI covers**;
  3. trains ONE multiclass `CPML.fit_matrix(X, y, feature_names)` and saves it. `CPML.classes_` holds the real target symbols in `predict_proba_matrix` column order.
- If a TIRP yields zero labeled rows, it is marked done with no model.
- New CLI: `--abstraction_output_dir`, `--tirp_model_run_dir`, `--tirp_list_file`, `--max_gap`, `--num_relations`, `--epsilon`, `--target_variable`, `--horizon`.

**Outputs**
```
{tirp_model_run_dir}/tirp_<id>/
├── durations_merged_df.csv              # one row per detected TIRP instance (+ TFS)
├── models/<tirp_str>-CPML.pkl           # single multiclass forecast model
└── stage3_build_<id>.done               # resumability marker
```

**Verified:** `explore_stage3.ipynb` runs the real `prepare_batch_data` / `process_single_tirp` locally (Stages 0→1→2 auto-run subsampled if missing) on the diabetes subsample and inspects the durations table, the labeled matrix + symbol distribution, and the trained model's `classes_` / `predict_proba_matrix`.

**Open / TODO**
- The mining params (`max_gap`, `num_relations`, `epsilon`) must be passed to match Stage 2 so the durations detector re-finds each TIRP consistently — currently the caller's responsibility.

### Stage 3.5 — Train-Set Forecast-Accuracy Check — IMPLEMENTED

A lightweight sanity check (not part of the FCPM pipeline; **not** Stage 5). For each built model it rebuilds the model's own training matrix (from the saved `durations_merged_df.csv` + target STIs, re-labeled at `t + HORIZON`) and scores the model back on it.

**Implementation (`run_stage3_5_validation.py`)**
- `load_target_stis(abstraction_output_dir, target_variable)` + `run_train_validation_for_tirp(tirp_path, tirp_model_run_dir, target_stis, horizon)`.
- Reports multiclass **accuracy / macro-F1 / weighted-F1 / log-loss** against a **majority-class baseline** — a model is informative when `train_accuracy` clears `majority_baseline_acc`.

**Outputs**
```
{tirp_model_run_dir}/tirp_<id>/
├── train_summary_metrics.csv            # accuracy, macro/weighted F1, log-loss, majority baseline
└── stage3_val_<id>.done
```

**Verified:** section 8 of `explore_stage3.ipynb` runs the real Stage 3.5 functions on the models built above and reports the per-TIRP metrics vs baseline.

### Stage 4 — Continuous Forecasting Inference — IMPLEMENTED

The **test-set twin of Stage 3**: same batch-by-TIRP shape, same `.done` resumability, same event-free durations detector — but it detects on the **Test** KL and *scores* each TIRP's Stage 3 model instead of training one. The old FCPM implementation (`FCPM_Package.Continuous_Prediction`, FCPM+TTE per-prefix models, `entity-class-relations.csv`) is entirely dropped.

**Decisions**
- **Batch by TIRP** (mirrors Stage 3), not by entity as FCPM did — the expensive step is detecting a TIRP's prefixes on the Test KL, so one detection per TIRP covers all test entities at once.
- **No rows dropped at inference.** Every `(entity, absolute t)` drawn from the evolving prefixes gets a forecast, even where `t + HORIZON` has no covering target STI (a gap / the end of data). Training drops those (no label); inference keeps them and flags `covered = False`.
- **Ground truth from the Test KL.** The target variable's STIs are loaded from `Test/KL.txt` (Stage 3 used `Train/KL.txt`) to supply each row's true `t + HORIZON` symbol for later evaluation.
- **Full probability vector per row.** Each forecast is the model's `predict_proba_matrix` distribution over the target symbols (`P_<symbol>` columns, aligned to the model's `classes_`), so Stage 5 can aggregate soft predictions across TIRPs.

**Implementation (`run_stage4_predict_entities.py`)**
- `prepare_batch_data(abstraction_output_dir, target_variable)` (once per batch): returns `{test_kl_path, target_stis}` — target STIs parsed from `Test/KL.txt`.
- `process_single_tirp(...)` (per TIRP):
  1. `build_forecast_durations(...)` on the **Test** KL → per-instance test durations (`durations_merged_df.csv`);
  2. `build_forecast_inference_matrix(durations_df, target_stis, horizon)` (new, in `Create_feature_matrix.py`) → `(X, meta, feature_names)` where `meta` is a row-aligned table `[EntityID, current_time, TFS, y_true, covered]` and **no rows are dropped**;
  3. loads the Stage 3 model (`find_tirp_model` → `tirp_<id>/models/*-CPML.pkl`) and calls `predict_proba_matrix(X)`;
  4. writes `forecasts.csv.gz` = `meta` + one `P_<symbol>` column per target symbol + `pred_symbol` (argmax convenience).
- CLI: `--abstraction_output_dir`, `--built_models_base_dir`, `--prediction_output_dir`, `--tirp_list_file`, `--max_gap`, `--num_relations`, `--epsilon`, `--target_variable`, `--horizon`.

**Outputs**
```
{prediction_output_dir}/tirp_<id>/
├── durations_merged_df.csv              # per-instance TEST durations (+ TFS)
├── forecasts.csv.gz                     # EntityID, current_time, TFS, y_true, covered,
│                                        #   P_<symbol>..., pred_symbol
└── stage4_predict_<id>.done             # resumability marker
```

**Orchestrator (`run_experiment.py`)** — converted to match. `submit_stage4_job` (old, entity-batched) was replaced by **`submit_stage4_batch_job`**, batched by TIRP exactly like `submit_stage3_batch_job`: it waits for Stage 1's `Test/KL.txt`, points at the Stage 3 `feature_matrix/` models, and threads `--tirp_list_file` + mining params + `--target_variable` + `--horizon`. The main loop now dispatches Stage 4 as TIRP batches (`batch_status_stage4/stage4_batch_XXXX.done` for submission de-dup; per-TIRP `stage4_predict_<id>.done` for completion) and runs the one-time post-3.5 TIRP-selection-score update via a `stage4_selection_updated` guard. Batch size honors `TIRP_BATCH_SIZE_STAGE4` (falls back to the Stage 3 / global size). The old entity-batch machinery (`get_test_entity_ids`, `ENTITY_BATCH_SIZE_FOR_PREDICTION`, `stage4_trigger_status`) is no longer used by Stage 4.

**Verified:** `explore_stage4.ipynb` runs the real `prepare_batch_data` / `process_single_tirp` locally (Stages 0→1→2→3 auto-run/built subsampled if missing) on the diabetes subsample. Forecasts checked: probability rows sum to 1, `covered` flags match target-STI coverage, and per-TIRP test accuracy is reported against a majority baseline. (A single weak TIRP need not beat the baseline alone — that is Stage 5's aggregation job.) Orchestrator: `run_experiment.py` compiles clean and the arguments `submit_stage4_batch_job` emits parse against the Stage 4 script's argparse. (Full SLURM dispatch is not exercisable locally.)

### Stage 5 — Aggregation & Forecast Evaluation — IMPLEMENTED

Full MARIO rewrite of `run_stage5_aggregation_eval.py` (the FCPM ROC/TTE/`Binary_{method}#{K}` path is entirely dropped). Consumes Stage 4's per-TIRP `forecasts.csv.gz` and collapses them into **one** forecast per `(entity, t)`.

**Decisions**
- **Active set = TIRPs with a forecast row at `t`.** Stage 4 emits a row for every integer `t` a prefix instance spans, so "has a row at `t`" ≡ "prefix active at `t`". A `context_window = C` param generalises this: a TIRP is active at `t` if its most-recent row falls in `[t-C, t]` (a grace period after the pattern completes), contributing that most-recent in-window distribution. `C = 0` (default) is exact.
- **Symbol alignment.** Each Stage 3 model only predicts the target symbols it saw in training, so TIRPs carry different `P_<symbol>` columns. Every distribution is reindexed to the **global symbol union**, missing symbols filled `0.0` (the model assigns them zero prob), so each row still sums to 1. Overlapping instances of the same TIRP at the same `t` are mean-reduced first (one vote per TIRP).
- **Aggregation method** is a hyperparameter: `average` (mean of active TIRPs' distributions) or `max` (per-symbol maximum); argmax → `pred_symbol`. Rows are renormalised to sum to exactly 1 (removes XGBoost softprob's ~1e-7 drift that trips `log_loss`; a real renormalisation for `max`).
- **Evaluation** on `covered` rows only, filtered by the Stage 0 `split_manifest.csv`: `seen_future` scored where `current_time > cut_time`; `new_entity` (holdout, `cut_time` NaN) scored across the timeline after an optional per-entity `warmup`. Reports accuracy / macro-F1 / weighted-F1 / log-loss vs a majority baseline, **overall and per regime**.

**Implementation (`run_stage5_aggregation_eval.py`)** — `load_all_forecasts` → `aggregate_forecasts(forecasts, symbols, context_window, method)` → `attach_regime_and_filter(agg, manifest, warmup)` → `evaluate(agg, symbols)`. CLI: `--prediction_output_dir --split_manifest_path --results_output_dir --aggregation_method --context_window --warmup`.

**Outputs** (into `results_output_dir`):
```
aggregated_forecasts.csv.gz   # per (entity, t): P_<symbol>..., pred_symbol, y_true, covered,
                              #   n_active_tirps, test_regime, cut_time, scored
stage5_metrics.csv            # one row per scope: overall / seen_future / new_entity
stage5_aggregation.done
```

**Config (config.py):** `STAGE5_AGGREGATION_METHOD='average'`, `STAGE5_CONTEXT_WINDOW=0`, `STAGE5_WARMUP=0`.

**Orchestrator (`run_experiment.py`) — CONVERTED.** The FCPM combinatorial dispatch (`aggregation_method` × `tirp_selection_method` × `TTE_W` × `e_w`, `Binary_{method}#{K}` selection, FCPM/CPML nicknames) is gone. `model_params_keys` is now the **aggregation hyperparameter grid** `["aggregation_method","context_window","warmup"]`; `generate_parameter_combinations` builds its Cartesian product. `submit_stage5_job(dataset_params, fold_num, abs_combo, mine_combo, model_combo, fold_dir, mining_run_dir, log_dir, base_dir)` submits ONE job per grid combo, reading `mining_run_dir/predictions` (Stage 4) + `fold_dir/split_manifest.csv` (Stage 0), writing to `mining_run_dir/results/<agg_param_string>/`; done = that subdir's `stage5_aggregation.done`. The main-loop Stage 5 block loops over `model_combinations` only. `experiment_params.py` (diabetes) sets `aggregation_method=['average','max']`, `context_window=[0,5]`, `warmup=[0]` and drops `num_tirps_for_selection`/`TTE_W`/`e_w`.

**Local debug notebook:** `explore_stage5.ipynb` (rewritten to MARIO) bootstraps Stages 0-4 subsampled, then runs the real `run_single_aggregation_evaluation` + the broken-open `load_all_forecasts`/`aggregate_forecasts`/`attach_regime_and_filter`/`evaluate`, explores accuracy-by-active-TIRP-count and per regime, and sweeps the 2×2 grid.

**Verified (2026-07-18):** run on the diabetes subsample (25 TIRPs' Stage 4 forecasts) — script directly, all 24 notebook cells end-to-end, and the orchestrator arg string parsing. `average, context=0`: 4205 `(entity,t)` forecasts, 1186 scored, **accuracy 0.68 vs 0.27 majority** (beats baseline in both regimes — aggregation lifts the individually-weak ~0.43–0.51 single-TIRP models). Grid sweep: `max` edges `average` (overall 0.68 vs 0.68; **seen_future 0.71**), `context=5` slightly lowers accuracy (stale in-window votes). All distributions sum to 1.0, no warnings. Not SLURM-dispatch-tested locally (no cluster).