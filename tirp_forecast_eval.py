"""
tirp_forecast_eval.py -- reusable per-TIRP forecast evaluation for MARIO.

Given a trained per-TIRP forecast model (a Stage 3 ``CPML`` pickle) this module scores
the model's ``t + HORIZON`` forecasts against ground truth on any data you hand it, as:

  * **classification** -- accuracy / macro-F1 / weighted-F1 / log-loss over the abstracted
    target symbols, next to a majority-class baseline (same numbers Stage 3.5 / Stage 5
    report); and
  * **ordinal regression** -- MSE / MAE / RMSE after mapping each abstracted target
    ``StateID`` to a real value at the centre of its abstraction bin (``BinAvg``). This is
    the piece the classification-only stages don't have; it is exactly the logic the
    ``tirp_models_eda`` notebook prototyped for a single model, generalised here.

The same code path scores **train** and **test** data -- the only thing that differs is
WHERE the durations table and the target STIs come from:

  * TRAIN: durations = the ``durations_merged_df.csv`` Stage 3 already saved per TIRP
           (built on ``Train/KL.txt``); target STIs = the target variable's STIs in
           ``Train/KL.txt``.
  * TEST : durations = built on ``Test/KL.txt`` with ``build_forecast_durations`` (exactly
           what Stage 4 does); target STIs = the target variable's STIs in ``Test/KL.txt``.

so a single ``TIRPForecastEvaluator`` handles both, per-model or across a whole experiment.

Typical use::

    ev = TIRPForecastEvaluator.from_states(
        states="stage1_.../Train/states.csv", horizon=5, target_variable=39,
        data="diabetes/hemoglobin_data.csv")          # data optional (tail handling)

    # one experiment, all TIRPs, on the TRAIN set (reuses saved durations -- cheap):
    train_df = ev.evaluate_experiment_train(built_models_dir, train_target_stis)

    # same models on the TEST set (rebuilds durations on Test/KL.txt per TIRP):
    test_df  = ev.evaluate_experiment_test(built_models_dir, tirp_paths, test_kl_path,
                                           test_target_stis, max_gap, num_relations, epsilon)
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_squared_error, mean_absolute_error,
)

# The forecast feature-matrix builders (shared with Stage 3 / 3.5 / 4) -- keeping the exact
# same builders guarantees the X we score on is bit-identical to what the model trained on.
from CPM_Feature_Matrix.Create_feature_matrix import build_forecast_training_arrays


# =====================================================================
# State -> real value map (abstracted symbol -> centre of its bin)
# =====================================================================

def build_state_value_map(states, target_variable=None, data=None,
                          value_col="TemporalPropertyValue", upper_quantile=0.95):
    """
    Map each abstracted target ``StateID`` to a real value at the centre of its bin, so
    ordinal forecasts can be scored with regression error (MSE / MAE / RMSE).

    ``BinAvg = (BinLow + BinHigh) / 2``. The open tail bins (``BinLow == -inf`` /
    ``BinHigh == +inf``) have no finite centre, so they are closed first:

      * if ``data`` is given (the raw long table with a ``value_col`` column, optionally a
        ``TemporalPropertyID`` column to filter on ``target_variable``) the tails are
        clamped to the data's ``min`` and its ``upper_quantile`` -- exactly the
        ``tirp_models_eda`` notebook's behaviour; otherwise
      * a data-free fallback closes each open tail by one median (finite) bin width, so the
        map is still usable straight from ``states.csv`` with no raw data on hand.

    :param states: path to a ``states.csv`` or an already-loaded DataFrame with columns
                   ``StateID, TemporalPropertyID, BinLow, BinHigh``.
    :param target_variable: if given, keep only that ``TemporalPropertyID``'s states (and
                            filter ``data`` likewise).
    :param data: optional raw long table (path or DataFrame) used only to place the open
                 tail bins; if None the data-free fallback is used.
    :param value_col: value column name in ``data``.
    :param upper_quantile: quantile of ``data[value_col]`` used to close the ``+inf`` tail.
    :return: pandas Series indexed by ``StateID`` giving the real ``BinAvg`` value.
    """
    states_df = pd.read_csv(states) if isinstance(states, str) else states.copy()
    if target_variable is not None and "TemporalPropertyID" in states_df.columns:
        states_df = states_df[states_df["TemporalPropertyID"] == int(target_variable)].copy()
    if states_df.empty:
        raise ValueError("build_state_value_map: no states rows (check target_variable filter).")

    low = states_df["BinLow"].astype(float).to_numpy().copy()
    high = states_df["BinHigh"].astype(float).to_numpy().copy()
    neg_inf = np.isneginf(low)
    pos_inf = np.isposinf(high)

    if data is not None:
        df = pd.read_csv(data) if isinstance(data, str) else data
        if target_variable is not None and "TemporalPropertyID" in df.columns:
            df = df[df["TemporalPropertyID"] == int(target_variable)]
        vals = df[value_col].to_numpy(dtype=float)
        low[neg_inf] = np.nanmin(vals)
        high[pos_inf] = np.nanquantile(vals, upper_quantile)
    else:
        # Data-free fallback: close each open tail by one median finite bin width.
        finite = (~np.isinf(low)) & (~np.isinf(high))
        widths = (high[finite] - low[finite])
        w = float(np.median(widths)) if widths.size else 1.0
        low[neg_inf] = high[neg_inf] - w
        high[pos_inf] = low[pos_inf] + w
        if (neg_inf | pos_inf).any():
            warnings.warn(
                "build_state_value_map: closed open tail bin(s) with a median-width "
                "fallback (no `data` given). Pass `data` for the notebook's exact "
                "data-quantile tails.", stacklevel=2,
            )

    bin_avg = (low + high) / 2.0
    return pd.Series(bin_avg, index=states_df["StateID"].to_numpy(), name="BinAvg")


# =====================================================================
# Metric core (classification + ordinal regression on one aligned set)
# =====================================================================

def forecast_metrics(y_true, y_pred, proba=None, classes=None, state_to_value=None):
    """
    Classification + (optional) ordinal-regression metrics for one aligned set of
    forecasts. This is the single source of truth behind every per-TIRP / per-experiment
    number in this module.

    :param y_true: 1D array of true target ``StateID``s.
    :param y_pred: 1D array of predicted target ``StateID``s (argmax of ``proba``).
    :param proba: optional (n, n_classes) probability matrix (columns aligned to
                  ``classes``) -- needed for log-loss only.
    :param classes: optional array of the symbols ``proba``'s columns correspond to
                    (``model.classes_``).
    :param state_to_value: optional Series ``StateID -> real value``; when given, MSE / MAE
                           / RMSE are added by mapping ``y_true`` / ``y_pred`` to values.
    :return: flat dict of metrics (NaNs where a metric is undefined, e.g. 0 rows).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(len(y_true))

    out = {
        "n_rows": n, "n_classes": 0,
        "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "logloss": np.nan,
        "majority_class": np.nan, "majority_baseline_acc": np.nan, "beats_baseline": False,
        "mse": np.nan, "mae": np.nan, "rmse": np.nan,
    }
    if n == 0:
        return out

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    vals, counts = np.unique(y_true, return_counts=True)
    out["majority_class"] = int(vals[counts.argmax()])
    out["majority_baseline_acc"] = float(counts.max() / n)
    out["beats_baseline"] = bool(out["accuracy"] > out["majority_baseline_acc"])

    if proba is not None and classes is not None:
        classes = np.asarray(classes)
        out["n_classes"] = int(len(classes))
        out["logloss"] = float(log_loss(y_true, proba, labels=classes))

    if state_to_value is not None:
        # Rows whose true/pred symbol is missing from the map are dropped from the
        # regression score (they still count in the classification numbers above).
        yt = pd.Series(y_true).map(state_to_value).to_numpy(dtype=float)
        yp = pd.Series(y_pred).map(state_to_value).to_numpy(dtype=float)
        ok = ~(np.isnan(yt) | np.isnan(yp))
        if ok.any():
            out["mse"] = float(mean_squared_error(yt[ok], yp[ok]))
            out["mae"] = float(mean_absolute_error(yt[ok], yp[ok]))
            out["rmse"] = float(np.sqrt(out["mse"]))
    return out


# =====================================================================
# Loading helpers
# =====================================================================

def load_target_stis(abstraction_output_dir, target_variable, split="Train"):
    """
    Load the forecast target variable's STIs from a Stage 1 KL (the ground-truth symbol
    timeline the ``t + HORIZON`` labels are read off). ``split`` selects ``Train`` or
    ``Test``. Returns DataFrame [EntityID, StartTime, EndTime, StateID].
    """
    # Imported here (not at module top) so importing this module never forces the SKL /
    # KarmaLego stack unless a caller actually parses a KL.
    from run_stage3_build_model import txt_2_csv
    kl_path = os.path.join(abstraction_output_dir, split, "KL.txt")
    if not os.path.exists(kl_path):
        raise FileNotFoundError(f"KL not found: {kl_path}")
    all_stis = txt_2_csv(kl_path)
    target = all_stis[all_stis["TemporalPropertyID"] == int(target_variable)][
        ["EntityID", "StartTime", "EndTime", "StateID"]
    ].copy()
    target["EntityID"] = target["EntityID"].astype(int)
    return target


def find_tirp_model(tirp_dir):
    """Path to the single ``*-CPML.pkl`` under ``tirp_dir/models`` (or None if absent)."""
    matches = glob.glob(os.path.join(tirp_dir, "models", "*-CPML.pkl"))
    return matches[0] if matches else None


def _tirp_name(tirp_dir):
    """Human-readable TIRP id from a ``tirp_<id>`` directory name."""
    base = os.path.basename(os.path.normpath(tirp_dir))
    return base[len("tirp_"):] if base.startswith("tirp_") else base


# =====================================================================
# The evaluator
# =====================================================================

class TIRPForecastEvaluator:
    """
    Scores per-TIRP MARIO forecast models -- one model, or every model in an experiment,
    on train or test data -- returning both classification and ordinal-regression metrics.

    Holds the two things shared across every evaluation: the forecast ``horizon`` and the
    ``state_to_value`` map used for the regression metrics. Build it with
    :meth:`from_states` (from a ``states.csv``) or pass a ready map to ``__init__``.
    """

    def __init__(self, horizon, state_to_value=None):
        self.horizon = int(horizon)
        self.state_to_value = state_to_value

    @classmethod
    def from_states(cls, states, horizon, target_variable=None, data=None,
                    value_col="TemporalPropertyValue", upper_quantile=0.95):
        """Build an evaluator whose regression metrics use the ``states.csv`` bin centres."""
        state_to_value = build_state_value_map(
            states, target_variable=target_variable, data=data,
            value_col=value_col, upper_quantile=upper_quantile,
        )
        return cls(horizon=horizon, state_to_value=state_to_value)

    # --- level 1: a model + an already-built (X, y) matrix ---------------------

    def evaluate_matrix(self, model, X, y_true):
        """
        Score a loaded model on a prebuilt forecast matrix. Predicts the per-symbol
        distribution, takes the argmax as the forecast, and returns
        :func:`forecast_metrics`. Returns the empty-metrics dict when ``X`` has 0 rows.
        """
        if X.shape[0] == 0:
            return forecast_metrics(np.array([]), np.array([]),
                                    state_to_value=self.state_to_value)
        proba = model.predict_proba_matrix(X)
        classes = np.asarray(model.classes_)
        y_pred = classes[proba.argmax(axis=1)]
        return forecast_metrics(y_true, y_pred, proba=proba, classes=classes,
                                state_to_value=self.state_to_value)

    # --- level 2: a model + a durations table + target STIs --------------------

    def evaluate_durations(self, model, durations_df, target_stis):
        """
        Score a model given its durations table and the target STIs. Rebuilds the exact
        labelled matrix (``build_forecast_training_arrays`` -- the covered-only rows, i.e.
        the rows with a real symbol at ``t + horizon``) and scores it. Works for both
        train (saved train durations + train STIs) and test (test durations + test STIs).
        """
        X, y_true, _ = build_forecast_training_arrays(durations_df, target_stis, self.horizon)
        return self.evaluate_matrix(model, X, y_true)

    # --- level 3: a TIRP output directory --------------------------------------

    def evaluate_tirp_dir(self, tirp_dir, target_stis, durations_df=None):
        """
        Score the model stored in ``tirp_dir`` (``tirp_dir/models/*-CPML.pkl``).

        ``durations_df`` selects TRAIN vs TEST:
          * None  -> load the ``durations_merged_df.csv`` Stage 3 saved in ``tirp_dir``
                     (the TRAIN durations) -- the cheap, no-recompute path.
          * given -> score against that durations table (e.g. TEST durations you built on
                     ``Test/KL.txt`` with ``build_forecast_durations``).

        Returns a metrics dict prefixed with ``TIRP_name``. If the model or (train)
        durations file is missing, returns a metrics dict with ``n_rows = 0`` and a
        ``note`` explaining why.
        """
        name = _tirp_name(tirp_dir)
        model_path = find_tirp_model(tirp_dir)
        if model_path is None:
            return {"TIRP_name": name, "note": "no model",
                    **forecast_metrics(np.array([]), np.array([]),
                                       state_to_value=self.state_to_value)}

        if durations_df is None:
            dur_path = os.path.join(tirp_dir, "durations_merged_df.csv")
            if not os.path.exists(dur_path):
                return {"TIRP_name": name, "note": "no durations",
                        **forecast_metrics(np.array([]), np.array([]),
                                           state_to_value=self.state_to_value)}
            durations_df = pd.read_csv(dur_path, low_memory=False)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        metrics = self.evaluate_durations(model, durations_df, target_stis)
        return {"TIRP_name": name, "note": "", **metrics}

    # --- level 4: a whole experiment (every tirp_* dir) ------------------------

    def evaluate_experiment_train(self, built_models_dir, target_stis, verbose=True):
        """
        Score every ``tirp_*`` model under ``built_models_dir`` on the TRAIN set, reusing
        each TIRP's saved ``durations_merged_df.csv`` (no re-detection). ``target_stis``
        must be the TRAIN target STIs. Returns a per-TIRP DataFrame sorted by accuracy.
        """
        tirp_dirs = sorted(glob.glob(os.path.join(built_models_dir, "tirp_*")))
        rows = []
        for d in tirp_dirs:
            if not os.path.isdir(d):
                continue
            row = self.evaluate_tirp_dir(d, target_stis, durations_df=None)
            rows.append(row)
            if verbose:
                print(f"[train] {row['TIRP_name']:>12}  n={row['n_rows']:>6}  "
                      f"acc={row['accuracy']!s:>6.6}  mae={row['mae']!s:>6.6}")
        return _rows_to_frame(rows)

    def evaluate_experiment_test(self, built_models_dir, tirp_paths, test_kl_path,
                                 target_stis, max_gap, num_relations, epsilon,
                                 verbose=True):
        """
        Score every model under ``built_models_dir`` on the TEST set. For each TIRP the
        test durations are rebuilt on ``test_kl_path`` with ``build_forecast_durations``
        (the same detection Stage 4 runs), then scored against the TEST ``target_stis``.

        :param tirp_paths: the TIRP ``.pkl`` object paths (needed to detect prefixes on the
                           test KL); each is matched to its ``tirp_<sanitized_id>`` model dir.
        :param max_gap / num_relations / epsilon: mining params (must match Stage 3/4).
        :return: per-TIRP DataFrame sorted by accuracy.
        """
        # Lazy: only the test path needs the detector-backed durations builder.
        from CPM_Feature_Matrix.Create_feature_matrix import build_forecast_durations
        from run_stage3_build_model import get_sanitized_tirp_id

        rows = []
        for tirp_path in tirp_paths:
            sanitized = get_sanitized_tirp_id(tirp_path)
            tirp_dir = os.path.join(built_models_dir, f"tirp_{sanitized}")
            if find_tirp_model(tirp_dir) is None:
                rows.append({"TIRP_name": sanitized, "note": "no model",
                             **forecast_metrics(np.array([]), np.array([]),
                                                state_to_value=self.state_to_value)})
                continue
            with open(tirp_path, "rb") as f:
                tirp_obj = pickle.load(f)
            durations_df = build_forecast_durations(
                file_path=test_kl_path, max_gap=max_gap, num_relations=num_relations,
                epsilon=epsilon, tirp_obj=tirp_obj.copy_tirp(), output_folder=None,
            )
            row = self.evaluate_tirp_dir(tirp_dir, target_stis, durations_df=durations_df)
            rows.append(row)
            if verbose:
                print(f"[test]  {row['TIRP_name']:>12}  n={row['n_rows']:>6}  "
                      f"acc={row['accuracy']!s:>6.6}  mae={row['mae']!s:>6.6}")
        return _rows_to_frame(rows)


def _rows_to_frame(rows):
    """Assemble per-TIRP metric dicts into a tidy DataFrame sorted by accuracy desc."""
    cols = ["TIRP_name", "note", "n_rows", "n_classes",
            "accuracy", "macro_f1", "weighted_f1", "logloss",
            "majority_class", "majority_baseline_acc", "beats_baseline",
            "mse", "mae", "rmse"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    return df.sort_values("accuracy", ascending=False, na_position="last").reset_index(drop=True)
