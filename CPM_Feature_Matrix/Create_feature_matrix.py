import numpy as np
import pandas as pd
import pickle
from SKL.Tirp_new import TIRP
from SKL.Tirp_new import *
from SKL.Tirp_detection import TIRPDetector
from CPM_Feature_Matrix.TIRP_prefixs_evolving_TIRPs import *
import os


def extract_tiep_durations(tirp, max_gap, num_relations=7,col_names=None):
    """
    Extract durations between consecutive TIEPs for all entities in the TIRP.

    Instead of relying on the first row to set the TIEP order, we use the TIRP object's
    compute_tiep_order() method. This method returns a list of tuples (each tuple representing a group of tieps
    that occur at the same time) in the following format:

        [('10+', '3+'), ('3-',), ('10-',), ('8+',), ('8-',), ('7+',), ('7-',)]

    For each entity (row) in tirp.instances (a Pandas DataFrame), the function extracts a timestamp
    for each tiep group according to:
       - If a tiep ends with '+', use the start time from the corresponding symbol column.
       - If it ends with '-', use the end time.

    The function then computes the duration (difference) between consecutive tiep groups and returns
    a DataFrame with:
       - "EntityID"
       - "TFS" (the timestamp of the first tiep group)
       - One column per consecutive tiep pair (named by the first tiep in each group pair)
    """

    # Use the TIRP object's compute_tiep_order() to get the correct ordering of TIEPs.
    # (It must return a list of tuples, each tuple containing tiep strings, e.g. "A+" or "B-".)
    # print(f'tirp {tirp.to_string()}')
    if tirp.instances.empty:
        return pd.DataFrame(columns=["EntityID","TFS"]+col_names)
    

    try:
        tiep_order = tirp.compute_tiep_order(num_relations)  # e.g., [('10+', '3+'), ('3-',), ('10-',), ('8+',), ('8-',), ('7+',), ('7-',)]
    except Exception as e:
        return pd.DataFrame(columns=["EntityID","TFS"]+col_names)  # Return empty DataFrame if there's an error in computing tiep_order.
    
    # For each entity row, extract timestamps in the order specified by tiep_order.
    # Assume that tirp.instances is a DataFrame with columns: "EntityID" and one column per symbol.
    # In each row, the cell for a symbol is a tuple: (start, end).
    timestamps_list = []
    for _, row in tirp.instances.iterrows():
        entity_timestamps = []
        for group in tiep_order:
            for tiep in group:
                # Use the first tiep from the tuple (all in the tuple are simultaneous)
                tiep_str = tiep  # e.g., "A+" or "B-"
                symbol = tiep_str[:-1]  # extract the symbol part, e.g., "A"
                sign = tiep_str[-1]  # either '+' or '-'
                cell = row[symbol]
                if pd.isna(cell):
                    ts = np.nan
                else:
                    # For '+' take the start, for '-' take the end.
                    ts = cell[0] if sign == '+' else cell[1]
                entity_timestamps.append(ts)
        timestamps_list.append(entity_timestamps)

    # Convert the list of timestamps to a NumPy array.
    timestamps_array = np.array(timestamps_list)

    # Compute durations between consecutive timestamps (axis=1).
    durations = np.diff(timestamps_array, axis=1)
    np.clip(durations, a_min=None, a_max=int(1.2 * max_gap), out=durations)

    # # Create a flattened list of all TIEPs in their correct processing order.
    # flattened_tieps = []
    # for group in tiep_order:
    #     for tiep_str_in_group in group:
    #         flattened_tieps.append(tiep_str_in_group)
    # col_names = [f"({flattened_tieps[i]}, {flattened_tieps[i+1]})" 
    #                  for i in range(len(flattened_tieps) - 1)]

    # # Create column names for durations between all consecutive TIEPs from the flattened list.
    # if len(flattened_tieps) < 2:
    #     col_names = [] # No durations if less than 2 TIEPs overall
    # else:
    #     col_names = [f"({flattened_tieps[i]}, {flattened_tieps[i+1]})" 
    #                  for i in range(len(flattened_tieps) - 1)]

    # Build the resulting DataFrame.
    duration_df = pd.DataFrame(durations[:, :len(col_names)], columns=col_names)
    # Insert the EntityID column at the beginning.
    duration_df.insert(0, "EntityID", tirp.instances["EntityID"].values)
    # Add a column "TFS" for the timestamp of the first tiep group.
    duration_df["TFS"] = timestamps_array[:, 0]

    # Filter out records that have any negative value in the duration columns
    if col_names:
        mask = ~(duration_df[col_names] < 0).any(axis=1)
        duration_df = duration_df[mask]

    if duration_df.empty:
        return pd.DataFrame(columns=["EntityID","TFS"]+col_names)

    return duration_df


def create_feature_matrix_for_TTE(duration_df):
    """
    Generate a feature matrix where each time step is expanded into multiple rows,
    tracking duration progression, binary observation flags, and Time-To-Event (TTE).

    Each original entity instance is assigned a unique instance_ID to allow tracking.

    :param duration_df: Pandas DataFrame containing durations for each entity.
    :return: Expanded DataFrame with additional features for each time step.
    """
    # Extract column names for durations
    entity_col = "EntityID"
    time_cols = [col for col in duration_df.columns if col not in [entity_col, "TFS", 'event_time']]

    # Compute total duration per entity (sum of all duration columns)
    total_duration = duration_df[time_cols].sum(axis=1)

    # Initialize a list to store expanded rows
    expanded_rows = []

    # Process each entity separately
    for idx, row in duration_df.iterrows():
        entity_id = row[entity_col]
        tfs = row["TFS"]
        max_time = total_duration[idx]
        instance_id = f"{idx}"  # Create a unique identifier for the instance

        # Create columns for observed status and progress tracking
        duration_values = row[time_cols].values
        time_steps = np.arange(max_time + 1)  # From 0 to total duration

        # Initialize feature storage for this entity
        entity_rows = []

        # Track cumulative sum to determine when each duration starts
        cumulative_times = np.cumsum(duration_values)
        start_times = np.insert(cumulative_times[:-1], 0, 0)  # Start times per duration

        for t in time_steps:
            row_data = {
                entity_col: entity_id,
                "instance_ID": instance_id,  # Add instance identifier
                "TFS": t
            }

            # Compute observed and progress for each duration
            for i, col_name in enumerate(time_cols):
                observed_col = f"{col_name}_Binary"
                progress_col = f"{col_name}_Duration"

                # Check if the time step is within this duration
                row_data[observed_col] = 1 if t >= start_times[i] else 0
                row_data[progress_col] = min(t - start_times[i], duration_values[i]) if t >= start_times[i] else 0

            # Compute TTE (Time-To-Event)
            row_data["TTE"] = max_time - t  # Time remaining until the last event

            # Store row
            entity_rows.append(row_data)

        # Append all rows for this entity
        expanded_rows.extend(entity_rows)

    # Convert the list of rows into a DataFrame
    tte_feature_matrix = pd.DataFrame(expanded_rows)

    return tte_feature_matrix


def create_test_feature_matrix(duration_df, event_time_dict):
    """
    Generate a feature matrix for testing where each time step is expanded into multiple rows,
    tracking duration progression, binary observation flags, and Time-To-Event (TTE).

    For each entity:
      - Time steps are absolute, starting from the TFS value given in the "TFS" column.
      - The function uses an event_time_dict (mapping EntityID -> event_time).
      - For each time step, if the entity exists in event_time_dict, then
            TTE = event_time_dict[EntityID] - current_time;
        otherwise, TTE is set to NaN.
      - Additionally, all computed features (binary observation, progress) are computed relative to the
        actual start time (TFS) of the entity.

    :param duration_df: Pandas DataFrame containing durations for each entity.
                        Expected columns: "EntityID", "TFS", plus duration columns.
    :param event_time_dict: Dictionary mapping EntityID to the event's absolute start time.
    :retu
    rn: Expanded DataFrame with additional features for each time step.
    """
    entity_col = "EntityID"
    # Identify duration columns: all columns except "EntityID" and "TFS" (and "event_time" if present).
    time_cols = [col for col in duration_df.columns if col not in [entity_col, "TFS", "event_time"]]

    # Compute total duration per entity (sum of all duration columns).
    total_duration = duration_df[time_cols].sum(axis=1)

    expanded_rows = []

    # Process each entity (row) in the DataFrame.
    for idx, row in duration_df.iterrows():
        entity_id = row[entity_col]
        tfs = row["TFS"]  # Absolute start time for this entity.
        max_time = total_duration[idx]
        instance_id = str(idx)  # Unique identifier for the instance.

        # Get duration values (as an array).
        duration_values = row[time_cols].values
        # Create absolute time steps: from tfs to tfs + max_time (inclusive).
        time_steps = np.arange(tfs, tfs + max_time + 1)

        # Compute cumulative sums of durations and then compute the absolute start time for each duration interval.
        cumulative_times = np.cumsum(duration_values)
        # The first interval starts at TFS; subsequent intervals start at TFS + cumulative sum (excluding the last interval).
        start_times = np.insert(tfs + cumulative_times[:-1], 0, tfs)

        # Determine if the event is defined for this entity.
        has_event = entity_id in event_time_dict
        event_time = event_time_dict.get(entity_id, np.nan)

        for current_time in time_steps:
            row_data = {
                entity_col: entity_id,
                "instance_ID": instance_id,
                "instance_start_time": tfs,  # Keep the original TFS value.
                "current_time": current_time
            }

            # For each duration column, compute:
            #   - a binary flag indicating whether the interval has started,
            #   - the progress within that duration (capped at the duration length).
            for i, col_name in enumerate(time_cols):
                observed_col = f"{col_name}_Binary"
                progress_col = f"{col_name}_Duration"
                if current_time >= start_times[i]:
                    row_data[observed_col] = 1
                    row_data[progress_col] = min(current_time - start_times[i], duration_values[i])
                else:
                    row_data[observed_col] = 0
                    row_data[progress_col] = 0

            # Compute TTE only if the event exists for this entity.
            if has_event:
                row_data["TTE"] = event_time - current_time
                row_data["event_time"] = event_time
            else:
                row_data["TTE"] = np.nan
                row_data["event_time"] = np.nan

            expanded_rows.append(row_data)

    return pd.DataFrame(expanded_rows)


def build_cpml_training_arrays(durations_df, class_map):
    """
    Build the CPML (XGBoost) training feature matrix directly as a compact integer
    NumPy array instead of a pandas DataFrame.

    This is a memory-lean replacement for calling ``create_test_feature_matrix`` just to
    feed XGBoost: that function materializes a Python list of millions of dicts (one per
    (entity, timestamp) row), which OOMs on large TIRPs. Here we preallocate ONE integer
    array and fill it vectorized per entity, storing only the columns XGBoost trains on.

    The feature values are bit-identical to the ``*_Binary`` / ``*_Duration`` columns of
    ``create_test_feature_matrix``: those features depend only on
    ``current_time - start_time`` (the absolute ``TFS`` cancels), so we generate them in
    cheap relative time. NaN durations (left by ``merge_prefix_tables``) are handled
    exactly as the original does -- ``sum`` skips them, ``cumsum`` propagates them, and
    Python's ``min(x, nan) == x`` leaves a NaN-duration column uncapped. NaN never leaks
    into the integer output (un-started cells are 0; started cells are finite integers).

    :param durations_df: DataFrame with columns ["EntityID", <duration cols...>, "TFS"].
    :param class_map: dict EntityID(int) -> ClassID(int); entities not present default to 0.
    :return: (X, y, feature_names)
             X            : np.ndarray (n_rows, 2*D) integer feature matrix
             y            : np.ndarray (n_rows,) int8 outcome_class per row
             feature_names: list[str] of length 2*D giving the column order of X
                            (matches the names create_test_feature_matrix emits).
    """
    entity_col = "EntityID"
    time_cols = [c for c in durations_df.columns if c not in (entity_col, "TFS", "event_time")]
    D = len(time_cols)

    feature_names = []
    for c in time_cols:
        feature_names.append(f"{c}_Binary")
        feature_names.append(f"{c}_Duration")

    if durations_df.empty or D == 0:
        return (np.empty((0, 2 * D), dtype=np.int16),
                np.empty((0,), dtype=np.int8),
                feature_names)

    # Duration cells as float (may contain NaN); they are whole numbers, so round to guard
    # against float text like "62.0" and any binary float artifacts.
    dur = np.round(durations_df[time_cols].to_numpy(dtype=np.float64))
    ent = durations_df[entity_col].to_numpy()

    with np.errstate(invalid="ignore"):
        # Per-instance span = sum of non-NaN durations (matches pandas sum skipna=True).
        spans_int = np.nansum(dur, axis=1).astype(np.int64)
    n = int(spans_int.sum() + len(durations_df))  # +1 row per instance (t = 0..span)

    # Every output cell is bounded by its instance span, so pick the smallest safe int.
    max_val = int(spans_int.max()) if len(spans_int) else 0
    idtype = np.int16 if max_val <= np.iinfo(np.int16).max else np.int32

    X = np.empty((n, 2 * D), dtype=idtype)
    y = np.empty((n,), dtype=np.int8)

    pos = 0
    for r in range(len(durations_df)):
        dv = dur[r]                       # (D,) float, may contain NaN
        span = int(spans_int[r])
        T = span + 1
        t = np.arange(T)                  # relative timestamps 0..span

        # Relative interval start times; cumsum propagates NaN exactly like the original.
        start_rel = np.empty(D, dtype=np.float64)
        start_rel[0] = 0.0
        if D > 1:
            start_rel[1:] = np.cumsum(dv)[:-1]

        with np.errstate(invalid="ignore"):
            rel = t[:, None] - start_rel[None, :]   # (T, D); NaN where start_rel is NaN
            observed = rel >= 0                     # (NaN >= 0) -> False
            capped = np.minimum(rel, dv[None, :])   # np.minimum propagates NaN
            # NaN-duration columns are uncapped (Python min(rel, nan) == rel).
            progress = np.where(np.isnan(dv)[None, :], rel, capped)
            # Zero out un-started cells; this also clears any residual NaN.
            progress = np.where(observed, progress, 0.0)

        block = X[pos:pos + T]
        block[:, 0::2] = observed                   # bool -> int (Binary)
        block[:, 1::2] = np.rint(progress)          # whole values -> int (Duration)
        y[pos:pos + T] = class_map.get(int(ent[r]), 0)
        pos += T

    return X, y, feature_names


def _build_entity_sti_index(target_stis):
    """
    Preprocess the forecast target variable's STIs into a per-entity lookup of
    sorted (starts, ends, states) arrays for fast interval-cover queries.

    :param target_stis: DataFrame with columns EntityID, StartTime, EndTime, StateID,
                        already filtered to the single forecast target variable.
    :return: dict EntityID(int) -> (starts, ends, states) int64 arrays, sorted by start.
    """
    index = {}
    if target_stis is None or target_stis.empty:
        return index
    for entity_id, grp in target_stis.groupby("EntityID"):
        g = grp.sort_values("StartTime")
        index[int(entity_id)] = (
            g["StartTime"].to_numpy(dtype=np.int64),
            g["EndTime"].to_numpy(dtype=np.int64),
            g["StateID"].to_numpy(dtype=np.int64),
        )
    return index


def _lookup_states(entity_index, taus):
    """
    For each query time in ``taus``, return the StateID of the target-variable STI
    covering it (start <= tau <= end), plus a boolean mask of which taus were covered.

    Assumes the entity's STIs are non-overlapping and sorted by start (true for a
    single abstracted variable), so the covering STI is the last one whose
    start <= tau. NaN taus (e.g. a NaN TFS) resolve to "not covered" and are dropped.

    :param entity_index: (starts, ends, states) sorted arrays for one entity, or None.
    :param taus: 1D array of absolute query times (t + HORIZON); may contain NaN.
    :return: (states, covered) each length len(taus); states is int64 (garbage where ~covered).
    """
    n = len(taus)
    if entity_index is None:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    starts, ends, states = entity_index
    # Last interval whose start <= tau; searchsorted puts NaN at the end (idx = last).
    idx = np.searchsorted(starts, taus, side="right") - 1
    safe_idx = np.clip(idx, 0, len(starts) - 1)
    # in-range only if a start <= tau existed AND tau <= that interval's end.
    # (NaN tau makes tau <= end False, so it is excluded.)
    covered = (idx >= 0) & (taus <= ends[safe_idx])
    return states[safe_idx], covered


def build_forecast_training_arrays(durations_df, target_stis, horizon):
    """
    MARIO forecast feature matrix, built as compact arrays. Each row is one
    (entity, absolute timestamp t) drawn from the evolving TIRP-prefix durations,
    and the label ``y`` is the abstracted state of the forecast target variable at
    absolute time ``t + horizon``.

    This is the forecasting counterpart of ``build_cpml_training_arrays``: the X
    features are produced by the exact same compact fill (they depend only on
    ``current_time - start_time``, so the absolute TFS cancels and is bit-identical).
    The two forecasting-specific changes are:
      (a) the absolute timestamp = ``TFS + t`` is reconstructed per row so the future
          label at ``t + horizon`` can be looked up, and
      (b) rows whose ``t + horizon`` is not covered by any target STI (a gap or the end
          of the entity's data) are dropped from BOTH X and y -- matching the global
          label definition ("if no STI covers t+HORIZON, that (entity, t) is dropped").

    Unlike the FCPM path, ``y`` varies per timestamp (a categorical target symbol),
    not per entity, so the downstream model is multiclass rather than binary.

    :param durations_df: DataFrame ["EntityID", <duration cols...>, "TFS"], one row per instance.
    :param target_stis: DataFrame [EntityID, StartTime, EndTime, StateID] of the forecast
                        target variable only (from the Train KL when training).
    :param horizon: forecast lead time in TimeStamp units.
    :return: (X, y, feature_names)
             X            : np.ndarray (n_kept, 2*D) integer feature matrix
             y            : np.ndarray (n_kept,) int32 target StateID at t + horizon
             feature_names: list[str] length 2*D giving the column order of X
                            (matches the names create_test_feature_matrix emits).
    """
    entity_col = "EntityID"
    time_cols = [c for c in durations_df.columns if c not in (entity_col, "TFS", "event_time")]
    D = len(time_cols)

    feature_names = []
    for c in time_cols:
        feature_names.append(f"{c}_Binary")
        feature_names.append(f"{c}_Duration")

    if durations_df.empty or D == 0:
        return (np.empty((0, 2 * D), dtype=np.int16),
                np.empty((0,), dtype=np.int32),
                feature_names)

    # Duration cells as float (may contain NaN); round to guard against "62.0" text
    # and binary float artifacts. Matches build_cpml_training_arrays.
    dur = np.round(durations_df[time_cols].to_numpy(dtype=np.float64))
    ent = durations_df[entity_col].to_numpy()
    tfs = durations_df["TFS"].to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore"):
        spans_int = np.nansum(dur, axis=1).astype(np.int64)
    n = int(spans_int.sum() + len(durations_df))  # +1 row per instance (t = 0..span)

    max_val = int(spans_int.max()) if len(spans_int) else 0
    idtype = np.int16 if max_val <= np.iinfo(np.int16).max else np.int32

    # Preallocate the full (pre-drop) matrix; the keep-mask compacts it at the end.
    X = np.empty((n, 2 * D), dtype=idtype)
    y = np.empty((n,), dtype=np.int32)
    keep = np.zeros((n,), dtype=bool)

    sti_index = _build_entity_sti_index(target_stis)

    pos = 0
    for r in range(len(durations_df)):
        dv = dur[r]                       # (D,) float, may contain NaN
        span = int(spans_int[r])
        T = span + 1
        t = np.arange(T)                  # relative timestamps 0..span

        # Relative interval start times; cumsum propagates NaN exactly like the original.
        start_rel = np.empty(D, dtype=np.float64)
        start_rel[0] = 0.0
        if D > 1:
            start_rel[1:] = np.cumsum(dv)[:-1]

        with np.errstate(invalid="ignore"):
            rel = t[:, None] - start_rel[None, :]   # (T, D); NaN where start_rel is NaN
            observed = rel >= 0                     # (NaN >= 0) -> False
            capped = np.minimum(rel, dv[None, :])   # np.minimum propagates NaN
            progress = np.where(np.isnan(dv)[None, :], rel, capped)
            progress = np.where(observed, progress, 0.0)

        block = X[pos:pos + T]
        block[:, 0::2] = observed                   # bool -> int (Binary)
        block[:, 1::2] = np.rint(progress)          # whole values -> int (Duration)

        # Forecast label: state of the target variable at absolute time t + horizon.
        taus = tfs[r] + t + horizon                 # absolute future query times
        states, covered = _lookup_states(sti_index.get(int(ent[r])), taus)
        y[pos:pos + T] = states
        keep[pos:pos + T] = covered
        pos += T

    return X[keep], y[keep], feature_names


def build_forecast_inference_matrix(durations_df, target_stis, horizon):
    """
    MARIO forecast feature matrix for INFERENCE on the test set (Stage 4).

    Uses the exact same compact per-(entity, absolute t) feature fill as
    ``build_forecast_training_arrays`` (the X features are bit-identical -- they depend
    only on ``current_time - start_time``, so the absolute TFS cancels), but it is built
    for scoring a saved model on unseen data rather than training:

      * **No rows are dropped.** Every ``(entity, t)`` drawn from the evolving
        TIRP-prefix durations gets a feature row and therefore a forecast, even where
        ``t + horizon`` has no covering target STI (a gap / the end of the entity's
        data). Training drops those rows because they have no label; at inference we
        still want the prediction and just mark it as having no ground truth.
      * **Per-row metadata is returned alongside X** so Stage 4/5 can align each
        forecast to its ``(EntityID, absolute t)``, evaluate against ground truth on
        the covered rows, and later filter by test regime / ``cut_time`` and aggregate
        across TIRPs.

    :param durations_df: DataFrame ["EntityID", <duration cols...>, "TFS"], one row per instance.
    :param target_stis: DataFrame [EntityID, StartTime, EndTime, StateID] of the forecast
                        target variable only. For inference these come from the **Test** KL,
                        supplying the ground-truth symbol at ``t + horizon``.
    :param horizon: forecast lead time in TimeStamp units.
    :return: (X, meta, feature_names)
             X            : np.ndarray (n_rows, 2*D) integer feature matrix (no rows dropped)
             meta         : DataFrame aligned row-for-row with X, columns
                            [EntityID, current_time, TFS, y_true, covered] where
                            ``current_time`` = absolute t, ``y_true`` = target StateID at
                            ``t + horizon`` (meaningless where ``covered`` is False), and
                            ``covered`` flags whether a target STI covered ``t + horizon``.
             feature_names: list[str] length 2*D giving the column order of X
                            (same names ``build_forecast_training_arrays`` emits).
    """
    entity_col = "EntityID"
    time_cols = [c for c in durations_df.columns if c not in (entity_col, "TFS", "event_time")]
    D = len(time_cols)

    feature_names = []
    for c in time_cols:
        feature_names.append(f"{c}_Binary")
        feature_names.append(f"{c}_Duration")

    empty_meta = pd.DataFrame(
        {"EntityID": [], "current_time": [], "TFS": [], "y_true": [], "covered": []}
    )
    if durations_df.empty or D == 0:
        return (np.empty((0, 2 * D), dtype=np.int16), empty_meta, feature_names)

    dur = np.round(durations_df[time_cols].to_numpy(dtype=np.float64))
    ent = durations_df[entity_col].to_numpy()
    tfs = durations_df["TFS"].to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore"):
        spans_int = np.nansum(dur, axis=1).astype(np.int64)
    n = int(spans_int.sum() + len(durations_df))  # +1 row per instance (t = 0..span)

    max_val = int(spans_int.max()) if len(spans_int) else 0
    idtype = np.int16 if max_val <= np.iinfo(np.int16).max else np.int32

    X = np.empty((n, 2 * D), dtype=idtype)
    # Row-aligned metadata (no drop): entity, absolute current time, TFS, label, coverage.
    ent_out = np.empty((n,), dtype=np.int64)
    curtime_out = np.empty((n,), dtype=np.int64)
    tfs_out = np.empty((n,), dtype=np.int64)
    y_out = np.empty((n,), dtype=np.int64)
    covered_out = np.zeros((n,), dtype=bool)

    sti_index = _build_entity_sti_index(target_stis)

    pos = 0
    for r in range(len(durations_df)):
        dv = dur[r]                       # (D,) float, may contain NaN
        span = int(spans_int[r])
        T = span + 1
        t = np.arange(T)                  # relative timestamps 0..span

        start_rel = np.empty(D, dtype=np.float64)
        start_rel[0] = 0.0
        if D > 1:
            start_rel[1:] = np.cumsum(dv)[:-1]

        with np.errstate(invalid="ignore"):
            rel = t[:, None] - start_rel[None, :]   # (T, D); NaN where start_rel is NaN
            observed = rel >= 0                     # (NaN >= 0) -> False
            capped = np.minimum(rel, dv[None, :])   # np.minimum propagates NaN
            progress = np.where(np.isnan(dv)[None, :], rel, capped)
            progress = np.where(observed, progress, 0.0)

        block = X[pos:pos + T]
        block[:, 0::2] = observed                   # bool -> int (Binary)
        block[:, 1::2] = np.rint(progress)          # whole values -> int (Duration)

        # Absolute current time per row and the future label at t + horizon.
        abstime = tfs[r] + t                        # TFS is a real instance time (never NaN)
        taus = abstime + horizon                    # absolute future query times
        states, covered = _lookup_states(sti_index.get(int(ent[r])), taus)

        ent_out[pos:pos + T] = int(ent[r])
        curtime_out[pos:pos + T] = np.rint(abstime).astype(np.int64)
        tfs_out[pos:pos + T] = int(round(float(tfs[r])))
        y_out[pos:pos + T] = states
        covered_out[pos:pos + T] = covered
        pos += T

    meta = pd.DataFrame({
        "EntityID": ent_out,
        "current_time": curtime_out,
        "TFS": tfs_out,
        "y_true": y_out,
        "covered": covered_out,
    })
    return X, meta, feature_names


def build_fcp_model_tables(tirp, max_gap, event_symbol, epsilon=0, num_relations=7, tirp_detector=None, class1=False):
    """
    High-level wrapper for the FCPM step:
      1) Generate TIRP-prefixes from the input TIRP.
      2) For each prefix, generate all possible complete TIRPs due to multiple unfinished STIs.
      3) Run a single detection on the union of all TIRPs.
      4) For each prefix, concatenate the tiep durations from each TIRP's detection result,
         identified by their to_string().
      5) Remove any columns from the merged DataFrame that contain missing values.

    :param tirp: TIRP object with ._symbols, ._tirp_matrix, etc.
    :param epsilon: Numeric value used in generate_all_complete_tirps.
    :param num_relations: Typically 3 or 7, for the TIRP relations domain.
    :param tirp_detector: Object with a run_detection(tirps_list) method that returns a dictionary
                          mapping TIRP objects to detection DataFrames.
    :return: A dict mapping each prefix -> final durations DataFrame (with columns containing missing values removed).
    """

    tiep_order = tirp.compute_tiep_order(num_relations)
        # Create a flattened list of all TIEPs in their correct processing order.
    flattened_tieps = []
    for group in tiep_order:
        for tiep_str_in_group in group:
            flattened_tieps.append(tiep_str_in_group)
    col_names = [f"({flattened_tieps[i]}, {flattened_tieps[i+1]})" 
                     for i in range(len(flattened_tieps) - 1)]


    # 1) Reveal the TIRP-prefixes.
    prefix_list = tirp_prefixes_revealer(tirp, num_relations=num_relations)

    # Keep track of the TIRPs that belong to each prefix.
    prefix_tirps_dict = {}

    # 2) For each prefix, build the set of TIRPs.
    all_tirps_for_detection = []
    for pref in prefix_list:
        # print(f"Processing prefix: {pref.to_string()} , unfinished: {pref.unfinished_symbols}, finished: {pref.finished_symbols}")
        candidate_tirps_set = generate_all_complete_tirps(pref, epsilon=epsilon, num_relations=num_relations)
        candidate_tirps = list(candidate_tirps_set)  # Convert set to list.
        prefix_tirps_dict[pref] = candidate_tirps
        all_tirps_for_detection.extend(candidate_tirps)

    # Remove duplicate TIRPs if any.
    all_tirps_for_detection = list(set(all_tirps_for_detection))

    if tirp_detector is None:
        raise ValueError("tirp_detector must be provided, with a run_detection(tirps_list) method")

    # 3) Run detection once for all TIRPs.
    detection_results = tirp_detector.run_detection(all_tirps_for_detection)

    # Build a dictionary keyed by each TIRP's to_string() representation.
    detection_results_by_string = {}
    for detected_tirp_obj in detection_results:
        key_str = detected_tirp_obj.to_string()
        detection_results_by_string[key_str] = detected_tirp_obj

    prefix_durations_map = {}

    # 4) For each prefix, gather TIEP durations for each TIRP by string matching.
    for i, pref in enumerate(prefix_list):
        num_col = max(1,((1*len(pref.unfinished_symbols)+2*len(pref.finished_symbols))-1))
        # print(f"Processing prefix {i + 1}/{len(prefix_list)}: {pref.to_string()}")
        # print(f'col_names: {col_names[:num_col]}')
        if (i < len(prefix_list) - 1) or class1:
            tirps_for_this_prefix = prefix_tirps_dict[pref]
            df_list = []

            for t in tirps_for_this_prefix:
                t_str = t.to_string()
                # print(f"Processing TIRP: {t_str}")
                if t_str not in detection_results_by_string:
                    continue  # Skip if no detection result for this TIRP.

                # Now compute durations from the detection result.
                single_tirp_durations = extract_tiep_durations(detection_results_by_string[t_str], max_gap=max_gap,
                                                               num_relations=num_relations,col_names=col_names[:num_col])
                if not single_tirp_durations.empty:
                    df_list.append(single_tirp_durations)

            if df_list:
                merged_df = pd.concat(df_list, ignore_index=True)
                # Remove columns that contain any missing values.
                merged_df = merged_df.dropna(axis=1, how='any')
            else:
                merged_df = single_tirp_durations

            prefix_durations_map[pref] = merged_df
        else:
            prev_prefix = prefix_list[-2]
            prev_Tirp = detection_results_by_string[prefix_tirps_dict[prev_prefix][0].to_string()]
            # print(prev_Tirp.to_string())
            prev_prefix_instances = prev_Tirp.instances
            event_instances = tirp_detector.karma.get_one_size_tirp_instances(str(event_symbol))

            # print("=== prev_prefix_instances (head) ===")
            # print(prev_prefix_instances.head())

            # print("=== event_instances (head) ===")
            # print(event_instances.head())

            last_prefix_instances = prev_prefix_instances.merge(event_instances, on=["EntityID"], how='left')
            last_prefix = prefix_tirps_dict[pref][0]
            last_prefix.instances = last_prefix_instances
            single_tirp_durations = extract_tiep_durations(last_prefix, max_gap=max_gap, num_relations=num_relations,col_names=col_names[:num_col])
            prefix_durations_map[pref] = single_tirp_durations

    return prefix_durations_map


def run_build_tables(file_path, max_gap, num_relations, epsilon, tirp_obj, event_symbol, TTE=False,
                     output_folder="output_tables", class1=False):
    """
    Wrapper function that:
      1. Creates a TIRPDetector object using the given file_path, max_gap, num_relations, and epsilon.
      2. Uses the provided TIRP object.
      3. Calls build_fcp_model_tables with the TIRP and detector to obtain a mapping of each TIRP-prefix
         to its corresponding durations DataFrame.
      4. Saves each generated table (DataFrame) as a CSV file in the specified output folder.
      5. If TTE is True, selects the last prefix's table and runs create_feature_matrix_for_TTE on it,
         saving the resulting feature matrix as a CSV file.
      6. Returns the resulting prefix durations mapping.

    :param file_path: Path to the time intervals file (e.g., "MIMIC_equal-frequency_bins-2_ig-40.txt").
    :param max_gap: Maximum gap allowed between intervals.
    :param num_relations: Number of temporal relation modes (e.g., 3 or 7).
    :param epsilon: Epsilon value for temporary relation determination.
    :param tirp_obj: A TIRP object.
    :param TTE: Binary flag; if True, process the last prefix's durations table with create_feature_matrix_for_TTE.
    :param output_folder: Path to the folder where CSV files will be saved.
    :return: A dictionary mapping each TIRP-prefix (keyed by its string representation) to its durations DataFrame.
    """
    # Ensure output folder exists.
    os.makedirs(output_folder, exist_ok=True)

    # Create the TIRPDetector object.
    detector = TIRPDetector(
        time_intervals_path=file_path,
        num_relations=num_relations,
        max_gap=max_gap,
        epsilon=epsilon,
        output_path="",
        print_instances=False,
        one_size_tirp=True
    )

    # Call build_fcp_model_tables to get the mapping.
    prefix_durations_map = build_fcp_model_tables(tirp_obj, tirp_detector=detector, max_gap=max_gap,
                                                  event_symbol=event_symbol,
                                                  num_relations=num_relations, epsilon=epsilon, class1=class1)

    # Save each table as a CSV file.
    for index, item in enumerate(prefix_durations_map.items(), start=1):
        prefix_key, df = item
        # Convert the prefix's string representation, replacing "*" with "~".
        safe_prefix_str = prefix_key.to_string().replace('*', '~')
        csv_path = os.path.join(output_folder, f"{index}_{safe_prefix_str}.csv")
        df.to_csv(csv_path, index=False)

    # If TTE flag is true, take the last prefix's table and run create_feature_matrix_for_TTE.
    if TTE and prefix_durations_map:
        # Here we assume that the dictionary preserves insertion order.
        last_key = list(prefix_durations_map.keys())[-1]
        last_table = prefix_durations_map[last_key]
        feature_matrix = create_feature_matrix_for_TTE(last_table)
        feature_csv_path = os.path.join(output_folder, "TTE_feature_matrix.csv")
        feature_matrix.to_csv(feature_csv_path, index=False)

    return prefix_durations_map


def merge_prefix_tables(dataframes):
    """
    Merge a list of TIRP-prefix DataFrames iteratively.
    The function starts with the first DataFrame and then merges each subsequent DataFrame.
    For each merge, it finds common columns between the current merged DataFrame and the next DataFrame,
    then retains rows in the merged DataFrame that are not present in the next DataFrame (i.e., non-evolved rows),
    and concatenates these with the next DataFrame.

    Args:
        dataframes (list): List of Pandas DataFrames to merge.

    Returns:
        pd.DataFrame: Final merged DataFrame.
    """
    if not dataframes:
        return pd.DataFrame()

    merged_df = dataframes[1]
    for next_df in dataframes[2:]:
        common_columns = list(set(merged_df.columns) & set(next_df.columns))
        if common_columns:
            # Identify rows in merged_df that do not appear in next_df based on common columns.
            non_evolved_rows = merged_df.merge(next_df, on=common_columns, how='left', indicator=True)
            non_evolved_rows = non_evolved_rows[non_evolved_rows['_merge'] == 'left_only']
            non_evolved_rows = non_evolved_rows.drop(columns=['_merge'])
            non_evolved_rows = non_evolved_rows[common_columns]
        else:
            non_evolved_rows = merged_df
        merged_df = pd.concat([next_df, non_evolved_rows], ignore_index=True)

    # # Remove the last column ---
    # if not merged_df.empty and merged_df.shape[1] >= 2:
    #     # Identify the name of the second-to-last column
    #     column_to_drop = merged_df.columns[-2]
    #     # Drop that column by name
    #     merged_df = merged_df.drop(columns=[column_to_drop])
    return merged_df


def add_event_time_to_table(merged_df, tiep_order, event_symbol):
    """
    Given a merged DataFrame (merged_df) produced from merging prefix tables and a tiep_order list
    (as returned by tirp.compute_tiep_order(num_relations)), this function adds an "event_time" column.

    The function works as follows:
      - It searches the tiep_order for the first occurrence of the event's starting tiep (i.e. event_symbol+'+').
      - Let p be its index in the tiep_order list.
      - It assumes that the merged DataFrame contains columns "EntityID", a set of duration columns (ordered),
        and a "TFS" column.
      - For each row, if all of the first p duration columns (from the ordered duration columns) are non-missing,
        then event_time is set to TFS + sum(durations[0:p]). Otherwise, event_time is set to NaN.
      - Finally, the DataFrame is trimmed to include only "EntityID", the first p duration columns, "TFS", and "event_time".

    :param merged_df: The merged DataFrame from all prefix tables.
    :param tiep_order: A list of tuples representing the tiep groups, e.g.:
                       [('A+', 'B+'), ('B-',), ('C+',), ('C-',)...]
    :param event_symbol: The symbol (string) representing the event, e.g., "X".
    :return: A new DataFrame with only the duration columns up until the event and an added "event_time" column.
    """
    # Identify duration columns (assuming "EntityID" and "TFS" are not durations).
    dur_columns = [col for col in merged_df.columns if col not in ["EntityID", "TFS"]]

    # Find the first tiep group (position p) that contains event_symbol with '+'
    event_tiep = f"{event_symbol}+"
    p = None
    for idx, group in enumerate(tiep_order):
        if event_tiep in group:
            p = idx
            break

    if p is None:
        print(f"Event symbol {event_symbol}+ not found in tiep_order. 'event_time' will be NaN for all rows.")
        merged_df["event_time"] = np.nan
        return merged_df

    # Ensure we do not exceed available duration columns.
    p = min(p, len(dur_columns))

    def compute_event_time(row):
        # If there are no durations to sum (p == 0), event_time equals TFS.
        if p == 0:
            return row["TFS"]
        # If any of the first p duration columns are NaN, event_time becomes NaN.
        if row[dur_columns[:p]].isnull().any():
            return np.nan
        # Otherwise, compute event_time as TFS plus the sum of the first p durations.
        return row["TFS"] + row[dur_columns[:p]].sum()

    merged_df["event_time"] = merged_df.apply(compute_event_time, axis=1)

    # Keep only columns "EntityID", first p duration columns, "TFS", and "event_time".
    new_df = merged_df[["EntityID"] + dur_columns[:p] + ["TFS", "event_time"]].copy()
    return new_df


def run_test_table(file_path, max_gap, num_relations, epsilon, tirp_obj, event_symbol,
                   output_folder="fcpm_test_tables"):
    """
    This wrapper function:
      1. Creates a TIRPDetector object using the given parameters.
      2. Calls build_fcp_model_tables to obtain a mapping from each TIRP-prefix to its durations DataFrame.
      3. Saves each prefix table as a CSV file.
      4. Merges all prefix tables into one final DataFrame.
      5. Computes the tiep order from the TIRP object.
      6. Uses add_event_time_to_table to trim the merged table (if desired).
      7. Saves the final merged & trimmed table as a CSV.
      8. Additionally, creates a TIRP candidate of size 1 using the event_symbol,
         runs detection on it, and builds an event_time_dict mapping EntityID to the event’s start time.
      9. Calls create_test_feature_matrix(duration_df, event_time_dict) using the merged table and the event dictionary.
     10. Saves the final test feature matrix as a CSV and returns it.

    :param file_path: Path to the time intervals file.
    :param max_gap: Maximum gap allowed between intervals.
    :param num_relations: Number of temporal relation modes (e.g., 3 or 7).
    :param epsilon: Epsilon value for temporary relation determination.
    :param tirp_obj: A TIRP object.
    :param event_symbol: The symbol (string) representing the event.
    :param output_folder: Folder where CSV files will be saved.
    :return: The final test feature matrix (DataFrame) as produced by create_test_feature_matrix.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)

    # --- Step 1: Compute tiep order from the TIRP object (for later use)
    tiep_order = tirp_obj.compute_tiep_order(num_relations)

    # --- Step 2: Create the TIRPDetector object.
    detector = TIRPDetector(
        time_intervals_path=file_path,
        num_relations=num_relations,
        max_gap=max_gap,
        epsilon=epsilon,
        output_path="",
        print_instances=False,
        one_size_tirp=True
    )

    # --- Step 3: Build mapping from TIRP-prefix to durations DataFrame.
    prefix_durations_map = build_fcp_model_tables(tirp_obj, epsilon=epsilon, max_gap=max_gap, event_symbol=event_symbol,
                                                  class1=False,
                                                  num_relations=num_relations, tirp_detector=detector)

    # --- Step 4: Save each prefix table to CSV and collect them.
    df_list = []
    for index, item in enumerate(prefix_durations_map.items(), start=1):
        prefix_key, df = item
        safe_prefix_str = prefix_key.to_string().replace('*', '~')
        csv_path = os.path.join(output_folder, f"{index}_{safe_prefix_str}.csv")
        df.to_csv(csv_path, index=False)
        df_list.append(df)

    # --- Step 5: Merge all prefix tables into one final DataFrame.
    final_merged_df = merge_prefix_tables(df_list)

    # --- Step 6: (Optional) Trim the merged table.
    # trimmed_df = add_event_time_to_table(final_merged_df, tiep_order, event_symbol)

    # --- Step 8: Run detection on the event candidate to obtain its instances.
    # detector.run_detection expects a list of TIRP objects and returns a dict mapping each candidate TIRP to its detection DataFrame.
    event_symbol = int(event_symbol) + 1
    event_instances = detector.karma.get_one_size_tirp_instances(str(event_symbol))

    # --- Step 9: Build the event_time_dict: map each EntityID to the event's start time.
    # We assume that in event_df, the column with the event symbol contains tuples (start, end).
    event_time_dict = {}
    for idx, row in event_instances.iterrows():
        entity = row["EntityID"]
        # print(row)
        # Only add if the cell is not NaN and is a tuple.
        if pd.notna(row[str(event_symbol)]) and isinstance(row[str(event_symbol)], (tuple, list)):
            event_time_dict[entity] = row[str(event_symbol)][0]

    # --- Step 10: Generate the final test feature matrix using create_test_feature_matrix.
    # (This function is assumed to use the merged durations table and event_time_dict to compute features.)

    final_merged_df_csv_path = os.path.join(output_folder, "durations_merged_df.csv")
    final_merged_df.to_csv(final_merged_df_csv_path, index=False)
    # Save event_time_dict as a pickle file
    event_time_dict_path = os.path.join(output_folder, "event_time_dict.pkl")
    with open(event_time_dict_path, "wb") as f:
        pickle.dump(event_time_dict, f)

    # final_feature_matrix = create_test_feature_matrix(final_merged_df, event_time_dict)
    # feature_csv_path = os.path.join(output_folder, "final_test_feature_matrix.csv")
    # final_feature_matrix.to_csv(feature_csv_path, index=False)
    # return final_feature_matrix
    return 1


def build_forecast_durations(file_path, max_gap, num_relations, epsilon, tirp_obj,
                             output_folder=None):
    """
    Build the per-instance TIEP-duration table for one TIRP, uniformly across all of
    its evolving prefixes (MARIO forecasting: no event symbol, no class split, no TTE).

    This is the event-free counterpart of ``run_test_table`` / ``run_build_tables``: it
    runs the TIRP detector on ``file_path``, reveals the TIRP's prefixes, and extracts
    each prefix's consecutive-TIEP durations via the SAME normal path FCPM used for its
    ``class1`` case (``class1=True``), so the final/full prefix is treated like any other
    -- there is no synthetic-event stitching on the last prefix and ``event_symbol`` is
    never consulted. The per-prefix tables are merged with ``merge_prefix_tables`` into
    one durations table (one row per detected instance; columns are the duration between
    consecutive TIEPs, plus ``TFS`` = absolute time of the first TIEP).

    :param file_path: KL time-intervals file (MARIO Stage 1 ``Train/KL.txt`` when training).
    :param tirp_obj: a TIRP object (pass a ``.copy_tirp()`` -- detection mutates instances).
    :param output_folder: if given, ``durations_merged_df.csv`` is written there.
    :return: durations_merged_df (DataFrame ["EntityID", <dur cols...>, "TFS"]).
    """
    detector = TIRPDetector(
        time_intervals_path=file_path,
        num_relations=num_relations,
        max_gap=max_gap,
        epsilon=epsilon,
        output_path="",
        print_instances=False,
        one_size_tirp=True,
    )

    # class1=True => every prefix (including the full TIRP) uses the normal duration
    # path; event_symbol is unused there, so pass None.
    prefix_durations_map = build_fcp_model_tables(
        tirp_obj, tirp_detector=detector, max_gap=max_gap, event_symbol=None,
        num_relations=num_relations, epsilon=epsilon, class1=True,
    )

    df_list = list(prefix_durations_map.values())
    if len(df_list) >= 2:
        # merge_prefix_tables intentionally drops the first (shortest) prefix table,
        # matching FCPM's durations construction that MARIO reuses unchanged.
        merged_df = merge_prefix_tables(df_list)
    elif len(df_list) == 1:
        merged_df = df_list[0]
    else:
        merged_df = pd.DataFrame(columns=["EntityID", "TFS"])

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        merged_df.to_csv(os.path.join(output_folder, "durations_merged_df.csv"), index=False)

    return merged_df





