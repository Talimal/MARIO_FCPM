# ta_package/utils.py
import pandas as pd
import numpy as np
from .constants import ENTITY_ID, VALUE, TEMPORAL_PROPERTY_ID, TIMESTAMP
from scipy.stats import entropy
import os 
import math
import logging
from tqdm import tqdm

# Setup logging for utils
logger = logging.getLogger(__name__)


# =============================================================================
# TID3 Configuration Mapping
# =============================================================================
# TID3 method naming convention: tid3[_mv][_asc|_random][_seed<N>][_tstat][_<duration_pref>][_mingap<value>][_<nb_candidates>]
#   - "tid3" alone uses defaults (max_t_stat_sum, two_sided, no multivariate, nb_candidates=None)
#   - "mv" flag: enables Phase 2 conditional intra-variable greedy refinement
#     (see Hugobot2/ta_package/methods/README.md for the algorithm).
#   - MV variable-order shorthand (only valid with "mv"; at most one):
#       "asc"    -> mv_variable_order = "univariate_score_asc"
#       "random" -> mv_variable_order = "random"
#     Bare "tid3_mv" keeps the default "univariate_score_desc" ordering.
#   - "seed<N>" token (only valid alongside "random"): sets mv_random_seed = <N>
#     where <N> is a non-negative integer. Required when "random" is specified —
#     callers must declare the seed explicitly so multi-seed ablations have
#     reproducible, distinct directories per seed.
#   - Scoring shorthand: "tstat" -> max_t_stat_sum, "logrank" -> max_logrank_sum
#     (censoring-aware log-rank test on the STI duration survival curves; not
#     combinable with 'mv'). The legacy "ks"/"kl"/"mw"/"ws"/"avgdiff"/"selftrans"
#     shortcuts have been removed because TID3.AVAILABLE_SCORING_METHODS no longer
#     contains those methods — they were silently rejected at construction time.
#     Re-add them here only if/when TID3 re-implements the corresponding scorers.
#   - Duration pref shorthand: c1longer (class1_longer), c0longer (class0_longer)
#   - Optional mingap<value>: min_mean_gap constraint. Uses 'p' as the decimal
#     separator so the token stays filesystem-safe. Extracted by strip_mingap_suffix
#     before parse_tid3_config runs, so it may appear in any position.
#   - Optional trailing integer: number of initial equal-frequency candidate cutpoints
#
# Examples:
#   "tid3"                  -> max_t_stat_sum, two_sided, no multivariate, nb_candidates=None
#   "tid3_tstat"            -> max_t_stat_sum, two_sided, no multivariate, nb_candidates=None
#   "tid3_mv"               -> max_t_stat_sum, two_sided, multivariate (desc), nb_candidates=None
#   "tid3_mv_asc"           -> max_t_stat_sum, two_sided, multivariate (asc),  nb_candidates=None
#   "tid3_mv_random_seed0"  -> max_t_stat_sum, two_sided, multivariate (random, seed=0)
#   "tid3_mv_random_seed7"  -> max_t_stat_sum, two_sided, multivariate (random, seed=7)
#   "tid3_c1longer"         -> max_t_stat_sum, class1_longer, no multivariate, nb_candidates=None
#   "tid3_mv_c0longer"      -> max_t_stat_sum, class0_longer, multivariate (desc), nb_candidates=None
#   "tid3_150"              -> max_t_stat_sum, two_sided, no multivariate, nb_candidates=150
#   "tid3_c0longer_50"      -> max_t_stat_sum, class0_longer, no multivariate, nb_candidates=50
#   "tid3_mv_mingap0p1"     -> tid3_mv (after mingap strip) with min_mean_gap=0.1
#   "tid3_mv_mingap0p5_100" -> tid3_mv with min_mean_gap=0.5, nb_candidates=100
#   "tid3_logrank"          -> max_logrank_sum, two_sided, no multivariate
#   "tid3_logrank_c1longer" -> max_logrank_sum, class1_longer, no multivariate
#   "tid3_logrank_150"      -> max_logrank_sum, two_sided, nb_candidates=150
#   ("tid3_logrank_mv" is invalid: logrank scoring has no MV-phase support)

TID3_SCORING_MAP = {
    "tstat": "max_t_stat_sum",
    "logrank": "max_logrank_sum",
}

TID3_DURATION_PREF_MAP = {
    "twosided": "two_sided",
    "c1longer": "class1_longer",
    "c0longer": "class0_longer",
}


def parse_tid3_config(method_name):
    """
    Parse a TID3 method string into scoring_method, duration_preference, multivariate flag,
    optional nb_candidates, mv_variable_order, and optional mv_random_seed.

    A trailing integer suffix specifies the number of initial equal-frequency candidate cutpoints.
    Strip any `mingap<value>` token with `strip_mingap_suffix` first; this parser does not handle it.

    Only `tstat` -> `max_t_stat_sum` is recognized as a scoring shortcut in this build, because
    `TID3.AVAILABLE_SCORING_METHODS` no longer contains the legacy scorers. Any other shortcut
    raises ValueError here rather than letting `TID3.__init__` fail later.

    A `seed<N>` token (e.g. `seed0`, `seed7`) is only valid alongside `random`; it sets
    mv_random_seed to the non-negative integer N. Specifying `seed<N>` without `random`,
    or specifying it more than once, raises ValueError.

    Examples:
        "tid3" -> ("max_t_stat_sum", "two_sided", False, None, "univariate_score_desc", None)
        "tid3_tstat" -> ("max_t_stat_sum", "two_sided", False, None, "univariate_score_desc", None)
        "tid3_mv" -> ("max_t_stat_sum", "two_sided", True, None, "univariate_score_desc", None)
        "tid3_mv_asc" -> ("max_t_stat_sum", "two_sided", True, None, "univariate_score_asc", None)
        "tid3_mv_random_seed0" -> ("max_t_stat_sum", "two_sided", True, None, "random", 0)
        "tid3_mv_random_seed7" -> ("max_t_stat_sum", "two_sided", True, None, "random", 7)
        "tid3_mv_c1longer" -> ("max_t_stat_sum", "class1_longer", True, None, "univariate_score_desc", None)
        "tid3_150" -> ("max_t_stat_sum", "two_sided", False, 150, "univariate_score_desc", None)
        "tid3_c0longer_50" -> ("max_t_stat_sum", "class0_longer", False, 50, "univariate_score_desc", None)

    Parameters:
        method_name (str): Method string to parse

    Returns:
        tuple: (scoring_method, duration_preference, multivariate_refinement,
                nb_candidates, mv_variable_order, mv_random_seed)
               nb_candidates is None when no numeric suffix is provided.
               mv_variable_order defaults to "univariate_score_desc".
               mv_random_seed is None unless a `seed<N>` token is present.
    """
    # Default values
    scoring_method = "max_t_stat_sum"
    duration_preference = "two_sided"
    multivariate_refinement = False
    nb_candidates = None
    mv_variable_order = "univariate_score_desc"
    mv_random_seed = None

    # Remove "tid3" prefix and split by underscore
    if method_name == "tid3":
        return (scoring_method, duration_preference, multivariate_refinement,
                nb_candidates, mv_variable_order, mv_random_seed)

    parts = method_name.split("_")
    if parts[0] != "tid3":
        raise ValueError(f"Invalid TID3 method name: {method_name}")

    # Check if the last part is a numeric nb_candidates suffix
    if parts[-1].isdigit():
        nb_candidates = int(parts[-1])
        parts = parts[:-1]

    # Track whether a variable-order token was already consumed (mutually exclusive)
    order_token_seen = False
    seed_token_seen = False

    # Parse remaining parts
    for part in parts[1:]:
        if part == "mv":
            multivariate_refinement = True
        elif part == "asc":
            if order_token_seen:
                raise ValueError(
                    f"Multiple mv variable-order tokens in '{method_name}'; "
                    f"use at most one of 'asc' / 'random'."
                )
            mv_variable_order = "univariate_score_asc"
            order_token_seen = True
        elif part == "random":
            if order_token_seen:
                raise ValueError(
                    f"Multiple mv variable-order tokens in '{method_name}'; "
                    f"use at most one of 'asc' / 'random'."
                )
            mv_variable_order = "random"
            order_token_seen = True
        elif part.startswith("seed"):
            if seed_token_seen:
                raise ValueError(
                    f"Multiple 'seed<N>' tokens in '{method_name}'; specify at most one."
                )
            seed_str = part[len("seed"):]
            if not seed_str.isdigit():
                raise ValueError(
                    f"Malformed 'seed' token '{part}' in '{method_name}'. "
                    f"Expected 'seed<N>' with a non-negative integer N."
                )
            mv_random_seed = int(seed_str)
            seed_token_seen = True
        elif part in TID3_SCORING_MAP:
            scoring_method = TID3_SCORING_MAP[part]
        elif part in TID3_DURATION_PREF_MAP:
            duration_preference = TID3_DURATION_PREF_MAP[part]
        else:
            raise ValueError(f"Unknown TID3 config part: '{part}' in method '{method_name}'. "
                           f"Valid scoring: {list(TID3_SCORING_MAP.keys())}, "
                           f"Valid duration_pref: {list(TID3_DURATION_PREF_MAP.keys())}, "
                           f"Valid flags: ['mv', 'asc', 'random', 'seed<N>']")

    if order_token_seen and not multivariate_refinement:
        raise ValueError(
            f"mv variable-order token (asc/random) requires 'mv' flag in '{method_name}'."
        )

    if seed_token_seen and mv_variable_order != "random":
        raise ValueError(
            f"'seed<N>' token in '{method_name}' is only valid alongside 'random' ordering."
        )

    if scoring_method == "max_logrank_sum" and multivariate_refinement:
        raise ValueError(
            f"'logrank' scoring in '{method_name}' does not support Phase-2 MV refinement; "
            f"remove the 'mv' token or use 'tstat' scoring."
        )

    return (scoring_method, duration_preference, multivariate_refinement,
            nb_candidates, mv_variable_order, mv_random_seed)


def strip_mingap_suffix(method_name):
    """Extract a 'mingap<value>' token from a TID3 method name.

    The encoded value uses 'p' as the decimal separator so the token is safe
    to embed in filesystem paths:

        mingap0p1  -> 0.1
        mingap1    -> 1.0
        mingap0p5  -> 0.5

    Returns (method_name_without_mingap_token, min_mean_gap_or_None). The
    returned method name is intended to be fed straight into parse_tid3_config.
    Raises ValueError if the value is malformed or negative, or if more than
    one mingap token appears.
    """
    parts = method_name.split("_")
    kept = []
    min_mean_gap = None
    for part in parts:
        if part.startswith("mingap"):
            payload = part[len("mingap"):].replace("p", ".")
            try:
                value = float(payload)
            except ValueError:
                raise ValueError(
                    f"Could not parse min_mean_gap from token '{part}' in '{method_name}'"
                )
            if value < 0:
                raise ValueError(
                    f"min_mean_gap must be >= 0, got {value} in '{method_name}'"
                )
            if min_mean_gap is not None:
                raise ValueError(
                    f"Multiple 'mingap' tokens in '{method_name}'"
                )
            min_mean_gap = value
        else:
            kept.append(part)
    return "_".join(kept), min_mean_gap


# =============================================================================
# TD4C Configuration Mapping
# =============================================================================
# TD4C method naming convention: td4c[_<distance_measure>]
#   - "td4c" alone uses default (cosine)
#   - Distance measure shorthand: cosine, kl (kullback_leibler), entropy
#
# Examples:
#   "td4c"        -> cosine
#   "td4c_cosine" -> cosine
#   "td4c_kl"     -> kullback_leibler
#   "td4c_entropy" -> entropy

TD4C_DISTANCE_MEASURE_MAP = {
    "cosine": "cosine",
    "kl": "kullback_leibler",
    "entropy": "entropy",
}


def parse_td4c_config(method_name):
    """
    Parse a TD4C method string into distance_measure.
    
    Examples:
        "td4c" -> "cosine"
        "td4c_cosine" -> "cosine"
        "td4c_kl" -> "kullback_leibler"
        "td4c_entropy" -> "entropy"
    
    Parameters:
        method_name (str): Method string to parse
    
    Returns:
        str: distance_measure name
    """
    # Default value
    distance_measure = "cosine"
    
    # Remove "td4c" prefix and split by underscore
    if method_name == "td4c":
        return distance_measure
    
    parts = method_name.split("_")
    if parts[0] != "td4c":
        raise ValueError(f"Invalid TD4C method name: {method_name}")
    
    # Parse remaining parts
    for part in parts[1:]:
        if part in TD4C_DISTANCE_MEASURE_MAP:
            distance_measure = TD4C_DISTANCE_MEASURE_MAP[part]
        else:
            raise ValueError(f"Unknown TD4C config part: '{part}' in method '{method_name}'. "
                           f"Valid distance measures: {list(TD4C_DISTANCE_MEASURE_MAP.keys())}")
    
    return distance_measure

def assign_state(value, boundaries):
    """
    Given a value and a sorted list of boundaries, assign a state id (starting at 1).
    
    For example, with 3 bins (boundaries = [b1, b2]):
      if value < b1: state = 1
      if b1 <= value < b2: state = 2
      if value >= b2: state = 3
    """
    for i, b in enumerate(boundaries):
        if value < b:
            return i + 1
    return len(boundaries) + 1

def generate_candidate_cutpoints(df, nb_candidates):
    """
    Generate candidate cutpoints from the DataFrame's TemporalPropertyValue column.
    
    Parameters:
      df: A DataFrame that contains a column "TemporalPropertyValue".
      nb_candidates: Desired number of candidate cutpoints.
      
    Returns:
      A sorted list of candidate cutpoints.
    """
    values = df["TemporalPropertyValue"].dropna().unique()
    values = np.sort(values)
    # If there are fewer than 2 unique values, return an empty list.
    if len(values) < 2:
        return []
    # Evenly space candidate indices between 1 and len(values)-1:
    indices = np.linspace(1, len(values) - 1, num=nb_candidates, dtype=int)
    candidates = values[indices]
    candidates = np.unique(candidates)
    return candidates.tolist()

def candidate_selection(df, nb_bins, scoring_function, nb_candidates=100):
    """
    Choose cutpoints from a pool of candidate cutpoints based on a scoring function.
    
    The candidate cutpoints are generated using generate_candidate_cutpoints().
    Then, iteratively, one candidate is chosen at a time to maximize the score.
    
    Parameters:
      df: A DataFrame with a column "TemporalPropertyValue".
      nb_bins: The final desired number of bins. (We choose nb_bins-1 cutpoints.)
      scoring_function: A function taking (df, cutoffs) and returning a numeric score.
      nb_candidates: Number of candidate cutpoints to generate initially.
      
    Returns:
      A tuple (chosen_cutpoints, chosen_scores) where:
       - chosen_cutpoints is the list of selected cutpoints (sorted), and
       - chosen_scores is a list of the corresponding scores.
    """
    # Generate candidate cutpoints.
    candidate_pool = generate_candidate_cutpoints(df, nb_candidates)
    chosen_cutpoints = np.array([], dtype=float)
    chosen_scores = np.array([], dtype=float)
    
    # Show progress only in debug mode
    show_progress = logger.isEnabledFor(logging.DEBUG)
    
    for iteration in range(1, nb_bins):
        scores = np.full(len(candidate_pool), -np.inf)
        
        # Progress bar for evaluating candidates in this iteration
        pbar = tqdm(total=len(candidate_pool), 
                   desc=f"    Evaluating candidates (bin {iteration}/{nb_bins-1})",
                   disable=not show_progress,
                   leave=False)
        
        for i, candidate in enumerate(candidate_pool):
            # Skip candidate if it is already (or nearly) in the chosen_cutpoints.
            if len(chosen_cutpoints) > 0 and np.any(np.isclose(candidate, chosen_cutpoints)):
                pbar.update(1)
                continue
            
            # Create a list of suggested cutpoints.
            suggested = np.sort(np.append(chosen_cutpoints, candidate))
            bins_edges = [-np.inf] + suggested.tolist() + [np.inf]
            df_temp = df.copy()
            # Use pd.cut with duplicates dropped.
            try:
                df_temp = df_temp.assign(
                    Bin=pd.cut(df_temp["TemporalPropertyValue"], bins=bins_edges, labels=False, duplicates="drop")
                )
            except Exception as e:
                scores[i] = -np.inf
                pbar.update(1)
                continue
            
            scores[i] = scoring_function(df_temp, suggested.tolist())
            pbar.update(1)
        
        pbar.close()
        
        # If no valid candidate remains, break early.
        if len(scores) == 0:
            logger.warning(f"Early termination at iteration {iteration}: No candidate scores available")
            break
        elif np.all(np.isneginf(scores)):
            logger.warning(f"Early termination at iteration {iteration}: All {len(scores)} candidates produced invalid scores")
            break
        
        best_idx = np.argmax(scores)
        best_candidate = candidate_pool[best_idx]
        best_score = scores[best_idx]
        chosen_cutpoints = np.append(chosen_cutpoints, best_candidate)
        chosen_scores = np.append(chosen_scores, best_score)
        
        # Remove this candidate from the pool.
        candidate_pool.pop(best_idx)
        # Also, remove any candidate that is nearly equal to a chosen candidate.
        candidate_pool = [c for c in candidate_pool if not np.any(np.isclose(c, chosen_cutpoints))]
        
        chosen_cutpoints = np.sort(chosen_cutpoints)
        chosen_scores = chosen_scores[np.argsort(chosen_cutpoints)]
        
        # Check if we're running out of candidates
        if len(candidate_pool) == 0 and iteration < nb_bins - 1:
            logger.warning(f"Candidate pool exhausted at iteration {iteration}: achieved {len(chosen_cutpoints) + 1}/{nb_bins} bins")
    
    return list(chosen_cutpoints), list(chosen_scores)

def symmetric_kullback_leibler(p, q):
    if sum(p) == 0 or sum(q) == 0:
        return 0
    return 0.5 * (entropy(p, q) + entropy(q, p))

def paa_transform(
    data: pd.DataFrame,
    window_size: int = 3,
    agg_method: str = 'mean',
    timestamp_strategy: str = 'bin_left_normalized'
) -> pd.DataFrame:
    """
    Apply Piecewise Aggregate Approximation (PAA) to the time series.

    The input DataFrame should have columns:
      ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE.

    The function groups data by ENTITY_ID and TEMPORAL_PROPERTY_ID, partitions each group
    into non-overlapping windows of size `window_size`, and computes aggregated values.

    Parameters:
      data: pd.DataFrame
      window_size: int, window length.
      agg_method: str, one of 'mean', 'min', 'max'.
      timestamp_strategy: str, one of 'first', 'bin_left_normalized'.

    Returns:
      A new DataFrame with the aggregated time series.
    """
    agg_funcs = {'mean': np.mean, 'min': np.min, 'max': np.max}
    timestamp_strategies = ['first', 'bin_left_normalized']

    # Check parameters
    if agg_method not in agg_funcs:
        raise ValueError("agg_method must be 'mean', 'min', or 'max'.")
    if timestamp_strategy not in timestamp_strategies:
        raise ValueError(f"timestamp_strategy must be one of {timestamp_strategies}.")
    if window_size < 1:
        raise Exception('ERROR: Invalid window size parameter')
    if window_size == 1 or data.empty:
        return data

    def paa_group(group):
        group = group.sort_values(by=TIMESTAMP)
        n = len(group)
        rows = []
        for start in range(0, n, window_size):
            window = group.iloc[start:start + window_size]
            if window.empty:
                continue
            aggregated_value = agg_funcs[agg_method](window[VALUE])

            if timestamp_strategy == 'first':
                aggregated_time = window[TIMESTAMP].iloc[0]

            elif timestamp_strategy == 'bin_left_normalized':
                # Determine the bin for the window based on its first timestamp
                first_timestamp = window[TIMESTAMP].iloc[0]
                # Find the bin left edge
                bin_left = (first_timestamp // window_size) * window_size
                # Normalize by window size and shift to start from 1
                aggregated_time = (bin_left / window_size) + 1

            row = {
                ENTITY_ID: window[ENTITY_ID].iloc[0],
                TEMPORAL_PROPERTY_ID: window[TEMPORAL_PROPERTY_ID].iloc[0],
                TIMESTAMP: aggregated_time,
                VALUE: aggregated_value
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    df = data.groupby([ENTITY_ID, TEMPORAL_PROPERTY_ID], group_keys=False).apply(paa_group)
    df.loc[df[TEMPORAL_PROPERTY_ID] == -1, TIMESTAMP] = 0.0

    return df



def save_symbolic_series(symbolic_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the symbolic time series to a CSV file.
    Expected columns: EntityID, TemporalPropertyID, Timestamp, state.
    """
    symbolic_df.to_csv(output_path, index=False)

def save_states(states, output_path: str) -> None:
    """
    Save the computed states (cutoffs) to a CSV file.
    """
    pd.DataFrame([states]).to_csv(output_path, index=False)


def generate_KL_content(symbolic_series: pd.DataFrame, max_gap: int) -> str:
    """
    Generate the content for the KL file.
    
    The KL file format:
      - First line: "startToncepts"
      - Second line: "numberOfEntities,<number>"
      - Then, for each entity:
          ENTITY_ID;
          start_time,end_time,StateID,TemporalPropertyID;start_time,end_time,StateID,TemporalPropertyID;...
    
    Intervals for a given entity and property are merged if consecutive points share the same state and
    the gap between the current point and the previous point is less than or equal to max_gap.
    
    Parameters:
      symbolic_series (pd.DataFrame): DataFrame with columns: ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, state.
      max_gap (int): The maximum gap threshold for merging intervals.
      
    Returns:
      A string representing the contents of the KL file.
    """
    # Ensure the DataFrame is sorted.
    df = symbolic_series.sort_values(by=[ENTITY_ID, TIMESTAMP, TEMPORAL_PROPERTY_ID])
    kl_lines = []
    
    entities = df[ENTITY_ID].unique()
    kl_lines.append("startToncepts")
    kl_lines.append(f"numberOfEntities,{len(entities)}")
    
    # Process each entity individually.
    for entity in entities:
        entity_df = df[df[ENTITY_ID] == entity].sort_values(by=[TIMESTAMP, TEMPORAL_PROPERTY_ID])
        intervals = []
        # Group by TEMPORAL_PROPERTY_ID, then merge consecutive points into intervals.
        for tpid, group in entity_df.groupby(TEMPORAL_PROPERTY_ID):
            group = group.sort_values(by=TIMESTAMP)
            current_interval = None
            for _, row in group.iterrows():
                ts = row[TIMESTAMP]
                state = row["StateID"]
                if current_interval is None:
                    current_interval = {"start": ts, "end": ts+1, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
                else:
                    # If the same state and property, and the gap is within max_gap, extend the interval.
                    if state == current_interval["StateID"] and (ts - current_interval["end"]) <= max_gap:
                        current_interval["end"] = ts+1
                    else:
                        intervals.append(current_interval)
                        current_interval = {"start": ts, "end": ts+1, "StateID": state, TEMPORAL_PROPERTY_ID: tpid}
            if current_interval is not None:
                intervals.append(current_interval)
        # Sort intervals by start time and then by TEMPORAL_PROPERTY_ID.
        intervals = sorted(intervals, key=lambda x: (x["start"], x["end"], x["StateID"], x[TEMPORAL_PROPERTY_ID]))
                # check if there any nan in interval stat and state id or temporal property id and if there is print them
        for interval in intervals:
            if pd.isna(interval["StateID"]):
                interval["StateID"] = -1  # Replace NaN with -1
        # Format intervals as "start_time,end_time,state,TEMPORAL_PROPERTY_ID"
        interval_strs = [f"{int(interval['start'])},{int(interval['end'])},{int(interval['StateID'])},{int(interval[TEMPORAL_PROPERTY_ID])}" 
                         for interval in intervals]

        # Build the line for the entity.
        entity_line = f"{entity};\n" + ";".join(interval_strs) + ";"
        kl_lines.append(entity_line)
    
    return "\n".join(kl_lines)

def split_train_test(data: pd.DataFrame, train_ratio: float = 0.7):
        unique_ids = data[ENTITY_ID].unique()
        unique_ids = sorted(unique_ids)
        cutoff = int(len(unique_ids) * train_ratio)
        train_ids = unique_ids[:cutoff]
        test_ids = unique_ids[cutoff:]
        train = data[data[ENTITY_ID].isin(train_ids)]
        test = data[data[ENTITY_ID].isin(test_ids)]
        return train, test

def remove_na(data_to_use):
    na_per_column = data_to_use.isna().sum()          # Series: one entry per column
    total_na       = int(na_per_column.sum())        

    if total_na > 0:
        # show only the columns that actually have missing values
        logger.warning(f"Removing {total_na} rows containing NaNs in {', '.join(na_per_column[na_per_column > 0].index)}")

    # Now drop rows that have a NaN in any of the critical columns
    data_to_use = data_to_use.dropna(
        subset=[ENTITY_ID, TEMPORAL_PROPERTY_ID, VALUE]
    )
    return data_to_use
    
def save_entity_ids(entity_relations: pd.DataFrame, output_path: str) -> None:
    
    df = pd.DataFrame(entity_relations.items(), columns=['EntityID', 'ClassID'])
    # make entity id and class id integers
    df['EntityID'] = df['EntityID'].astype(int)
    df['ClassID'] = df['ClassID'].astype(int)
    save_path = output_path + "/entity-class-relations.csv"
    df.to_csv(save_path, index=False)

def map_states_to_test_composite(test_df, states_list, method_config, output_dir , max_gap):
    """
    Maps state IDs to dataframe rows based on temporal property values and method configurations.
    
    Parameters:
    df: DataFrame with columns ['EntityID', 'TemporalPropertyID', 'TimeStamp', 'TemporalPropertyValue']
    states_list: List of state dictionaries with StateID, TemporalPropertyID, MethodName, BinLow, BinHigh
    method_config: Configuration dictionary defining methods for each property
    
    Returns:
    DataFrame with additional StateID column and expanded rows for each method
    """
    
    # Convert states list to DataFrame for easier manipulation
    states_df = pd.DataFrame(states_list)
    entity_class = {}

    class_rows = test_df[test_df[TEMPORAL_PROPERTY_ID] == -1].copy()
    if not class_rows.empty:
        for _, row in class_rows.iterrows():
            ent = row[ENTITY_ID]
            entity_class[ent] = int(float(row[VALUE]))
    test_df = test_df[test_df[TEMPORAL_PROPERTY_ID] != -1]
    test_df = remove_na(test_df)  # drop na

    # Get methods from config
    methods = method_config["default"]
    
    result_rows = []
    
    for _, row in test_df.iterrows():
        entity_id = row['EntityID']
        temporal_property_id = row['TemporalPropertyID']
        timestamp = row['TimeStamp']
        value = row['TemporalPropertyValue']
        
        # For each method, find the appropriate state
        for method in methods:
            method_name = method['method']
            
            # Filter states for this temporal property and method
            relevant_states = states_df[
                (states_df['TemporalPropertyID'] == temporal_property_id) & 
                (states_df['MethodName'] == method_name)
            ]
            
            # Find which bin this value falls into
            state_id = None
            for _, state in relevant_states.iterrows():
                bin_low = state['BinLow']
                bin_high = state['BinHigh']
                
                # Handle infinity values
                if bin_low == -np.inf:
                    bin_low = float('-inf')
                if bin_high == np.inf:
                    bin_high = float('inf')
                
                # Handle NaN values   - or maybe need to put it specific symbol
                if pd.isna(bin_low):
                    bin_low = float('-inf')
                if pd.isna(bin_high):
                    bin_high = float('inf')
                
                # Check if value falls in this bin
                if bin_low <= value < bin_high:
                    state_id = state['StateID']
                    break
                # Special case for the highest bin (inclusive of upper bound)
                elif value == bin_high and bin_high != float('inf'):
                    state_id = state['StateID']
                    break
                else:
                    # If no state found, assign a default state ID (e.g., 0 or -1)
                    state_id = -1
            
            # Add row to results
            result_rows.append({
                'EntityID': entity_id,
                'TemporalPropertyID': temporal_property_id,
                'TimeStamp': timestamp,
                'TemporalPropertyValue': value,
                'StateID': state_id,
                'MethodName': method_name
            })
        # convert result_rows to DataFrame
    df = pd.DataFrame(result_rows)
    # drop col entitiy class
    # make timestamp entity id and temporal property id integers
    df[ENTITY_ID] = df[ENTITY_ID].astype(int)
    df[TEMPORAL_PROPERTY_ID] = df[TEMPORAL_PROPERTY_ID].astype(int)
    df[TIMESTAMP] = df[TIMESTAMP].astype(float)
    
    save_results(entity_class, output_dir, df, states_df, max_gap)

    df = df.drop(columns=['EntityClass'])
    return df

def save_results(entity_class, output_dir: str, symbolic_series: pd.DataFrame, states_df, max_gap: int):
        os.makedirs(output_dir, exist_ok=True)
        
        # Define a helper function for sorting keys:
        def sort_key(x):
            if isinstance(x, int):
                return (0, x)
            else:
                return (1, str(x))
        

        states_file = os.path.join(output_dir, "states.csv")
        states_df.to_csv(states_file, index=False)
        
        symbolic_file = os.path.join(output_dir, "symbolic_time_series.csv")
        symbolic_series.to_csv(symbolic_file, index=False)
        
        kl_content = generate_KL_content(symbolic_series, max_gap)
        kl_file = os.path.join(output_dir, "KL.txt")
        with open(kl_file, "w") as f:
            f.write(kl_content)
        
        if entity_class:
            symbolic_series["EntityClass"] = symbolic_series[ENTITY_ID].map(entity_class)
            for cls in sorted(set(entity_class.values())):
                subset = symbolic_series[symbolic_series["EntityClass"] == cls]
                kl_content_cls = generate_KL_content(subset, max_gap)
                kl_file_cls = os.path.join(output_dir, f"KL-class-{float(cls)}.txt")
                with open(kl_file_cls, "w") as f:
                    f.write(kl_content_cls)

        save_entity_ids(entity_class, output_dir)

        logger.info(f"Results saved in directory: {output_dir}")