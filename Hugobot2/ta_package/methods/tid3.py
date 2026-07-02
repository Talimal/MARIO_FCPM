# File: ta_package/methods/tid3.py
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, logrank, CensoredData
import logging
import warnings
from tqdm import tqdm
import os
from typing import Optional

# Suppress scipy warnings for nearly identical data (expected in constant variables)
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation due to catastrophic cancellation')
# warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
try:
    # Try relative imports first (when used as a module)
    from .base import TAMethod
    from ..utils import assign_state, candidate_selection, generate_candidate_cutpoints
    from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE
except ImportError:
    # Fallback to absolute imports (when running as a script)
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    
    from ta_package.methods.base import TAMethod
    from ta_package.utils import assign_state, candidate_selection, generate_candidate_cutpoints
    from ta_package.constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TID3(TAMethod):
    """
    TID3 (Time Interval Duration Driven Discretization) - A supervised
    discretization method that selects cutoffs by maximizing the divergence
    of symbolic time interval (STI) time-duration distributions between
    case and control populations.

    Scoring is based on the sum of per-state Welch's t-statistics computed
    over time-duration distributions (scoring_method="max_t_stat_sum"), or on
    the sum of per-state censoring-aware log-rank statistics over the duration
    survival curves (scoring_method="max_logrank_sum"; gap-truncated and
    record-end STIs are right-censored). Both support an optional one-sided
    directional preference (class 0 longer / class 1 longer). The t-stat scorer
    additionally supports an optional Phase 2 conditional intra-variable greedy
    refinement that, for each subsequent variable, picks every cutoff by argmax
    merged cross-variable size-2 TIRP score against previously committed
    variables (not available for log-rank scoring).

    ZERO-VARIANCE HANDLING:
    When a specific cutoff configuration produces identical durations
    (zero variance), the scoring functions skip that state (contribute 0).
    The greedy algorithm naturally prefers configurations producing
    duration variance, allowing meaningful t-test comparisons.

    ZERO-SCORE HANDLING:
    A greedy iteration whose best candidate scores exactly 0 (no state carries
    in-direction signal) never commits a cutoff. At the root, directional runs
    are re-fit with two-sided TID3 (recorded as 'tid3_two_sided' in
    fallback_variables); if that also fails, or the run was already two-sided,
    the variable routes to the TD4C fallback. At a later iteration the search
    stops early and keeps the positive-scoring cutoffs found so far.
    """

    AVAILABLE_SCORING_METHODS = {
        "max_t_stat_sum": "Maximize sum of t-statistics for each state separately",
        "max_logrank_sum": ("Censoring-aware: maximize sum of per-state log-rank statistics "
                            "between the two classes' STI duration survival curves. STIs "
                            "truncated by an observation gap or record end are right-censored. "
                            "Phase-1 univariate only (no MV refinement)."),
    }
    
    # Available duration preference options
    AVAILABLE_DURATION_PREFERENCES = {
        "two_sided": "Favor any difference in duration distributions between classes (default)",
        "class1_longer": "Favor cutoffs where class 1 has longer durations than class 0",
        "class0_longer": "Favor cutoffs where class 0 has longer durations than class 1"
    }
    
    def __init__(self, bins: int, min_duration_threshold: int = 2, max_gap: int = 1, scoring_method: str = "max_t_stat_sum", nb_candidates: int = 100, duration_preference: str = "two_sided", multivariate_refinement: bool = False, num_relations: int = 3, mv_top_tirps: int = 100, min_mean_gap: float = 0.0, state_selection_p_threshold: Optional[float] = None, fallback_method: Optional[str] = "td4c", fallback_td4c_distance: str = "cosine", mv_variable_order: str = "univariate_score_desc", mv_random_seed: Optional[int] = None):
        """
        Initialize TID3 with specified parameters.

        Parameters:
            bins (int): Desired number of bins (discretization intervals).
            min_duration_threshold (int): Minimum duration length to consider for analysis.
                                        States with fewer consecutive occurrences are filtered out.
                                        If None, defaults to 0 (no filtering).
            max_gap (int): Maximum gap between timestamps to consider as consecutive states.
                          Same logic as used in KL interval generation.
            scoring_method (str): Scoring method. One of:
                                - "max_t_stat_sum" (default): Welch t-statistic sum over states.
                                - "max_logrank_sum": censoring-aware log-rank statistic sum over
                                  states; STIs ended by an observed state change are events,
                                  STIs truncated by an observation gap (> max_gap) or by the
                                  end of the entity's record are right-censored. Not combinable
                                  with multivariate_refinement.
            nb_candidates (int): Number of candidate cutpoints to evaluate. Default is 100.
            duration_preference (str): Direction preference for duration comparison. Options:
                                - "two_sided": Favor any difference (two-tailed tests, default)
                                - "class1_longer": Favor class 1 having longer durations (one-tailed)
                                - "class0_longer": Favor class 0 having longer durations (one-tailed)
                                Use TID3.AVAILABLE_DURATION_PREFERENCES to see all options.
            multivariate_refinement (bool): If True, run Phase 2 conditional intra-variable greedy
                                           refinement after univariate cutoff selection. Variables are
                                           processed in `mv_variable_order`; for each subsequent
                                           variable V_k, every cutoff is chosen by argmax merged
                                           cross-variable Karma score against all previously fixed
                                           variables, given V_k's already-committed cutoffs at this
                                           step. V_1 keeps its univariate-best cutoffs. Default: False.
            num_relations (int): Number of Allen temporal relations for Karma computation.
                                3 = before/overlap/contain, 7 = full Allen algebra. Default: 3.
            mv_top_tirps (int): Phase 2 only, scoring_method="max_t_stat_sum" only. Divide the
                               per-combo sum of the top-K |t| values by K (padding with 0 when
                               fewer than K scorable TIRPs exist). Default: 100. Set to -1 to
                               disable the fixed denominator and divide by the actual scorable
                               count instead. Ignored for other scoring methods and for Phase 1.
            min_mean_gap (float): Minimum required absolute difference between the pooled mean
                                 durations of the two child bins produced by a candidate cutoff.
                                 During Phase 1 greedy search, a candidate is feasible iff
                                 `|mean(durL) - mean(durR)| >= min_mean_gap` (closed inequality),
                                 with durations pooled class-agnostically over the new cutoff set.
                                 Infeasible candidates are skipped before scoring. If no feasible
                                 candidate exists at the first (root) iteration, the variable is
                                 abandoned entirely (no symbols emitted, absent from KL output).
                                 If no feasible candidate exists at a later iteration, the search
                                 stops on that variable and the bins produced so far are kept.
                                 Default: 0.0 — preserves original TID3 behaviour exactly (every
                                 candidate feasible). Only applied to Phase 1 greedy; Phase 2
                                 MV conditional-greedy is unaffected. Note: with very small
                                 parent bins late in the search the mean estimate is noisy, so
                                 the gate becomes unreliable — there is no special-case for this.
            state_selection_p_threshold (float | None): Optional raw p-value threshold for
                                 per-variable state selection. If set, transform() maps states
                                 whose final per-state p-value is above this threshold, or absent
                                 from per_state_stats, to -1. Default None preserves all states.
            fallback_method (str | None): Optional fallback discretization for variables TID3
                                 cannot meaningfully handle (empty per-state stats, e.g. all STIs
                                 have duration 1; root-skipped under min_mean_gap; or zero-score
                                 at the root after the directional -> two-sided re-fit chain also
                                 failed). Currently only "td4c" is supported. Set to None to
                                 disable the fallback (such variables stay without stats and,
                                 if root-abandoned, get dropped in transform(); the two-sided
                                 re-fit for directional runs happens regardless, as it is TID3
                                 itself, not the fallback). Default "td4c".
            fallback_td4c_distance (str): Distance measure passed to the TD4C fallback. One of
                                 TD4C.AVAILABLE_DISTANCE_MEASURES ("cosine", "kullback_leibler",
                                 "entropy"). Ignored when fallback_method is None. Default "cosine".
            mv_variable_order (str): Order in which Phase 2 MV refinement visits variables. One of
                                 {"univariate_score_desc" (default), "univariate_score_asc", "random"}.
                                 Under "_desc"/"_asc" variables are sorted by self.final_scores_per_tpid;
                                 TPIDs missing a score (e.g. TD4C-fallback variables) are treated as
                                 0.0. Under "random" the order is a deterministic permutation seeded by
                                 `mv_random_seed`. Only used when multivariate_refinement=True.
            mv_random_seed (int | None): Seed passed to numpy.random.default_rng when
                                 `mv_variable_order == "random"`. Default None. Ignored otherwise.
        """
        self.bins = bins
        self.min_duration_threshold = min_duration_threshold if min_duration_threshold is not None else 0
        self.max_gap = max_gap
        self.boundaries = None
        self.entity_class = {}  # Initialize entity class mapping
        self.nb_candidates = nb_candidates
        # Validate and set scoring method
        if scoring_method not in self.AVAILABLE_SCORING_METHODS:
            raise ValueError(f"Unknown scoring method '{scoring_method}'. Available methods: {list(self.AVAILABLE_SCORING_METHODS.keys())}")
        self.scoring_method = scoring_method
        # Validate and set duration preference
        if duration_preference not in self.AVAILABLE_DURATION_PREFERENCES:
            raise ValueError(f"Unknown duration_preference '{duration_preference}'. Available options: {list(self.AVAILABLE_DURATION_PREFERENCES.keys())}")
        self.duration_preference = duration_preference
        # Multivariate refinement parameters
        self.multivariate_refinement = multivariate_refinement
        # The Phase-2 cross-variable scorer is t-stat only; defense in depth behind
        # the parse_tid3_config check so direct construction fails just as loudly.
        if self.multivariate_refinement and self.scoring_method == "max_logrank_sum":
            raise ValueError(
                "scoring_method 'max_logrank_sum' does not support Phase-2 multivariate "
                "refinement; use 'max_t_stat_sum' or set multivariate_refinement=False."
            )
        self.num_relations = num_relations
        self.mv_top_tirps = mv_top_tirps
        _ALLOWED_MV_ORDERS = {"univariate_score_desc", "univariate_score_asc", "random"}
        if mv_variable_order not in _ALLOWED_MV_ORDERS:
            raise ValueError(
                f"Unknown mv_variable_order '{mv_variable_order}'. "
                f"Allowed: {sorted(_ALLOWED_MV_ORDERS)}."
            )
        self.mv_variable_order = mv_variable_order
        if mv_random_seed is not None:
            if not isinstance(mv_random_seed, (int, np.integer)) or mv_random_seed < 0:
                raise ValueError(
                    f"mv_random_seed must be a non-negative integer or None, got {mv_random_seed!r}"
                )
            mv_random_seed = int(mv_random_seed)
        self.mv_random_seed = mv_random_seed
        self.final_scores_per_tpid = {}   # {tpid: final_t_stat_score} for summary CSV
        self.final_cutoffs_per_tpid = {}  # {tpid: list_of_cutoff_values} for summary CSV
        # Feasibility-gated cutoff search: min absolute difference between pooled mean
        # durations of the two child bins produced by a candidate cutoff. Greedy Phase 1 only.
        if min_mean_gap < 0:
            raise ValueError(f"min_mean_gap must be >= 0, got {min_mean_gap}")
        self.min_mean_gap = float(min_mean_gap)
        if state_selection_p_threshold is not None:
            if not 0 <= state_selection_p_threshold <= 1:
                raise ValueError(
                    "state_selection_p_threshold must be between 0 and 1 "
                    f"or None, got {state_selection_p_threshold}"
                )
            state_selection_p_threshold = float(state_selection_p_threshold)
        self.state_selection_p_threshold = state_selection_p_threshold
        # TPIDs whose root cutoff was infeasible under min_mean_gap; their rows are dropped
        # in transform() and they never appear in self.boundaries / states.csv / KL output.
        self.skipped_variables = set()
        # Per-state t/p stats for the final chosen cutoffs of each TPID.
        # {tpid: {state_id: {'t': float, 'p': float}}}. Populated during the cutoff search:
        # each greedy iteration's winning candidate carries the stats for its (then-final) cutoff set,
        # so by construction the last iteration's winner holds the final state stats.
        self.per_state_stats = {}
        # Fallback discretization for variables TID3 cannot meaningfully handle.
        if fallback_method is not None and fallback_method != "td4c":
            raise ValueError(
                f"Unknown fallback_method '{fallback_method}'. Supported: None, 'td4c'."
            )
        self.fallback_method = fallback_method
        # Validate fallback_td4c_distance against TD4C's available measures so misconfiguration
        # is caught at construction time rather than mid-fit. Mirror the dual-import pattern used
        # at the top of this module so it works both as a package import and as a script run.
        try:
            from .td4c import TD4C as _TD4C
        except ImportError:
            from ta_package.methods.td4c import TD4C as _TD4C
        if fallback_td4c_distance not in _TD4C.AVAILABLE_DISTANCE_MEASURES:
            raise ValueError(
                f"Unknown fallback_td4c_distance '{fallback_td4c_distance}'. "
                f"Available: {list(_TD4C.AVAILABLE_DISTANCE_MEASURES.keys())}."
            )
        self.fallback_td4c_distance = fallback_td4c_distance
        # TPIDs whose cutoffs were not produced by the configured TID3 search.
        # {tpid: method_name}, where method_name is 'td4c' (fallback discretization;
        # no per-state t/p stats, so _is_state_selected bypasses the p-value gate) or
        # 'tid3_two_sided' (directional search found no in-direction signal at the
        # root and the variable was re-fit with two-sided TID3; has stats, gated
        # normally). Exposed for diagnostics downstream.
        self.fallback_variables = {}

    def _get_scoring_function(self):
        """
        Return the scoring function for the (single) supported scoring method.

        Returns:
            callable: scorer with signature (df, cutoffs, max_gap, temporal_property_id) -> float
        """
        return self._max_t_stat_sum_scoring
    
    def _get_scipy_alternative(self) -> str:
        """
        Map duration_preference to scipy's alternative parameter for hypothesis tests.
        
        Returns:
            str: 'two-sided', 'greater', or 'less' for scipy statistical tests
        """
        if self.duration_preference == "two_sided":
            return "two-sided"
        elif self.duration_preference == "class1_longer":
            # class1 > class0: when comparing (class1, class0), we want 'greater'
            return "greater"
        elif self.duration_preference == "class0_longer":
            # class0 > class1: when comparing (class1, class0), we want 'less'
            return "less"
        else:
            return "two-sided"
    

    
    
    # ======================================================================================
    # SCORING METHODS SECTION
    # ======================================================================================
    # Each scoring method implements a different optimization strategy for finding optimal cutoffs.
    # 
    # HOW TO ADD A NEW SCORING METHOD:
    # 1. Add the method name and description to AVAILABLE_SCORING_METHODS class variable
    # 2. Implement the scoring function with signature: (df, cutoffs, max_gap, temporal_property_id) -> float
    # 3. Add the method to _get_scoring_function() method
    # 4. Add a clear header comment explaining the strategy and score interpretation
    #
    # SCORING FUNCTION REQUIREMENTS:
    # - Must accept: df (DataFrame), cutoffs (list), max_gap (int), temporal_property_id (int)
    # - Must return: float (numeric score)
    # - Should handle edge cases (no data, insufficient classes, etc.)
    # - Should be well-documented with clear strategy explanation
    #
    # EXAMPLE NEW SCORING METHOD:
    # def _my_new_scoring(self, df, cutoffs, max_gap=1, temporal_property_id=None):
    #     """
    #     My new scoring method that does X.
    #     Strategy: [explain the optimization strategy]
    #     Score interpretation: [higher/lower is better]
    #     """
    #     # Implementation here
    #     return score

    def _calculate_interval_durations(self, df: pd.DataFrame, max_gap: int = 1):
        """
        Calculate the duration of consecutive time intervals for each entity and class,
        creating temporal intervals using the same interval merging logic as KL generation.
        
        Parameters:
            df: DataFrame with columns [ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE, 'Class', 'Bin']
            max_gap: Maximum gap between timestamps to consider as consecutive
            
        Returns:
            dict: {class_id: [list of durations for each interval]}
        """
        durations_by_class = {}
        
        # Group by entity and class
        for (entity_id, class_id), entity_group in df.groupby([ENTITY_ID, 'Class']):
            if class_id not in durations_by_class:
                durations_by_class[class_id] = []
            
            # Group by temporal property within this entity
            for tpid, prop_group in entity_group.groupby(TEMPORAL_PROPERTY_ID):
                prop_group = prop_group.sort_values(TIMESTAMP)
                
                if len(prop_group) == 0:
                    continue
                
                # Create intervals using the same logic as KL generation
                intervals = []
                current_interval = None

                for _, row in prop_group.iterrows():
                    ts = row[TIMESTAMP]
                    state = row['Bin']

                    if current_interval is None:
                        current_interval = {"start": ts, "end": ts + 1, "state": state}
                    else:
                        # If the same state and the gap is within max_gap, extend the current interval
                        if state == current_interval["state"] and (ts - current_interval["end"]) <= max_gap:
                            current_interval["end"] = ts + 1
                        else:
                            # Save the current interval and start a new one
                            intervals.append(current_interval)
                            current_interval = {"start": ts, "end": ts + 1, "state": state}

                # Don't forget the last interval
                if current_interval is not None:
                    intervals.append(current_interval)

                # Calculate durations from intervals
                for interval in intervals:
                    duration = interval["end"] - interval["start"]
                    if duration >= self.min_duration_threshold:
                        durations_by_class[class_id].append(duration)

        return durations_by_class

    def _calculate_interval_durations_by_state_from_df(self, df: pd.DataFrame, max_gap: int = 1):
        """
        Calculate durations by class and state from a DataFrame that already has 'Bin' column.
        This is a helper function for scoring methods that need durations separated by state.
        
        Parameters:
            df: DataFrame with columns [ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE, 'Class', 'Bin']
            max_gap: Maximum gap between timestamps to consider as consecutive
            
        Returns:
            dict: {class_id: {state: [list of durations for this class-state combination]}}
        """
        durations_by_class_and_state = {}
        
        # Group by entity and class
        for (entity_id, class_id), entity_group in df.groupby([ENTITY_ID, 'Class']):
            if class_id not in durations_by_class_and_state:
                durations_by_class_and_state[class_id] = {}
            
            # Group by temporal property within this entity
            for tpid, prop_group in entity_group.groupby(TEMPORAL_PROPERTY_ID):
                prop_group = prop_group.sort_values(TIMESTAMP)
                
                if len(prop_group) == 0:
                    continue
                
                # Create intervals using the same logic as KL generation
                intervals = []
                current_interval = None

                for _, row in prop_group.iterrows():
                    ts = row[TIMESTAMP]
                    state = row['Bin']

                    if current_interval is None:
                        current_interval = {"start": ts, "end": ts + 1, "state": state}
                    else:
                        # If the same state and the gap is within max_gap, extend the current interval
                        if state == current_interval["state"] and (ts - current_interval["end"]) <= max_gap:
                            current_interval["end"] = ts + 1
                        else:
                            # Save the current interval and start a new one
                            intervals.append(current_interval)
                            current_interval = {"start": ts, "end": ts + 1, "state": state}

                # Don't forget the last interval
                if current_interval is not None:
                    intervals.append(current_interval)

                # Calculate durations from intervals
                for interval in intervals:
                    duration = interval["end"] - interval["start"]
                    state = interval["state"]
                    
                    if duration >= self.min_duration_threshold:
                        if state not in durations_by_class_and_state[class_id]:
                            durations_by_class_and_state[class_id][state] = []
                        durations_by_class_and_state[class_id][state].append(duration)
        
        return durations_by_class_and_state

    # ======================================================================================
    # SCORING METHOD: avg_duration_diff_sum
    # ======================================================================================
    # Strategy: Minimize sum of absolute differences in average durations between classes for each state
    # Tie-breaking: When absolute differences are equal, prefer lower standard deviations
    # Rationale: Find cutoffs that make classes have similar duration patterns in each state
    # Score interpretation: Lower is better (minimization problem)
    
  
    # ======================================================================================
    # SCORING METHOD: max_t_stat_sum
    # ======================================================================================
    # Strategy: Maximize sum of t-statistics for each state separately
    # Rationale: Find cutoffs that create the strongest statistical differences between classes in each state
    # Score interpretation: Higher is better (maximization problem)
    
    def _max_t_stat_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that evaluates cutoffs by maximizing the sum of t-statistics 
        for each state separately.
        
        This method calculates t-statistics between classes for each state independently,
        then sums all the t-statistics. Higher absolute t-statistics indicate stronger
        differences between classes.
        
        Respects duration_preference parameter:
        - "two_sided": Uses absolute t-statistic (any difference is good)
        - "class1_longer": Only counts positive t-statistics (class1 > class0)
        - "class0_longer": Only counts negative t-statistics (class0 > class1)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of t-statistics across all states (absolute or directional based on preference)
                   Higher score = better (stronger statistical differences between classes)
        """
        # Create bins using the cutoffs
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df_temp = df.copy()
        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))
        
        # Calculate durations by class and state
        durations_by_class_and_state = self._calculate_interval_durations_by_state_from_df(df_temp, max_gap)
        
        # If we don't have at least 2 classes, return 0
        classes_with_data = list(durations_by_class_and_state.keys())
        if len(classes_with_data) < 2:
            return 0.0
        
        # For directional preference, we need class 0 and class 1
        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in classes_with_data or 1 not in classes_with_data):
            return 0.0  # Need both classes for directional comparison
        
        # Calculate sum of t-statistics across all states
        total_t_stat_sum = 0.0
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate t-statistics between class pairs
        for state in states:
            state_durations_by_class = {}
            
            # Get durations for each class in this state
            for class_id in classes_with_data:
                if class_id in durations_by_class_and_state and state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_durations_by_class[class_id] = durations
                else:
                    state_durations_by_class[class_id] = []
            
            # For directional preference, only compare class 1 vs class 0
            if is_directional:
                durations_class1 = state_durations_by_class.get(1, [])
                durations_class0 = state_durations_by_class.get(0, [])
                
                if len(durations_class1) >= 2 and len(durations_class0) >= 2:
                    std1 = np.std(durations_class1)
                    std0 = np.std(durations_class0)
                    
                    if std1 == 0.0 and std0 == 0.0:
                        mean1 = np.mean(durations_class1)
                        mean0 = np.mean(durations_class0)
                        if self.duration_preference == "class1_longer" and mean1 > mean0:
                            total_t_stat_sum += (mean1 - mean0)
                        elif self.duration_preference == "class0_longer" and mean0 > mean1:
                            total_t_stat_sum += (mean0 - mean1)
                    else:
                        try:
                            # t-test with class1 as first arg: positive t means class1 > class0
                            t_stat, p_value = ttest_ind(durations_class1, durations_class0,
                                                        alternative=self._get_scipy_alternative(),
                                                        equal_var=False)
                            # For directional, only add if in the right direction
                            if self.duration_preference == "class1_longer" and t_stat > 0:
                                total_t_stat_sum += t_stat
                            elif self.duration_preference == "class0_longer" and t_stat < 0:
                                total_t_stat_sum += abs(t_stat)
                        except Exception:
                            pass
            else:
                # Two-sided: compare all pairs of classes
                class_ids = list(state_durations_by_class.keys())
                
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        durations1 = state_durations_by_class[class1]
                        durations2 = state_durations_by_class[class2]
                        
                        # Need at least 2 samples for t-test
                        if len(durations1) >= 2 and len(durations2) >= 2:
                            std1 = np.std(durations1)
                            std2 = np.std(durations2)
                            
                            if std1 == 0.0 and std2 == 0.0:
                                continue  # Skip zero-variance comparisons
                            else:
                                try:
                                    t_stat, p_value = ttest_ind(durations1, durations2, equal_var=False)
                                    total_t_stat_sum += abs(t_stat)
                                except Exception:
                                    continue
        
        return total_t_stat_sum

    # ======================================================================================
    # SCORING METHOD: max_ks_stat_sum
    # ======================================================================================
    # Strategy: Maximize sum of Kolmogorov-Smirnov statistics between class duration distributions
    # Rationale: KS test is non-parametric and compares the entire CDF, not just means
    # Score interpretation: Higher is better (maximization problem, KS statistic ranges 0-1)
    
 

    def _score_from_durations_direct(self, durations_by_class_and_state: dict,
                                     events_by_class_and_state: dict = None):
        """
        Compute the score directly from a precomputed
        {class_id: {state_id: [durations]}} structure, dispatching on scoring_method.

        Parameters:
            durations_by_class_and_state: {class_id: {state_id: [durations]}}
            events_by_class_and_state: {class_id: {state_id: [bool event flags]}}
                parallel to the durations structure; required when
                scoring_method == "max_logrank_sum", ignored otherwise.

        Returns:
            tuple[float, dict[int, dict[str, float]]]: (score, per_state_stats),
                propagated directly from the active scorer. The dict maps
                state_id -> {'t': float, 'p': float} for states that
                contributed to the score; empty when no classes have data.
        """
        classes_with_data = list(durations_by_class_and_state.keys())
        if len(classes_with_data) < 2:
            return 0.0, {}

        states = set()
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())

        if self.scoring_method == "max_logrank_sum":
            if events_by_class_and_state is None:
                raise ValueError(
                    "max_logrank_sum scoring requires the parallel event-flag structure"
                )
            return self._score_logrank_sum(
                durations_by_class_and_state,
                events_by_class_and_state,
                classes_with_data,
                states,
            )

        return self._score_t_stat_sum(
            durations_by_class_and_state,
            classes_with_data,
            states,
        )

    def _describe_zero_score_reason(self, durations, classes, states, duration_filter_stats=None,
                                    events=None) -> str:
        """Summarize why a candidate produced no positive score contribution."""
        if len(classes) < 2:
            return "fewer than 2 classes have valid durations"

        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in durations or 1 not in durations):
            return "directional scoring requires both class 0 and class 1 durations"

        if self.scoring_method == "max_logrank_sum" and events is not None:
            return self._describe_zero_score_reason_logrank(
                durations, events, classes, states, duration_filter_stats)

        insufficient_samples = 0
        zero_variance_both = 0
        ttest_exceptions = 0
        wrong_direction = 0
        valid_zero_t = 0
        evaluated_pairs = 0

        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                evaluated_pairs += 1
                if len(d1) < 2 or len(d0) < 2:
                    insufficient_samples += 1
                    continue
                if np.std(d1) == 0 and np.std(d0) == 0:
                    zero_variance_both += 1
                    continue
                try:
                    t, _ = ttest_ind(d1, d0, alternative=self._get_scipy_alternative(), equal_var=False)
                except Exception:
                    ttest_exceptions += 1
                    continue
                if self.duration_preference == "class1_longer" and t <= 0:
                    wrong_direction += 1
                elif self.duration_preference == "class0_longer" and t >= 0:
                    wrong_direction += 1
                elif t == 0:
                    valid_zero_t += 1
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        evaluated_pairs += 1
                        if len(d1) < 2 or len(d2) < 2:
                            insufficient_samples += 1
                            continue
                        if np.std(d1) == 0 and np.std(d2) == 0:
                            zero_variance_both += 1
                            continue
                        try:
                            t, _ = ttest_ind(d1, d2, equal_var=False)
                        except Exception:
                            ttest_exceptions += 1
                            continue
                        if t == 0:
                            valid_zero_t += 1

        parts = []
        if duration_filter_stats is not None:
            parts.extend([
                f"raw_stis={duration_filter_stats['raw_stis']}",
                f"dropped_by_min_duration={duration_filter_stats['dropped_by_min_duration']}",
                f"dropped_by_nan_state={duration_filter_stats['dropped_by_nan_state']}",
                f"kept_after_duration_filter={duration_filter_stats['kept_after_duration_filter']}",
            ])
        parts.extend([
            f"states={len(states)}",
            f"class_pairs_checked={evaluated_pairs}",
            f"insufficient_samples={insufficient_samples}",
            f"zero_variance_both={zero_variance_both}",
            f"ttest_exceptions={ttest_exceptions}",
        ])
        if is_directional:
            parts.append(f"wrong_direction={wrong_direction}")
        if valid_zero_t:
            parts.append(f"valid_zero_t={valid_zero_t}")
        return ", ".join(parts)

    def _describe_zero_score_reason_logrank(self, durations, events, classes, states,
                                            duration_filter_stats=None) -> str:
        """Logrank-scoring counterpart of _describe_zero_score_reason."""
        is_directional = self.duration_preference != "two_sided"
        alternative = self._get_logrank_alternative()

        insufficient_samples = 0
        all_censored = 0
        logrank_exceptions = 0
        wrong_direction = 0
        valid_zero_z = 0
        evaluated_pairs = 0

        if is_directional:
            class_pairs = [(1, 0)]
        else:
            class_pairs = [(c1, c2) for i, c1 in enumerate(classes) for c2 in classes[i+1:]]

        for state in states:
            for c1, c2 in class_pairs:
                d1 = durations.get(c1, {}).get(state, [])
                d2 = durations.get(c2, {}).get(state, [])
                evaluated_pairs += 1
                if len(d1) < 2 or len(d2) < 2:
                    insufficient_samples += 1
                    continue
                e1 = events.get(c1, {}).get(state, [])
                e2 = events.get(c2, {}).get(state, [])
                if not (any(e1) or any(e2)):
                    all_censored += 1
                    continue
                try:
                    res = logrank(x=self._censored_sample(d1, e1),
                                  y=self._censored_sample(d2, e2),
                                  alternative=alternative if is_directional else "two-sided")
                    z = float(res.statistic)
                except Exception:
                    logrank_exceptions += 1
                    continue
                # Sign convention mirrors the t-test's: class-1 longer => z < 0.
                if self.duration_preference == "class1_longer" and z >= 0:
                    wrong_direction += 1
                elif self.duration_preference == "class0_longer" and z <= 0:
                    wrong_direction += 1
                elif z == 0 or np.isnan(z):
                    valid_zero_z += 1

        parts = []
        if duration_filter_stats is not None:
            parts.extend([
                f"raw_stis={duration_filter_stats['raw_stis']}",
                f"dropped_by_min_duration={duration_filter_stats['dropped_by_min_duration']}",
                f"dropped_by_nan_state={duration_filter_stats['dropped_by_nan_state']}",
                f"kept_after_duration_filter={duration_filter_stats['kept_after_duration_filter']}",
            ])
        parts.extend([
            f"states={len(states)}",
            f"class_pairs_checked={evaluated_pairs}",
            f"insufficient_samples={insufficient_samples}",
            f"all_censored={all_censored}",
            f"logrank_exceptions={logrank_exceptions}",
        ])
        if is_directional:
            parts.append(f"wrong_direction={wrong_direction}")
        if valid_zero_z:
            parts.append(f"valid_zero_z={valid_zero_z}")
        return ", ".join(parts)

    def _score_t_stat_sum(self, durations, classes, states):
        """Score using sum of t-statistics. Respects duration_preference.

        Returns:
            tuple[float, dict[int, dict[str, float]]]: (score, per_state_stats),
                where per_state_stats maps state_id -> {'t': float, 'p': float}
                for every state that contributed to the score. States with <2
                samples in either class or zero variance in both are absent.
        """
        is_directional = self.duration_preference != "two_sided"

        # For directional, we need class 0 and 1
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0, {}

        per_pattern_stats = []
        per_state_stats = {}  # {state_id: {'t': float, 'p': float}}

        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 2 and len(d0) >= 2:
                    if np.std(d1) == 0 and np.std(d0) == 0:
                        continue
                    try:
                        t, p = ttest_ind(d1, d0, alternative=self._get_scipy_alternative(), equal_var=False)
                        if self.duration_preference == "class1_longer" and t > 0:
                            per_pattern_stats.append(t)
                            per_state_stats[state] = {'t': float(t), 'p': float(p)}
                        elif self.duration_preference == "class0_longer" and t < 0:
                            per_pattern_stats.append(abs(t))
                            per_state_stats[state] = {'t': float(t), 'p': float(p)}
                    except:
                        pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 2 and len(d2) >= 2:
                            if np.std(d1) == 0 and np.std(d2) == 0:
                                continue
                            try:
                                t, p = ttest_ind(d1, d2, equal_var=False)
                                per_pattern_stats.append(abs(t))
                                per_state_stats[state] = {'t': float(t), 'p': float(p)}
                            except:
                                pass

        return float(sum(per_pattern_stats)), per_state_stats

    def _get_logrank_alternative(self) -> str:
        """
        Map duration_preference to scipy.stats.logrank's ``alternative`` parameter.

        scipy's log-rank statistic counts (observed - expected) events in the FIRST
        sample (here class 1): when class-1 STIs survive longer there are fewer early
        class-1 events than expected, so the statistic is NEGATIVE and the matching
        one-sided alternative is 'less' — the MIRROR of the Welch t-test convention
        in _get_scipy_alternative. Verified empirically on scipy 1.13/1.14.
        """
        if self.duration_preference == "class1_longer":
            return "less"
        elif self.duration_preference == "class0_longer":
            return "greater"
        return "two-sided"

    @staticmethod
    def _censored_sample(durs, evts):
        """Build a scipy CensoredData sample from parallel duration/event-flag lists."""
        d = np.asarray(durs, dtype=np.float64)
        f = np.asarray(evts, dtype=bool)
        return CensoredData(uncensored=d[f], right=d[~f])

    def _score_logrank_sum(self, durations, events, classes, states):
        """Censoring-aware scoring: sum of per-state log-rank z-statistics.

        STIs terminated by an observed state change are events (complete durations);
        STIs truncated by an observation gap (> max_gap) or by the end of the
        entity's record enter the test right-censored at their observed length.
        Respects duration_preference; note the statistic's sign convention is the
        mirror of the t-test's (see _get_logrank_alternative).

        Parameters:
            durations: {class_id: {state_id: [durations]}}
            events: {class_id: {state_id: [bool event flags]}} parallel to `durations`
                    (True = observed state change, False = right-censored).

        Returns:
            tuple[float, dict[int, dict[str, float]]]: (score, per_state_stats),
                where per_state_stats maps state_id -> {'t': float(z), 'p': float}
                — the same keys as the t-stat scorer so downstream consumers
                (state_selection_p_threshold gating, summary CSVs, MV-phase stats
                refresh) work unchanged. States with <2 samples in either class
                or with no observed event in both classes are absent.
        """
        is_directional = self.duration_preference != "two_sided"

        # For directional, we need class 0 and 1
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0, {}

        per_pattern_stats = []
        per_state_stats = {}  # {state_id: {'t': float(z), 'p': float}}
        alternative = self._get_logrank_alternative()

        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 2 and len(d0) >= 2:
                    e1 = events.get(1, {}).get(state, [])
                    e0 = events.get(0, {}).get(state, [])
                    if not (any(e1) or any(e0)):
                        continue  # log-rank undefined when every STI is censored
                    try:
                        res = logrank(x=self._censored_sample(d1, e1),
                                      y=self._censored_sample(d0, e0),
                                      alternative=alternative)
                        z, p = float(res.statistic), float(res.pvalue)
                        if np.isnan(z):
                            continue
                        # Sign convention (empirically verified): class-1 longer => z < 0.
                        if self.duration_preference == "class1_longer" and z < 0:
                            per_pattern_stats.append(abs(z))
                            per_state_stats[state] = {'t': z, 'p': p}
                        elif self.duration_preference == "class0_longer" and z > 0:
                            per_pattern_stats.append(z)
                            per_state_stats[state] = {'t': z, 'p': p}
                    except Exception:
                        pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 2 and len(d2) >= 2:
                            e1 = events.get(c1, {}).get(state, [])
                            e2 = events.get(c2, {}).get(state, [])
                            if not (any(e1) or any(e2)):
                                continue
                            try:
                                res = logrank(x=self._censored_sample(d1, e1),
                                              y=self._censored_sample(d2, e2))
                                z, p = float(res.statistic), float(res.pvalue)
                                if np.isnan(z):
                                    continue
                                per_pattern_stats.append(abs(z))
                                per_state_stats[state] = {'t': z, 'p': p}
                            except Exception:
                                pass

        return float(sum(per_pattern_stats)), per_state_stats

    @staticmethod
    def check_mean_duration_gap_constraint(durations_left, durations_right, min_mean_gap):
        """Return True iff |mean(L) - mean(R)| >= min_mean_gap.

        Caller pools durations across class labels before calling (class-agnostic).
        Short-circuits to False if either side is empty (avoids nan mean).
        Closed inequality: ties at exactly min_mean_gap are feasible.
        min_mean_gap == 0.0 short-circuits to True so the original cutoff search
        is preserved bit-identically when the constraint is disabled.
        """
        if min_mean_gap <= 0.0:
            return True
        if not durations_left or not durations_right:
            return False

        return abs(np.mean(durations_left) - np.mean(durations_right)) >= min_mean_gap

    # ======================================================================================
    # STATE SELECTION
    # ======================================================================================

    # ======================================================================================
    # OPTIMIZED CANDIDATE SELECTION (vectorized, pre-sorted)
    # ======================================================================================
    # Pre-sorts data once and uses vectorized numpy operations for interval computation.
    # Avoids redundant df.copy(), pd.cut(), groupby(), sort_values(), and iterrows()
    # across candidate evaluations. Only for t-stat scoring variants.
    #
    # Interval concatenation logic (gap-tolerant):
    #   Original code uses interpolated end: end = ts + 1
    #   Gap = ts_next - end = ts_next - (prev_ts + 1) = ts_next - prev_ts - 1
    #   Break when gap > max_gap, i.e., ts_next - prev_ts > max_gap + 1
    #   Vectorized: np.diff(timestamps) > max_gap + 1
    #
    # Example: s1,s1,s1,s2,s1 at ts=[1,2,3,4,5] → 3 intervals:
    #   (s1, dur=3), (s2, dur=1), (s1, dur=1)
    #   The s2 interruption correctly splits s1 into two separate intervals.

    # =====================================================================
    # Multivariate Refinement Methods
    # =====================================================================

    def raw_to_STIs(self, df, cutoffs, tpid):
        """
        Convert a variable's raw data + cutoffs into intervals for Karma-style computation.

        Parameters:
            df: DataFrame with columns [ENTITY_ID, TIMESTAMP, VALUE] for one variable
            cutoffs: list of cutoff values
            tpid: TemporalPropertyID for this variable

        Returns:
            dict: {entity_id: [(start, end, symbolID, variableID), ...]} sorted by start time
        """
        if not cutoffs or len(df) == 0:
            return {}

        cutoffs_arr = np.array(sorted(cutoffs))
        entity_intervals = {}

        for entity_id, entity_group in df.groupby(ENTITY_ID):
            entity_group = entity_group.sort_values(TIMESTAMP)
            timestamps = entity_group[TIMESTAMP].values.astype(np.float64)
            values = entity_group[VALUE].values.astype(np.float64)
            n = len(timestamps)

            if n == 0:
                continue

            # Assign states using searchsorted (same as _optimized_candidate_selection)
            states = np.searchsorted(cutoffs_arr, values, side='left')

            if n == 1:
                symbol_id = int(tpid) * (self.bins + 1) + int(states[0])
                entity_intervals[entity_id] = [
                    (int(timestamps[0]), int(timestamps[0]) + 1, symbol_id, int(tpid))
                ]
                continue

            # Vectorized interval break detection (same logic as _optimized_candidate_selection)
            state_changes = np.diff(states) != 0
            time_gap_breaks = np.diff(timestamps) > self.max_gap + 1
            breaks = state_changes | time_gap_breaks

            break_idx = np.where(breaks)[0]
            start_indices = np.concatenate([[0], break_idx + 1])
            end_indices = np.concatenate([break_idx, [n - 1]])

            intervals = []
            for si, ei in zip(start_indices, end_indices):
                start_time = int(timestamps[si])
                end_time = int(timestamps[ei]) + 1
                state = int(states[si])
                symbol_id = int(tpid) * (self.bins + 1) + state
                intervals.append((start_time, end_time, symbol_id, int(tpid)))

            if intervals:
                entity_intervals[entity_id] = intervals

        return entity_intervals

    def _compute_cross_durations(self, v_intervals, other_intervals, entity_class):
        """
        Compute Karma-style cross-variable durations between V's intervals and other variables'.

        For each entity, iterates V_intervals × other_intervals, computes Allen relation
        and duration = max(end1, end2) - min(start1, start2) for co-occurring pairs.

        Parameters:
            v_intervals: {entity_id: [(start, end, symbol, variable), ...]} for variable V
            other_intervals: {entity_id: [(start, end, symbol, variable), ...]} for all other variables
            entity_class: {entity_id: class_label} mapping

        Returns:
            dict: {pattern_key: {class_id: [durations]}}
                  where pattern_key = "symbolV_symbolOther_relation"
        """
        cross_durations = {}

        for entity_id in v_intervals:
            if entity_id not in other_intervals:
                continue

            class_id = entity_class.get(entity_id, None)
            if class_id is None:
                continue

            v_ivs = v_intervals[entity_id]
            o_ivs = other_intervals[entity_id]

            for v_start, v_end, v_sym, v_var in v_ivs:
                for o_start, o_end, o_sym, o_var in o_ivs:
                    # Ensure consistent ordering that is invariant under argument swap:
                    # break ties by (start, end, variable_id, symbol_id) total order.
                    v_key = (v_start, v_end, v_var, v_sym)
                    o_key = (o_start, o_end, o_var, o_sym)
                    if o_key < v_key:
                        start1, end1, sym1 = o_start, o_end, o_sym
                        start2, end2, sym2 = v_start, v_end, v_sym
                    else:
                        start1, end1, sym1 = v_start, v_end, v_sym
                        start2, end2, sym2 = o_start, o_end, o_sym

                    # Apply max_gap filter
                    gap = start2 - end1
                    if gap > self.max_gap:
                        continue

                    # Skip if start1 > start2 (shouldn't happen after ordering above)
                    if start1 > start2:
                        continue

                    # Compute Allen relation (3-relation mode)
                    relation = self._compute_allen_relation(start1, end1, start2, end2)
                    if relation is None:
                        continue

                    # Compute duration
                    duration = max(end1, end2) - min(start1, start2)

                    # Build pattern key and store
                    key = f"{sym1}_{sym2}_{relation}"
                    if key not in cross_durations:
                        cross_durations[key] = {}
                    if class_id not in cross_durations[key]:
                        cross_durations[key][class_id] = []
                    cross_durations[key][class_id].append(duration)

        return cross_durations

    def _merge_cross_durations(self, dicts):
        """
        Merge a list of {pattern_key: {class_id: [durations]}} dicts into one.
        Per-class duration lists are concatenated.
        """
        merged = {}
        for d in dicts:
            for key, class_map in d.items():
                tgt = merged.setdefault(key, {})
                for cls, durs in class_map.items():
                    tgt.setdefault(cls, []).extend(durs)
        return merged

    def _compute_allen_relation(self, start1, end1, start2, end2):
        """
        Compute Allen temporal relation between two intervals.
        Supports 3-relation mode (before/overlap/contain) and 7-relation mode.

        Reuses logic from Karma_New_TID3.compute_relationship.

        Parameters:
            start1, end1: first interval (must have start1 <= start2)
            start2, end2: second interval

        Returns:
            str or None: relation name, or None if not_defined
        """
        epsilon = 0  # Using 0 epsilon for simplicity, matching Karma default

        s2_minus_e1 = start2 - end1
        e1_minus_s2 = end1 - start2
        s2_minus_s1 = start2 - start1
        e1_minus_e2 = end1 - end2
        e2_minus_e1 = end2 - end1

        if self.num_relations == 3:
            if epsilon < s2_minus_e1 < self.max_gap:
                return "before"
            elif s2_minus_s1 > epsilon >= abs(e1_minus_s2) and e1_minus_e2 < epsilon:
                return "before"
            elif s2_minus_s1 > epsilon < e1_minus_s2 and e1_minus_e2 < epsilon:
                return "overlap"
            elif abs(s2_minus_s1) <= epsilon and abs(e1_minus_e2) <= epsilon:
                return "contain"
            elif s2_minus_s1 > epsilon and e1_minus_e2 > epsilon:
                return "contain"
            elif abs(s2_minus_s1) <= epsilon < e2_minus_e1:
                return "contain"
            elif s2_minus_s1 > epsilon >= abs(e1_minus_e2):
                return "contain"
        elif self.num_relations == 7:
            if epsilon < s2_minus_e1 < self.max_gap:
                return "before"
            elif s2_minus_s1 > epsilon >= abs(e1_minus_s2) and e1_minus_e2 < epsilon:
                return "meet"
            elif s2_minus_s1 > epsilon < e1_minus_s2 and e1_minus_e2 < epsilon:
                return "overlap"
            elif abs(s2_minus_s1) <= epsilon and abs(e1_minus_e2) <= epsilon:
                return "equal"
            elif s2_minus_s1 > epsilon and e1_minus_e2 > epsilon:
                return "contain"
            elif abs(s2_minus_s1) <= epsilon < e2_minus_e1:
                return "starts"
            elif s2_minus_s1 > epsilon >= abs(e1_minus_e2):
                return "finishby"

        return None
    
    def _score_cross_variable(self, cross_durations):
        """
        Vectorized scoring of cross-variable size-2 TIRP time-duration patterns.
        Replaces individual scipy.stats.ttest_ind calls with batched Numpy matrix math.
        """
        if not cross_durations:
            return 0.0

        # Determine unique classes present in the data
        classes = set()
        for class_map in cross_durations.values():
            classes.update(class_map.keys())
        classes = list(classes)
        
        if len(classes) < 2:
            return 0.0

        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in classes or 1 not in classes):
            return 0.0

        patterns = list(cross_durations.keys())
        num_patterns = len(patterns)

        # 1. Determine maximum lengths to size our padded matrices
        max_lens = {c: 0 for c in classes}
        for pat in patterns:
            for c in classes:
                durs = cross_durations[pat].get(c, [])
                if len(durs) > max_lens[c]:
                    max_lens[c] = len(durs)

        # 2. Build padded Numpy arrays (fill empty spaces with NaN)
        matrices = {c: np.full((num_patterns, max_lens[c]), np.nan) for c in classes}
        
        for i, pat in enumerate(patterns):
            for c in classes:
                durs = cross_durations[pat].get(c, [])
                if durs:
                    matrices[c][i, :len(durs)] = durs

        # 3. Vectorized Math: Calculate Mean, Variance, and Count for all patterns instantly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # np.nanmean and np.nanvar safely ignore the padded NaNs
            means = {c: np.nanmean(matrices[c], axis=1) for c in classes}
            vars_ = {c: np.nanvar(matrices[c], axis=1, ddof=1) for c in classes}
            counts = {c: np.sum(~np.isnan(matrices[c]), axis=1) for c in classes}

        t_stats_all = []

        # 4. Vectorized Welch's T-Test
        if is_directional:
            n1, n0 = counts[1], counts[0]
            
            # Mask to find valid patterns: >= 2 samples in both, and not zero-variance in both
            valid = (n1 >= 2) & (n0 >= 2)
            valid &= ~((vars_[1] == 0) & (vars_[0] == 0))
            
            m1, m0 = means[1][valid], means[0][valid]
            v1, v0 = vars_[1][valid], vars_[0][valid]
            num1, num0 = n1[valid], n0[valid]
            
            # Welch's formula: (m1 - m0) / sqrt( (v1/n1) + (v0/n0) )
            denom = np.sqrt((v1 / num1) + (v0 / num0))
            # Safe division: where denom is 0, output 0
            t_stats = np.divide((m1 - m0), denom, out=np.zeros_like(m1), where=denom!=0)
            
            if self.duration_preference == "class1_longer":
                t_stats_valid = t_stats[t_stats > 0]
            else: # class0_longer
                t_stats_valid = np.abs(t_stats[t_stats < 0])
                
            t_stats_all.extend(t_stats_valid)
            
        else:
            # Two-sided testing for all class pairs
            for i, c1 in enumerate(classes):
                for c2 in classes[i+1:]:
                    n1, n2 = counts[c1], counts[c2]
                    
                    valid = (n1 >= 2) & (n2 >= 2)
                    valid &= ~((vars_[c1] == 0) & (vars_[c2] == 0))
                    
                    m1, m2 = means[c1][valid], means[c2][valid]
                    v1, v2 = vars_[c1][valid], vars_[c2][valid]
                    num1, num2 = n1[valid], n2[valid]
                    
                    denom = np.sqrt((v1 / num1) + (v2 / num2))
                    t_stats = np.divide((m1 - m2), denom, out=np.zeros_like(m1), where=denom!=0)
                    
                    t_stats_all.extend(np.abs(t_stats))

        # 5. Apply Top-K Padding and Scoring Logic
        use_fixed_k = self.mv_top_tirps is not None and self.mv_top_tirps > 0
        top_k = self.mv_top_tirps if use_fixed_k else None
        
        if top_k is not None and len(t_stats_all) > top_k:
            # Sort descending and take top K
            t_stats_all = sorted(t_stats_all, reverse=True)[:top_k]

        total_score = float(np.sum(t_stats_all))
        num_scored = len(t_stats_all)

        if use_fixed_k:
            return total_score / self.mv_top_tirps
            
        return total_score / num_scored if num_scored > 0 else 0.0

    def _score_candidates_cross_variable_iter(self, df_vk, tpid, chosen_cutoffs,
                                                candidate_pool, fixed_intervals):
        """
        Score each candidate as an additional cutoff added to V_k's chosen_cutoffs, using
        the merged cross-variable Karma duration score against all variables in
        fixed_intervals. Used as the inner per-iteration evaluator inside
        _multivariate_refinement_greedy.

        For each candidate c in candidate_pool:
          - vk_cutoffs = sorted(chosen_cutoffs + [c]); skip if c is np.isclose to any
            already-chosen cutoff (score stays -inf).
          - vk_intervals = raw_to_STIs(df_vk, vk_cutoffs, tpid).
          - For each V_j in fixed_intervals, compute pair_V_j_V_k = _compute_cross_durations.
          - merged = _merge_cross_durations([...]); score = _score_cross_variable(merged).

        Returns 1-D np.ndarray of scores aligned with candidate_pool.
        """
        scores = np.full(len(candidate_pool), -np.inf)
        for i, candidate in enumerate(candidate_pool):
            if chosen_cutoffs and any(np.isclose(candidate, chosen_cutoffs)):
                continue
            vk_cutoffs = sorted(list(chosen_cutoffs) + [candidate])
            vk_intervals = self.raw_to_STIs(df_vk, vk_cutoffs, tpid)
            per_pair = [
                self._compute_cross_durations(
                    fixed_intervals[v_j], vk_intervals, self.entity_class
                )
                for v_j in fixed_intervals
            ]
            merged = self._merge_cross_durations(per_pair)
            scores[i] = self._score_cross_variable(merged)
        return scores

    def _multivariate_refinement_greedy(self, data):
        """
        Phase 2 MV refinement: conditional intra-variable greedy.

        Variables are visited in `self.mv_variable_order` (default: descending univariate
        t-stat-sum score; ties by TPID asc). V_1 keeps its univariate-best cutoffs (no
        fixed variable to score against). For each subsequent V_k:

          1. Generate an equal-frequency candidate pool for V_k via
             generate_candidate_cutpoints (same call Phase 1 makes).
          2. For each iteration i in 1..self.bins-1:
              a. For each remaining candidate c, score (chosen_cutoffs + [c]) by the merged
                 cross-variable Karma score against every previously-fixed V_j (via
                 _score_candidates_cross_variable_iter).
              b. Commit the argmax candidate; remove it and near-duplicates from the pool.
          3. Refresh self.per_state_stats[V_k] so it matches the MV-chosen cutoffs (via a
             single _evaluate_candidate_pool call on the univariate path).

        Updates: self.boundaries, self.final_cutoffs_per_tpid, self.per_state_stats.

        Complexity (per call, ignoring constants inside _compute_cross_durations):
          - pair-dict computations: (bins-1) * nb_candidates * sum_{k=2..N}(k-1)
          - _score_cross_variable calls: (bins-1) * nb_candidates * (N-1)
          - sequential by design (step k depends on step k-1); inner-loop parallelization
            is intentionally not added — add later only if profiling demands it.
        """
        if not self.boundaries or len(self.boundaries) < 2:
            logger.info("Multivariate refinement skipped: need at least 2 variables")
            return

        # fit() attaches the 'Class' column per-TPID inside its loop; the raw `data`
        # passed in here doesn't carry it. We need it for _build_candidate_search_context
        # (which sorts by ENTITY_ID + Class + TPID + TIMESTAMP) during the per-state-stats
        # refresh, so attach it once up front.
        if self.entity_class and 'Class' not in data.columns:
            data = data.assign(Class=data[ENTITY_ID].map(self.entity_class))

        # Build variable ordering. sorted(keys) first so tie-breaking and "random"
        # permutation are deterministic regardless of dict insertion order.
        keys = sorted(self.boundaries.keys())
        if self.mv_variable_order == "univariate_score_desc":
            order = sorted(keys, key=lambda t: (-self.final_scores_per_tpid.get(t, 0.0), t))
        elif self.mv_variable_order == "univariate_score_asc":
            order = sorted(keys, key=lambda t: (self.final_scores_per_tpid.get(t, 0.0), t))
        elif self.mv_variable_order == "random":
            rng = np.random.default_rng(self.mv_random_seed)
            order = [int(t) for t in rng.permutation(keys)]
            logger.info(
                f"[MV_REFINE_GREEDY] random order seed={self.mv_random_seed}, order={order}"
            )
            print(
                f"[MV_REFINE_GREEDY] random order seed={self.mv_random_seed}, order={order}",
                flush=True,
            )
        else:
            raise ValueError(f"Unknown mv_variable_order '{self.mv_variable_order}'")

        if len(order) < 2:
            logger.info("Multivariate refinement skipped: <2 usable variables")
            return

        logger.info(
            f"[MV_REFINE_GREEDY] Start: {len(order)} variables, "
            f"order_scheme={self.mv_variable_order}, order={order}"
        )
        print(
            f"[MV_REFINE_GREEDY] Start: {len(order)} vars, "
            f"order_scheme={self.mv_variable_order}",
            flush=True,
        )

        fixed_intervals = {}
        changes = 0

        for k, v_k in enumerate(order, start=1):
            df_vk = data[data[TEMPORAL_PROPERTY_ID] == v_k]
            prev_cutoffs = list(self.boundaries[v_k])

            if k == 1:
                fixed_intervals[v_k] = self.raw_to_STIs(df_vk, prev_cutoffs, v_k)
                logger.info(
                    f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} — kept "
                    f"univariate-best (no fixed variables yet), cutoffs={prev_cutoffs}"
                )
                print(
                    f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} kept "
                    f"univariate-best, cutoffs={prev_cutoffs}",
                    flush=True,
                )
                continue

            if df_vk.empty:
                logger.warning(
                    f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} has no data; "
                    f"keeping previous cutoffs={prev_cutoffs}"
                )
                fixed_intervals[v_k] = {}
                continue

            candidate_pool = generate_candidate_cutpoints(df_vk, self.nb_candidates)
            if not candidate_pool:
                logger.warning(
                    f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} produced empty "
                    f"candidate pool; keeping previous cutoffs={prev_cutoffs}"
                )
                fixed_intervals[v_k] = self.raw_to_STIs(df_vk, prev_cutoffs, v_k)
                continue

            # Fixed-context log: enumerate the previously-committed variables and
            # the static states/cutoffs that V_k's greedy iterations will score
            # against. Symbol IDs in fixed_intervals follow the raw_to_STIs
            # convention symbol_id = v_j * (bins+1) + state, so state is
            # recoverable inline.
            fixed_var_ids = list(fixed_intervals.keys())
            header = (
                f"[MV_REFINE_GREEDY] Step {k}/{len(order)}, var={v_k}: "
                f"scoring against {len(fixed_var_ids)} fixed variable(s) "
                f"{fixed_var_ids}"
            )
            logger.info(header)
            print(header, flush=True)
            for v_j in fixed_var_ids:
                j_cutoffs = list(self.boundaries.get(v_j, []))
                j_states = sorted({
                    sym - v_j * (self.bins + 1)
                    for ivs in fixed_intervals[v_j].values()
                    for (_, _, sym, _) in ivs
                })
                j_symbols = sorted({
                    sym
                    for ivs in fixed_intervals[v_j].values()
                    for (_, _, sym, _) in ivs
                })
                line = (
                    f"[MV_REFINE_GREEDY]   fixed V_{v_j}: cutoffs={j_cutoffs} "
                    f"states={j_states} symbols={j_symbols}"
                )
                logger.info(line)
                print(line, flush=True)

            chosen_cutoffs = []
            for i in range(1, self.bins):
                scores = self._score_candidates_cross_variable_iter(
                    df_vk, v_k, chosen_cutoffs, candidate_pool, fixed_intervals
                )
                if len(scores) == 0 or np.all(np.isneginf(scores)):
                    logger.warning(
                        f"[MV_REFINE_GREEDY]   Step {k}/{len(order)}, var={v_k}: "
                        f"early-stop at iter {i}/{self.bins-1}; no valid candidates"
                    )
                    break

                best_idx = int(np.argmax(scores))
                best_candidate = candidate_pool[best_idx]
                best_score = float(scores[best_idx])

                chosen_so_far = list(chosen_cutoffs)
                fixed_vars_now = list(fixed_intervals.keys())
                print(
                    f"  [MV_REFINE_GREEDY]   Step {k}/{len(order)}, var={v_k}, "
                    f"iter {i}/{self.bins-1}: winner={best_candidate:.4f} "
                    f"(score={best_score:.6f}) | chosen_so_far={chosen_so_far} "
                    f"| fixed_vars={fixed_vars_now}",
                    flush=True,
                )

                chosen_cutoffs.append(best_candidate)
                chosen_cutoffs.sort()
                candidate_pool.pop(best_idx)
                candidate_pool = [
                    c for c in candidate_pool
                    if not any(np.isclose(c, chosen_cutoffs))
                ]

                if len(candidate_pool) == 0 and i < self.bins - 1:
                    logger.warning(
                        f"[MV_REFINE_GREEDY]   Step {k}/{len(order)}, var={v_k}: pool "
                        f"exhausted at iter {i}; keeping {len(chosen_cutoffs)} cutoff(s)"
                    )

            # Commit V_k cutoffs and intervals.
            self.boundaries[v_k] = list(chosen_cutoffs)
            self.final_cutoffs_per_tpid[v_k] = list(chosen_cutoffs)
            fixed_intervals[v_k] = self.raw_to_STIs(df_vk, chosen_cutoffs, v_k)
            changed = chosen_cutoffs != prev_cutoffs
            if changed:
                changes += 1

            # Refresh per_state_stats so it reflects the MV-chosen cutoffs (instead of
            # the univariate-best from Phase 1). This keeps state_selection_p_threshold
            # filtering at transform() time consistent with the actual cutoffs.
            if chosen_cutoffs:
                try:
                    ctx = self._build_candidate_search_context(df_vk, v_k)
                    _, refresh_stats_list, _ = self._evaluate_candidate_pool(
                        ctx,
                        existing_cutoffs=list(chosen_cutoffs[:-1]),
                        pool=[chosen_cutoffs[-1]],
                        iteration_label=f"MV per-state-stats refresh, var={v_k}",
                        apply_min_mean_gap=False,
                    )
                    if refresh_stats_list and refresh_stats_list[0] is not None:
                        self.per_state_stats[v_k] = refresh_stats_list[0]
                except Exception as exc:
                    logger.warning(
                        f"[MV_REFINE_GREEDY]   Step {k}/{len(order)}, var={v_k}: "
                        f"per_state_stats refresh failed ({exc}); leaving previous stats"
                    )

            logger.info(
                f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} complete: "
                f"cutoffs={chosen_cutoffs}, changed={'Y' if changed else 'N'}"
            )
            print(
                f"[MV_REFINE_GREEDY] Step {k}/{len(order)}: var={v_k} "
                f"{'UPDATED' if changed else 'unchanged'} "
                f"before={prev_cutoffs} -> after={chosen_cutoffs}",
                flush=True,
            )

        logger.info(
            f"[MV_REFINE_GREEDY] Complete: {changes}/{len(order)} variables refined, "
            f"ordering={self.mv_variable_order}"
        )
        print(
            f"[MV_REFINE_GREEDY] Complete: {changes}/{len(order)} variables refined, "
            f"ordering={self.mv_variable_order}",
            flush=True,
        )

    # =====================================================================
    # End of Multivariate Refinement Methods
    # =====================================================================

    def _build_candidate_search_context(self, df: pd.DataFrame, temporal_property_id: int = None):
        """
        Pre-compute the numpy arrays and break masks consumed by _evaluate_candidate_pool.

        Sorts data once, computes per-entity group boundaries (one group per
        entity/class/tpid), the time-gap-break mask (positions where a STI cannot span
        regardless of state assignment), and the NaN mask. Passing this context dict
        explicitly — rather than capturing it in a closure — lets the candidate-evaluation
        logic be reused by the greedy univariate path AND by the per-state-stats refresh
        after MV refinement picks new cutoffs.
        """
        df_sorted = df.sort_values([ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID, TIMESTAMP])

        timestamps = df_sorted[TIMESTAMP].values.astype(np.float64)
        values = df_sorted[VALUE].values.astype(np.float64)
        classes_arr = df_sorted['Class'].values

        n_total = len(df_sorted)

        if n_total <= 1:
            logger.error(f"Insufficient data for TemporalPropertyID {temporal_property_id}: n_total={n_total}. At least 2 observations are required.")
            raise ValueError(f"_build_candidate_search_context requires at least 2 observations for TemporalPropertyID {temporal_property_id}, got {n_total}")

        group_labels = df_sorted.groupby(
            [ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID], sort=False
        ).ngroup().values

        group_boundary_mask = np.diff(group_labels) != 0
        group_boundaries = np.where(group_boundary_mask)[0] + 1
        group_starts = np.concatenate([[0], group_boundaries])
        group_ends = np.concatenate([group_boundaries, [n_total]])

        num_groups = len(group_starts)
        group_classes = classes_arr[group_starts]

        # Pre-compute positions where no candidate interval can ever span across, regardless of cutpoint choice.
        # A hard boundary occurs when the time gap between consecutive observations exceeds the allowed gap (max_gap).
        # This is computed once here — time gaps depend only on timestamps, not on state assignment.
        # Each observation spans [ts, ts+1), so the gap is ts_next - ts_prev - 1.
        # A hard boundary occurs when ts_next - ts_prev > max_gap + 1.
        # These boundaries are later combined with state changes to produce the final STI segmentation.
        time_gap_breaks = np.zeros(n_total, dtype=bool)
        for g in range(num_groups):
            s, e = group_starts[g], group_ends[g]
            if e - s > 1:
                time_gap_breaks[s + 1:e] = np.diff(timestamps[s:e]) > self.max_gap + 1

        nan_mask = np.isnan(values)
        has_nans = np.any(nan_mask)

        return {
            'timestamps': timestamps,
            'values': values,
            'classes_arr': classes_arr,
            'group_starts': group_starts,
            'group_ends': group_ends,
            'group_classes': group_classes,
            'time_gap_breaks': time_gap_breaks,
            'nan_mask': nan_mask,
            'has_nans': has_nans,
            'num_groups': num_groups,
            'n_total': n_total,
            'tpid': temporal_property_id,
        }

    def _evaluate_candidate_pool(self, ctx, existing_cutoffs, pool, iteration_label,
                                  apply_min_mean_gap=False):
        """
        Score each candidate in `pool` as an additional cutoff added to `existing_cutoffs`.

        Builds STI durations per (class, state) over the candidate's full cutoff list,
        applies the optional min_mean_gap feasibility gate, and scores via
        _score_from_durations_direct. Returns parallel arrays/lists over `pool`.

        apply_min_mean_gap gates each candidate by the mean-duration-gap constraint
        BEFORE scoring (greedy Phase 1 only; MV path calls with the default False).
        """
        timestamps = ctx['timestamps']
        values = ctx['values']
        group_starts = ctx['group_starts']
        group_ends = ctx['group_ends']
        group_classes = ctx['group_classes']
        time_gap_breaks = ctx['time_gap_breaks']
        nan_mask = ctx['nan_mask']
        has_nans = ctx['has_nans']
        num_groups = ctx['num_groups']

        scores = np.full(len(pool), -np.inf)
        per_state_stats_list = [None] * len(pool)
        zero_score_reasons = [None] * len(pool)
        gate_active = apply_min_mean_gap and self.min_mean_gap > 0.0
        show_progress = logger.isEnabledFor(logging.DEBUG)
        pbar = tqdm(total=len(pool),
                    desc=f"    Evaluating candidates ({iteration_label})",
                    disable=not show_progress,
                    leave=False)

        for i, candidate in enumerate(pool):
            if len(existing_cutoffs) > 0 and any(np.isclose(candidate, existing_cutoffs)):
                pbar.update(1)
                continue

            cutoffs = np.sort(np.append(existing_cutoffs, candidate))
            states = np.searchsorted(cutoffs, values, side='left')

            if has_nans:
                states = states.astype(np.int64)
                states[nan_mask] = -1

            durations_by_class_and_state = {}
            # Parallel event-flag structure, built only for censoring-aware scoring so the
            # t-stat path stays byte-identical. True = STI ended by an observed state
            # change; False = right-censored (gap-truncated or record end).
            need_censoring = self.scoring_method == "max_logrank_sum"
            events_by_class_and_state = {} if need_censoring else None
            duration_filter_stats = {
                "raw_stis": 0,
                "dropped_by_min_duration": 0,
                "dropped_by_nan_state": 0,
                "kept_after_duration_filter": 0,
            }

            # Build STI durations per class and state across all entities
            for g in range(num_groups):
                # g = one entity's observations for this variable
                s, e = group_starts[g], group_ends[g]
                class_id = group_classes[g]
                n = e - s  # number of observations for this entity

                if n == 0:
                    continue

                if class_id not in durations_by_class_and_state:
                    durations_by_class_and_state[class_id] = {}
                    if need_censoring:
                        events_by_class_and_state[class_id] = {}

                st_group = states[s:e]
                ts_group = timestamps[s:e]

                # Single observation: STI duration is always 1 by convention
                if n == 1:
                    dur = 1
                    state = int(st_group[0])
                    duration_filter_stats["raw_stis"] += 1
                    if dur >= self.min_duration_threshold and state != -1:
                        duration_filter_stats["kept_after_duration_filter"] += 1
                        durations_by_class_and_state[class_id].setdefault(state, []).append(dur)
                        if need_censoring:
                            # Sole observation reaches the record end: right-censored.
                            events_by_class_and_state[class_id].setdefault(state, []).append(False)
                    elif dur < self.min_duration_threshold:
                        duration_filter_stats["dropped_by_min_duration"] += 1
                    elif state == -1:
                        duration_filter_stats["dropped_by_nan_state"] += 1
                    continue

                # Detect STI boundaries: state change or time gap exceeds allowed gap
                state_changes = np.diff(st_group) != 0
                breaks = state_changes | time_gap_breaks[s + 1:e]

                # Convert break positions to STI start/end index pairs
                break_idx = np.where(breaks)[0]
                start_indices = np.concatenate([[0], break_idx + 1])
                end_indices = np.concatenate([break_idx, [n - 1]])

                # Compute STI durations: end timestamp + 1 - start timestamp
                durations_arr = ts_group[end_indices] + 1 - ts_group[start_indices]
                interval_states = st_group[start_indices]
                duration_filter_stats["raw_stis"] += len(durations_arr)

                # Filter out STIs below min duration and NaN-valued intervals
                min_duration_valid = durations_arr >= self.min_duration_threshold
                duration_filter_stats["dropped_by_min_duration"] += int(np.sum(~min_duration_valid))
                valid = min_duration_valid
                if has_nans:
                    non_nan_state = interval_states != -1
                    duration_filter_stats["dropped_by_nan_state"] += int(
                        np.sum(min_duration_valid & ~non_nan_state)
                    )
                    valid = valid & non_nan_state

                durations_valid = durations_arr[valid]
                states_valid = interval_states[valid]
                duration_filter_stats["kept_after_duration_filter"] += len(durations_valid)

                if need_censoring:
                    # Event iff the STI ended with an observed state change NOT coincident
                    # with an observation gap (a gap-coincident change means the true end
                    # was unobserved). The final STI reaches the record end: censored.
                    gap_slice = time_gap_breaks[s + 1:e]
                    event_flags = np.zeros(len(end_indices), dtype=bool)
                    if len(break_idx) > 0:
                        event_flags[:-1] = state_changes[break_idx] & ~gap_slice[break_idx]
                    events_valid = event_flags[valid]

                # Accumulate durations into class -> state -> [durations] structure
                class_dict = durations_by_class_and_state[class_id]
                if need_censoring:
                    events_class_dict = events_by_class_and_state[class_id]
                for j in range(len(durations_valid)):
                    state = int(states_valid[j])
                    if state not in class_dict:
                        class_dict[state] = []
                    class_dict[state].append(int(durations_valid[j]))
                    if need_censoring:
                        events_class_dict.setdefault(state, []).append(bool(events_valid[j]))

            # Mean-duration-gap feasibility gate (greedy Phase 1 only).
            # Inserting `candidate` into the sorted cutoff list splits exactly one
            # existing bin into two; those new state IDs are k and k+1 under
            # searchsorted(side='left'). Pool durations class-agnostically and
            # require |mean(L) - mean(R)| >= min_mean_gap. Skip scoring on failure.
            if gate_active:
                k = int(np.searchsorted(cutoffs, candidate, side='left'))
                dur_L = []
                dur_R = []
                for class_dict in durations_by_class_and_state.values():
                    dur_L.extend(class_dict.get(k, []))
                    dur_R.extend(class_dict.get(k + 1, []))
                if not self.check_mean_duration_gap_constraint(dur_L, dur_R, self.min_mean_gap):
                    # scores[i] remains -np.inf; do not score this candidate
                    pbar.update(1)
                    continue

            # Score this candidate by how well STI durations separate the classes
            try:
                scores[i], per_state_stats_list[i] = self._score_from_durations_direct(
                    durations_by_class_and_state, events_by_class_and_state)
                if scores[i] == 0.0:
                    if per_state_stats_list[i]:
                        zero_score_reasons[i] = "valid tests existed, but all t-stat contributions were 0"
                    else:
                        classes_with_data = list(durations_by_class_and_state.keys())
                        candidate_states = set()
                        for class_dict in durations_by_class_and_state.values():
                            candidate_states.update(class_dict.keys())
                        zero_score_reasons[i] = self._describe_zero_score_reason(
                            durations_by_class_and_state,
                            classes_with_data,
                            candidate_states,
                            duration_filter_stats,
                            events=events_by_class_and_state,
                        )
            except Exception:
                scores[i] = -np.inf

            pbar.update(1)

        pbar.close()
        return scores, per_state_stats_list, zero_score_reasons

    def _run_greedy_search(self, ctx, initial_cutpoints, candidate_pool, target_bins,
                            temporal_property_id, apply_min_mean_gap=True):
        """
        Greedy iterative cutpoint selection starting from `initial_cutpoints`.

        At each iteration calls _evaluate_candidate_pool over the remaining pool,
        appends the argmax candidate, removes it and near-duplicates from the pool,
        and continues until target_bins - 1 cutpoints exist or candidates run out.

        Zero-score handling: a best score of exactly 0.0 means no candidate carries
        any in-direction signal (argmax over all-zero scores would just return pool
        order), so such a candidate is never committed. At the root the search
        aborts; at a later iteration it stops early and the positive-scoring
        cutpoints found so far are kept. Invariant: every committed cutpoint
        had score > 0.

        Returns (cutpoints, scores, per_state_stats, abort_reason). per_state_stats
        corresponds to the final cutpoints (by construction the last committed
        iteration's winner holds the stats for the final set). abort_reason is None
        on success; on root abort cutpoints is None and abort_reason is one of:
          - "min_mean_gap_root": iteration 1 had no feasible candidate under
            min_mean_gap.
          - "zero_score_root": iteration 1's best candidate scored 0.0.
        """
        chosen_cutpoints = list(initial_cutpoints)
        chosen_scores = []
        # Local copy so we don't mutate the caller's pool; pre-filter anything already chosen.
        candidate_pool = list(candidate_pool)
        if chosen_cutpoints:
            candidate_pool = [c for c in candidate_pool
                              if not any(np.isclose(c, chosen_cutpoints))]

        latest_per_state_stats = None
        start_iter = len(chosen_cutpoints) + 1
        for iteration in range(start_iter, target_bins):
            scores, per_state_stats_list, zero_score_reasons = self._evaluate_candidate_pool(
                ctx, chosen_cutpoints, candidate_pool,
                f"bin {iteration}/{target_bins-1}",
                apply_min_mean_gap=apply_min_mean_gap,
            )

            # Check for valid candidates
            if len(scores) == 0:
                logger.warning(f"Early termination at iteration {iteration}: No candidate scores available")
                break
            elif np.all(np.isneginf(scores)):
                # Distinguish root-skip from non-root early-stop.
                # Root (iteration==1) with no feasible candidate under min_mean_gap means
                # the variable cannot be split at all — abandon it entirely. Signal to
                # _generate_cutpoints / fit() by returning None as the cutpoints sentinel.
                if iteration == 1 and self.min_mean_gap > 0.0 and len(chosen_cutpoints) == 0:
                    logger.warning(
                        f"TemporalPropertyID {temporal_property_id}: skipped — "
                        f"no feasible root cutoff under min_mean_gap={self.min_mean_gap}"
                    )
                    return None, [], None, "min_mean_gap_root"
                if self.min_mean_gap > 0.0:
                    logger.info(
                        f"TemporalPropertyID {temporal_property_id}: early-stop at iteration "
                        f"{iteration} — no feasible candidate under min_mean_gap="
                        f"{self.min_mean_gap}; keeping {len(chosen_cutpoints)} cutoff(s) / "
                        f"{len(chosen_cutpoints) + 1} bins"
                    )
                else:
                    logger.warning(f"Early termination at iteration {iteration}: All {len(scores)} candidates produced invalid scores")
                break

            # Select best candidate
            best_idx = int(np.argmax(scores))
            best_candidate = candidate_pool[best_idx]
            best_score = scores[best_idx]

            if best_score == 0.0:
                # No candidate carries in-direction signal; committing the argmax of
                # an all-zero array would place a cutoff by pool order, not by data.
                if len(chosen_cutpoints) == 0:
                    msg = (
                        f"TPID {temporal_property_id}: zero-score at root — all "
                        f"{len(scores)} candidates scored 0 "
                        f"(reason: {zero_score_reasons[best_idx]})"
                    )
                    logger.warning(msg)
                    print(f"  [TID3]   {msg}")
                    return None, [], None, "zero_score_root"
                msg = (
                    f"TPID {temporal_property_id}: zero-score at iteration {iteration} "
                    f"— keeping {len(chosen_cutpoints)} cutoff(s) / "
                    f"{len(chosen_cutpoints) + 1} bins (scores={chosen_scores})"
                )
                logger.info(msg)
                print(f"  [TID3]   {msg}")
                break

            latest_per_state_stats = per_state_stats_list[best_idx]
            print(f"  [TID3]   Iter {iteration}/{target_bins-1}: winner={best_candidate:.4f} (score={best_score:.4f})")

            chosen_cutpoints.append(best_candidate)
            chosen_cutpoints.sort()
            chosen_scores.append(best_score)
            self._log_contributing_states(iteration, chosen_cutpoints, latest_per_state_stats)

            # Remove chosen candidate and near-duplicates from pool
            candidate_pool.pop(best_idx)
            candidate_pool = [c for c in candidate_pool
                              if not any(np.isclose(c, chosen_cutpoints))]

            # Check if pool is exhausted
            if len(candidate_pool) == 0 and iteration < target_bins - 1:
                logger.warning(f"Candidate pool exhausted at iteration {iteration}: achieved {len(chosen_cutpoints) + 1}/{target_bins} bins")

        return chosen_cutpoints, chosen_scores, latest_per_state_stats, None

    @staticmethod
    def _log_contributing_states(iteration, sorted_cutpoints, per_state_stats):
        """One-line log of the states contributing to the committed winner's score.

        State k is the interval between sorted cutoffs k-1 and k (searchsorted
        side='left' convention; outer edges are -inf/inf). On log-rank runs the
        't' key holds the z-statistic.
        """
        if not per_state_stats:
            return
        edges = [-np.inf] + list(sorted_cutpoints) + [np.inf]
        parts = []
        for state in sorted(per_state_stats):
            stats = per_state_stats[state]
            parts.append(
                f"state {state}=[{edges[state]:.4g}, {edges[state + 1]:.4g}): "
                f"t={stats['t']:.2f}, p={stats['p']:.4g}"
            )
        print(f"  [TID3]   Iter {iteration}: contributing states: " + "; ".join(parts))

    def _optimized_candidate_selection(self, df: pd.DataFrame, nb_bins: int, temporal_property_id: int = None):
        """
        Greedy univariate candidate selection for the t-stat-sum scoring path.

        Builds the precomputed numpy context, generates the equal-frequency candidate pool,
        and delegates to _run_greedy_search for the iterative cutpoint selection.

        If a directional search aborts with "zero_score_root" (no candidate had any
        in-direction signal), the variable is re-fit with two-sided TID3 on the same
        context and pool — duration divergence may exist in the opposite direction,
        which two-sided scoring still exploits (unlike the duration-blind TD4C
        fallback, which remains the last resort via the None sentinel).

        Returns:
            tuple: (chosen_cutpoints, chosen_scores). chosen_cutpoints is None if the
            variable was root-abandoned (min_mean_gap-infeasible, or zero-score at
            root with no usable two-sided re-fit).
        """
        ctx = self._build_candidate_search_context(df, temporal_property_id)
        candidate_pool = generate_candidate_cutpoints(df, self.nb_candidates)

        print(f"\n  [TID3] Variable {temporal_property_id}: {len(candidate_pool)} initial candidates "
              f"(equal-frequency from {len(df[VALUE].dropna().unique())} unique values)")
        if candidate_pool:
            print(f"  [TID3]   Range: [{candidate_pool[0]:.4f}, {candidate_pool[-1]:.4f}]")

        chosen_cutpoints, chosen_scores, latest_per_state_stats, abort_reason = self._run_greedy_search(
            ctx, initial_cutpoints=[], candidate_pool=candidate_pool,
            target_bins=nb_bins, temporal_property_id=temporal_property_id,
            apply_min_mean_gap=True,
        )

        if abort_reason == "zero_score_root" and self.duration_preference != "two_sided":
            msg = (
                f"TPID {temporal_property_id}: re-fitting with two-sided TID3 "
                f"(directional hypothesis '{self.duration_preference}' found no "
                f"in-direction signal)"
            )
            logger.warning(msg)
            print(f"  [TID3]   {msg}")
            original_preference = self.duration_preference
            try:
                self.duration_preference = "two_sided"
                chosen_cutpoints, chosen_scores, latest_per_state_stats, abort_reason = \
                    self._run_greedy_search(
                        ctx, initial_cutpoints=[], candidate_pool=candidate_pool,
                        target_bins=nb_bins, temporal_property_id=temporal_property_id,
                        apply_min_mean_gap=True,
                    )
            finally:
                self.duration_preference = original_preference
            if chosen_cutpoints is not None:
                self.fallback_variables[temporal_property_id] = "tid3_two_sided"

        if chosen_cutpoints is None:
            return None, []

        cutpoints_str = ", ".join([f"{c:.4f}" for c in chosen_cutpoints])
        print(f"  [TID3]   Final cutpoints: [{cutpoints_str}] → {len(chosen_cutpoints)+1} bins")

        if latest_per_state_stats is not None:
            self.per_state_stats[temporal_property_id] = latest_per_state_stats

        return chosen_cutpoints, chosen_scores

    def _generate_cutpoints(self, df: pd.DataFrame, temporal_property_id: int = None):
        """
        For a given DataFrame (corresponding to one variable), choose candidate cutpoints
        that maximize exploitation of TID3 duration patterns between classes.

        If all durations are identical (zero variance), automatically falls back to
        TD4C-style distribution-based scoring.
        """
        if temporal_property_id is None:
            logger.error("_generate_cutpoints called without a temporal_property_id; this argument is required.")
            raise ValueError("_generate_cutpoints requires a temporal_property_id")

        # Handle case where all values are the same
        if df[VALUE].nunique() == 1:
            logger.warning(f"Insufficient variability for TemporalPropertyID {temporal_property_id}: only 1 unique value ({df[VALUE].iloc[0]})")
            candidates = [df[VALUE].min()] * (self.bins - 1)
            return candidates

        # Use optimized path for the duration-divergence scorers (t-stat and log-rank
        # share the same STI-duration construction; log-rank additionally carries
        # censor flags built inside _evaluate_candidate_pool)
        if self.scoring_method in ("max_t_stat_sum", "max_logrank_sum"):
            candidates, scores = self._optimized_candidate_selection(df, self.bins, temporal_property_id)
        else:
            # Fall back to generic candidate_selection for other scoring methods
            scoring_func = self._get_scoring_function()
            scoring_wrapper = lambda d, cutoffs: scoring_func(d, cutoffs, self.max_gap, temporal_property_id)
            candidates, scores = candidate_selection(
                df,
                self.bins,
                scoring_wrapper,
                nb_candidates=self.nb_candidates
            )

        # Root-abandon sentinel from _optimized_candidate_selection (min_mean_gap
        # infeasible at root, or zero-score at root with no usable two-sided re-fit):
        # propagate up so fit() can record the TPID in self.skipped_variables and
        # omit it from self.boundaries.
        if candidates is None:
            return None

        # Log final bin analysis
        achieved_bins = len(candidates) + 1
        if achieved_bins < self.bins:
            logger.warning(f"TemporalPropertyID {temporal_property_id}: achieved {achieved_bins}/{self.bins} bins - missing {self.bins - achieved_bins} bins due to optimization constraints")

        # Store final score for scores summary CSV
        final_score = scores[-1] if scores and len(scores) > 0 else 0.0
        if temporal_property_id is not None:
            self.final_scores_per_tpid[temporal_property_id] = final_score
            self.final_cutoffs_per_tpid[temporal_property_id] = list(candidates)

        return candidates

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the TID3 model by generating cutpoints for each variable based on
        exploitation of interval duration patterns.
        """
        
        boundaries = {}
        temporal_properties = list(data.groupby(TEMPORAL_PROPERTY_ID).groups.keys())

        logger.info(f"Processing {len(temporal_properties)} temporal properties sequentially")

        # Phase 1: Univariate cutoff selection
        phase1_start = time.time()

        # Create progress bar for sequential processing
        pbar = tqdm(total=len(temporal_properties),
                   desc="🌊 Processing temporal properties",
                   unit="property",
                   ncols=120)

        for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
            try:
                if not self.entity_class:
                    raise ValueError('No entity class mapping found')
                group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))

                # Process this temporal property directly
                cutpoints = self._generate_cutpoints(group, tpid)

                # Root-abandoned (min_mean_gap-infeasible, or zero-score at root with no
                # usable two-sided re-fit): no symbols emitted for this variable.
                # Do NOT add tpid to boundaries; record in skipped_variables so transform()
                # drops its rows (or _run_td4c_fallback rescues it). The warning is
                # already logged by the greedy search.
                if cutpoints is None:
                    self.skipped_variables.add(tpid)
                    pbar.set_postfix({
                        'current_tpid': tpid,
                        'bins': 'SKIPPED',
                        'completed': f'{pbar.n + 1}/{len(temporal_properties)}'
                    })
                    pbar.update(1)
                    continue

                boundaries[tpid] = cutpoints

                # Update progress bar
                achieved_bins = len(boundaries[tpid]) + 1
                pbar.set_postfix({
                    'current_tpid': tpid,
                    'bins': f'{achieved_bins}/{self.bins}',
                    'completed': f'{pbar.n + 1}/{len(temporal_properties)}'
                })
                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to process TemporalPropertyID {tpid}: {e}")
                # boundaries[tpid] = []
                # pbar.update(1)
                raise ValueError(f"Error processing TemporalPropertyID {tpid}: {e}")


        pbar.close()

        self.boundaries = boundaries

        self.runtime_phase1_s = time.time() - phase1_start

        # Fallback for variables TID3 couldn't meaningfully discretize. Runs after Phase 1 so it
        # sees the full per_state_stats / skipped_variables state, and before Phase 2 so any
        # multivariate refinement sees the fallback cutpoints as the starting point.
        self._run_td4c_fallback(data)

        # Phase 2: Multivariate refinement - conditional intra-variable greedy
        # (cross-variable Karma duration scores against previously committed variables).
        self.runtime_phase2_s = 0.0
        if self.multivariate_refinement:
            logger.info("Starting multivariate refinement via Karma cross-variable scoring...")
            phase2_start = time.time()
            self._multivariate_refinement_greedy(data)
            self.runtime_phase2_s = time.time() - phase2_start

    def _run_td4c_fallback(self, data: pd.DataFrame) -> None:
        """Re-fit cutpoints via TD4C for variables TID3 couldn't meaningfully handle.

        A variable is eligible if either:
        - It has an entry in ``self.per_state_stats`` that is an empty dict (TID3 produced
          cutoffs but no state ever yielded a finite t/p), or
        - It is in ``self.skipped_variables`` (root-abandoned: min_mean_gap-infeasible at
          the root, or zero-score at the root including a failed two-sided re-fit; TID3
          emitted no boundaries and ``transform()`` would otherwise drop its rows).

        On success, the TD4C cutpoints are written into ``self.boundaries[tpid]``, the tpid
        is recorded in ``self.fallback_variables`` (so ``_is_state_selected`` bypasses the
        p-value gate for it), and root-skipped tpids are removed from
        ``self.skipped_variables`` so their rows survive ``transform()``. On failure the
        variable's state is left unchanged.
        """
        if self.fallback_method != "td4c":
            return

        empty_stats_tpids = {
            tpid for tpid, stats in self.per_state_stats.items()
            if isinstance(stats, dict) and len(stats) == 0 and tpid in self.boundaries
        }
        root_skipped_tpids = set(self.skipped_variables)
        fallback_tpids = sorted(empty_stats_tpids | root_skipped_tpids)
        if not fallback_tpids:
            return

        try:
            from .td4c import TD4C
        except ImportError:
            from ta_package.methods.td4c import TD4C

        logger.info(
            f"TID3 TD4C fallback: re-fitting {len(fallback_tpids)} variable(s) "
            f"({len(empty_stats_tpids)} empty-stats, {len(root_skipped_tpids)} root-skipped): "
            f"{fallback_tpids}"
        )

        for tpid in fallback_tpids:
            sub = data[data[TEMPORAL_PROPERTY_ID] == tpid]
            if sub.empty:
                logger.warning(f"TD4C fallback skipped for TPID {tpid}: no data rows.")
                continue
            td4c = TD4C(
                bins=self.bins,
                per_variable=True,
                distance_measure=self.fallback_td4c_distance,
            )
            td4c.entity_class = self.entity_class
            print(td4c.entity_class)
            try:
                td4c.fit(sub)
            except Exception as exc:
                logger.warning(f"TD4C fallback failed for TPID {tpid}: {exc}")
                continue
            new_cutoffs = td4c.boundaries.get(tpid) if td4c.boundaries else None
            if new_cutoffs is None:
                logger.warning(f"TD4C fallback produced no cutoffs for TPID {tpid}.")
                continue
            self.boundaries[tpid] = list(new_cutoffs)
            self.fallback_variables[tpid] = "td4c"
            self.skipped_variables.discard(tpid)
            logger.info(f"  TPID {tpid}: TD4C cutoffs = {list(new_cutoffs)}")

    def save_scores_summary(self, output_dir: str):
        """Save a short CSV with per-variable final t-stat scores and a grand total row."""
        if not self.final_scores_per_tpid:
            return
        rows = []
        for tpid in sorted(self.final_scores_per_tpid.keys()):
            cutoffs = self.final_cutoffs_per_tpid.get(tpid, [])
            rows.append({
                'TemporalPropertyID': tpid,
                'NumCutoffs': len(cutoffs),
                'CutoffValues': str([round(c, 4) for c in cutoffs]),
                'FinalTStatScore': round(self.final_scores_per_tpid[tpid], 6),
            })
        total = sum(self.final_scores_per_tpid.values())
        rows.append({
            'TemporalPropertyID': 'TOTAL',
            'NumCutoffs': '-',
            'CutoffValues': '-',
            'FinalTStatScore': round(total, 6),
        })
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, 'tid3_scores_summary.csv')
        df.to_csv(path, index=False)
        logger.info(f"TID3 scores summary saved to: {path}")

    def _is_state_selected(self, temporal_property_id, assigned_state: int) -> bool:
        """
        Return whether a transformed 1-based state passes p-value selection.

        per_state_stats is keyed by the 0-based states used by searchsorted during
        cutoff search, while transform() assigns 1-based states via assign_state().

        Variables marked "td4c" in ``self.fallback_variables`` were discretized by the
        TD4C fallback, so they have no per-state t/p stats and the p-value gate does
        not apply: those states are always kept. Variables marked "tid3_two_sided"
        DO have per-state stats (from the two-sided re-fit) and are gated normally.
        """
        if self.state_selection_p_threshold is None:
            return True

        if self.fallback_variables.get(temporal_property_id) == "td4c":
            return True

        internal_state = int(assigned_state) - 1
        state_stats = self.per_state_stats.get(temporal_property_id, {}).get(internal_state)
        if state_stats is None:
            return False

        p_value = state_stats.get("p")
        if p_value is None or pd.isna(p_value):
            return False

        return float(p_value) <= self.state_selection_p_threshold

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the learned TID3 cutpoints.
        For each sample, assign a state via the common assign_state() helper.
        Root-abandoned variables (min_mean_gap-infeasible or zero-score at root,
        not rescued by the two-sided re-fit or the TD4C fallback) have their
        rows dropped from the output entirely.
        """
        data = data.copy()
        # Drop rows for variables that were root-skipped during fit().
        if self.skipped_variables:
            data = data[~data[TEMPORAL_PROPERTY_ID].isin(self.skipped_variables)]
        def _assign(row):
            assigned_state = assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], []))
            if not self._is_state_selected(row[TEMPORAL_PROPERTY_ID], assigned_state):
                return -1
            return assigned_state
        data["state"] = data.apply(_assign, axis=1)

        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """Return the computed TID3 boundaries."""
        return self.boundaries

    @classmethod
    def get_available_scoring_methods(cls):
        """
        Get a dictionary of available scoring methods and their descriptions.
        
        Returns:
            dict: Dictionary mapping scoring method names to their descriptions.
        """
        return cls.AVAILABLE_SCORING_METHODS.copy()
    
    @classmethod
    def list_scoring_methods(cls):
        """
        Print available scoring methods and their descriptions.
        """
        print("Available TID3 Scoring Methods:")
        print("=" * 50)
        for method, description in cls.AVAILABLE_SCORING_METHODS.items():
            print(f"• {method}: {description}")
        print("=" * 50)
    
    def get_scoring_method_info(self):
        """
        Get information about the currently selected scoring method.
        
        Returns:
            dict: Information about the current scoring method.
        """
        return {
            "current_method": self.scoring_method,
            "description": self.AVAILABLE_SCORING_METHODS[self.scoring_method],
            "all_available": self.get_available_scoring_methods()
        }

def tid3(data: pd.DataFrame, bins: int,
         min_duration_threshold: int = 2, max_gap: int = 1,
         nb_candidates: int = 100, duration_preference: str = "two_sided",
         multivariate_refinement: bool = False,
         num_relations: int = 3, mv_top_tirps: int = 100,
         min_mean_gap: float = 0.0, state_selection_p_threshold: Optional[float] = None,
         fallback_method: Optional[str] = "td4c", fallback_td4c_distance: str = "cosine",
         mv_variable_order: str = "univariate_score_desc", mv_random_seed: Optional[int] = None):
    """
    Convenience wrapper to run TID3 on a dataset.

    TID3 selects cutoffs by maximizing the sum of per-state Welch's
    t-statistics computed over symbolic time interval (STI) time-duration
    distributions between case and control populations.

    Parameters:
      data: input DataFrame.
      bins: number of bins desired.
      min_duration_threshold: minimum consecutive state occurrences considered
          for time-duration analysis.
      max_gap: maximum gap between timestamps treated as consecutive states.
      nb_candidates: number of candidate cutpoints to evaluate.
      duration_preference: direction preference for time-duration comparison.
          - "two_sided": favor any difference (two-tailed test, default)
          - "class1_longer": favor class 1 having longer time durations (one-tailed)
          - "class0_longer": favor class 0 having longer time durations (one-tailed)
      multivariate_refinement: if True, run Phase 2 conditional intra-variable
          greedy cross-variable refinement after univariate cutoff selection.
      num_relations: number of Allen temporal relations for size-2 TIRP
          construction (3 or 7).
      mv_top_tirps: Phase 2 normalization. Sum the top-K |t| values and
          divide by K. Set to -1 (or 0) to divide by the actual scorable count.
      min_mean_gap: minimum absolute difference between pooled mean durations
          of the two child bins produced by a candidate cutoff (Phase 1 greedy
          only; closed inequality). Default 0.0 preserves original behaviour.
      state_selection_p_threshold: optional raw p-value threshold for
          per-variable state selection. If set, transformed states whose
          per-state p-value is above the threshold, or absent from
          per_state_stats, are mapped to -1.
      fallback_method: optional fallback discretization for variables TID3
          cannot meaningfully handle (empty per-state stats, or root-skipped
          under min_mean_gap). Currently only "td4c" is supported. Set to
          None to disable. Default "td4c".
      fallback_td4c_distance: distance measure for the TD4C fallback, one of
          {"cosine", "kullback_leibler", "entropy"}. Ignored when
          fallback_method is None. Default "cosine".
      mv_variable_order: ordering of variables visited by Phase 2 MV refinement.
          One of {"univariate_score_desc", "univariate_score_asc", "random"}.
      mv_random_seed: seed for numpy.random.default_rng when
          mv_variable_order == "random". Default None.

    Returns:
      symbolic_series: transformed DataFrame with a "state" column.
      states: cutpoints (boundaries) per variable.
    """
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]
    method_instance = TID3(
        bins,
        min_duration_threshold=min_duration_threshold,
        max_gap=max_gap,
        scoring_method="max_t_stat_sum",
        nb_candidates=nb_candidates,
        duration_preference=duration_preference,
        multivariate_refinement=multivariate_refinement,
        num_relations=num_relations,
        mv_top_tirps=mv_top_tirps,
        min_mean_gap=min_mean_gap,
        state_selection_p_threshold=state_selection_p_threshold,
        fallback_method=fallback_method,
        fallback_td4c_distance=fallback_td4c_distance,
        mv_variable_order=mv_variable_order,
        mv_random_seed=mv_random_seed,
    )
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states

# if __name__ == "__main__":
#     # Define the list of dataset paths you want to process
#     dataset_paths = [
#         "/sise/robertmo-group/Eldar/CPM_dataset/aki/stg1_progression_minAGE18_minTIME6_fromStage1_enhanced_metavision_carevue_newcohort/final_formatted_data.csv",
#         # "/sise/robertmo-group/Eldar/CPM_dataset/diabetes/diabetes_with_outcome.csv",
#         # "/sise/robertmo-group/Eldar/CPM_dataset/falls_small/small_falls_classes_paa_7.csv",
#         # "/sise/robertmo-group/Eldar/CPM_dataset/icu/icu_with_outcome_paa_10.csv",
#         # "/sise/robertmo-group/Eldar/CPM_dataset/ahe_small/small_ahe_paa_15.csv"
#     ]

#     min_mean_gap_list = [0.0]  # Example values to test


#     # List to store the row dictionaries for the final CSV
#     all_results = []

#     for data_path in dataset_paths:
#         for min_mean_gap in min_mean_gap_list:
#             print(f"\nRunning TID3 with min_mean_gap={min_mean_gap} on dataset: {data_path}")
#             dataset_name = os.path.basename(data_path)
#             print(f"\n{'='*80}")
#             print(f"Processing Dataset: {dataset_name}")
#             print(f"{'='*80}")

#             if not os.path.exists(data_path):
#                 print(f"Path does not exist: {data_path}. Skipping.")
#                 continue

#             # Load and prep data
#             df = pd.read_csv(data_path, low_memory=False)
            
#             # Extract entity_class from special rows where TemporalPropertyID == -1
#             class_rows = df[df[TEMPORAL_PROPERTY_ID] == -1]
#             entity_class = {
#                 int(row[ENTITY_ID]): int(float(row[VALUE]))
#                 for _, row in class_rows.iterrows()
#             }
#             data = df[df[TEMPORAL_PROPERTY_ID] != -1].copy()

#             tid3_model = TID3(
#                 bins=3,
#                 min_duration_threshold=1,
#                 max_gap=1,
#                 scoring_method="max_t_stat_sum",
#                 nb_candidates=100,
#                 duration_preference="two_sided",
#                 multivariate_refinement=False,
#                 min_mean_gap=min_mean_gap,
#             )
            
#             tid3_model.entity_class = entity_class
            
#             # We only need fit() to get the boundaries, saving compute time over fit_transform()
#             tid3_model.fit(data)

#             print(tid3_model.per_state_stats)

#             # Extract the boundaries and format them for the CSV
#             # for tpid, cutoffs in tid3_model.boundaries.items():
#             #     # Format the list exactly as requested: [val1, val2]
#             #     cutoffs_formatted = f"[{', '.join(map(str, cutoffs))}]"
                
#             #     all_results.append({
#             #         "Dataset Name": dataset_name,
#             #         "Temporal Variable ID": tpid,
#             #         "Min Mean Gap": min_mean_gap,
#             #         "Cutoffs": cutoffs_formatted
#             #     })

#     # Export to a single CSV file after processing all datasets
#     if all_results:
#         results_df = pd.DataFrame(all_results)
#         results_df.to_csv("tid3_cutoffs_summary.csv", index=False)
#     else:
#         print("\nNo results were generated. Please check your dataset paths.")

# if __name__ == "__main__":
#     # Debug entry point: instantiate TID3 directly so breakpoints inside fit()
#     # are hit without going through the TemporalAbstraction wrapper.

#     DATA_PATH = "/sise/robertmo-group/Eldar/CPM_dataset/diabetes/diabetes_with_outcome.csv"

#     df = pd.read_csv(DATA_PATH, low_memory=False)

#     # Extract entity_class from special rows where TemporalPropertyID == -1.
#     class_rows = df[df[TEMPORAL_PROPERTY_ID] == -1]
#     entity_class = {
#         int(row[ENTITY_ID]): int(float(row[VALUE]))
#         for _, row in class_rows.iterrows()
#     }
#     data = df[df[TEMPORAL_PROPERTY_ID] != -1].copy()

#     # ------------------------------------------------------------------
#     # Option 1: TID3 WITHOUT multivariate refinement (Phase 1 only)
#     # ------------------------------------------------------------------
#     tid3_no_mv = TID3(
#         bins=3,
#         min_duration_threshold=2,
#         max_gap=1,
#         scoring_method="max_t_stat_sum",
#         nb_candidates=100,
#         duration_preference="two_sided",
#         multivariate_refinement=False,
#         num_relations=3,
#         mv_top_tirps=100,
#         min_mean_gap=0.0,
#     )
#     tid3_no_mv.entity_class = entity_class
#     result_no_mv = tid3_no_mv.fit_transform(data)
#     print("=== TID3 without multivariate refinement ===")
#     print("Boundaries:", tid3_no_mv.boundaries)
# if __name__ == "__main__":
#     DATA_PATH = "/sise/robertmo-group/Eldar/CPM_dataset/diabetes/diabetes_with_outcome.csv"

#     df = pd.read_csv(DATA_PATH, low_memory=False)

#     # Extract entity_class from special rows where TemporalPropertyID == -1.
#     class_rows = df[df[TEMPORAL_PROPERTY_ID] == -1]
#     entity_class = {
#         int(row[ENTITY_ID]): int(float(row[VALUE]))
#         for _, row in class_rows.iterrows()
#     }
#     data = df[df[TEMPORAL_PROPERTY_ID] != -1].copy()
#     # ------------------------------------------------------------------
#     # Option 2: TID3 WITH multivariate refinement (Phase 1 + Phase 2)
#     # ------------------------------------------------------------------
#     tid3_mv = TID3(
#         bins=3,
#         min_duration_threshold=2,
#         max_gap=1,
#         scoring_method="max_t_stat_sum",
#         nb_candidates=5,
#         duration_preference="two_sided",
#         multivariate_refinement=True,
#         num_relations=3,
#         mv_top_tirps=100,
#         min_mean_gap=0.0,
#     )
#     tid3_mv.entity_class = entity_class
#     result_mv = tid3_mv.fit_transform(data)
#     print("=== TID3 with multivariate refinement ===")
#     print("Boundaries:", tid3_mv.boundaries)