# File: ta_package/methods/tid3.py
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, entropy, ks_2samp, mannwhitneyu, wasserstein_distance
import logging
import warnings
from tqdm import tqdm

# Suppress scipy warnings for nearly identical data (expected in constant variables)
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation due to catastrophic cancellation')
# warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
try:
    # Try relative imports first (when used as a module)
    from .base import TAMethod
    from ..utils import assign_state, candidate_selection, symmetric_kullback_leibler
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
    from ta_package.utils import assign_state, candidate_selection, symmetric_kullback_leibler
    from ta_package.constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TID3(TAMethod):
    """
    TIDE (Time Interval Duration Exploitation) - A discretization method that optimizes cutoffs 
    by exploiting significant differences in interval durations between classes.
    
    This method creates temporal "tides" by finding bin boundaries that maximize the 
    statistical significance of duration pattern differences between different classes.
    
    The class supports multiple scoring methods through a well-organized scoring system.
    Each scoring method implements a different optimization strategy for finding optimal cutoffs.
    
    ZERO-VARIANCE HANDLING:
    When a specific cutoff configuration produces identical durations (zero variance),
    the scoring functions gracefully skip that configuration (contribute 0 to score).
    The greedy algorithm will naturally prefer cutoff configurations that produce
    duration variance, allowing meaningful t-test comparisons.
    
    Note: Different cutoffs can produce different durations! A cutoff that creates
    frequent state changes might yield duration=1 everywhere, while a different cutoff
    that aligns with natural value clusters might yield longer, more meaningful durations.
    """
    
    # Available scoring methods - easily extensible
    AVAILABLE_SCORING_METHODS = {
        "avg_duration_diff_sum": "Minimize sum of absolute differences in average durations per state",
        "max_t_stat_sum": "Maximize sum of t-statistics for each state separately",
        "max_ks_stat_sum": "Maximize sum of Kolmogorov-Smirnov statistics (non-parametric CDF comparison)",
        "max_kl_divergence_sum": "Maximize sum of symmetric KL divergence between duration distributions",
        "max_mannwhitney_sum": "Maximize sum of Mann-Whitney U test statistics (non-parametric rank-based)",
        "max_wasserstein_sum": "Maximize sum of Wasserstein (Earth Mover's) distances between duration distributions",
        "self_transition_diff": "Maximize sum of self-transition probability differences between classes (no STI creation needed)"
    }
    
    # Available duration preference options
    AVAILABLE_DURATION_PREFERENCES = {
        "two_sided": "Favor any difference in duration distributions between classes (default)",
        "class1_longer": "Favor cutoffs where class 1 has longer durations than class 0",
        "class0_longer": "Favor cutoffs where class 0 has longer durations than class 1"
    }
    
    def __init__(self, bins: int, per_variable: bool = True, min_duration_threshold: int = 2, max_gap: int = 1, extended_output: bool = False, output_path: str = "tid3_extended_output.csv", scoring_method: str = "max_t_stat_sum", nb_candidates: int = 100, duration_preference: str = "two_sided", significance_threshold: float = None, state_selection: bool = False):
        """
        Initialize TIDE with specified parameters.

        Parameters:
            bins (int): Desired number of bins (discretization intervals).
            per_variable (bool): If True, each TemporalPropertyID is fitted independently.
            min_duration_threshold (int): Minimum duration length to consider for analysis.
                                        States with fewer consecutive occurrences are filtered out.
                                        If None, defaults to 0 (no filtering).
            max_gap (int): Maximum gap between timestamps to consider as consecutive states.
                          Same logic as used in KL interval generation.
            extended_output (bool): If True, generates detailed CSV report with statistics.
            output_path (str): Path for the extended output CSV file.
            scoring_method (str): Scoring method to use. Available methods:
                                - "max_t_stat_sum": Maximize sum of t-statistics for each state separately (default)
                                - "avg_duration_diff_sum": Minimize sum of absolute differences in average durations
                                - "max_ks_stat_sum": Maximize sum of Kolmogorov-Smirnov statistics (non-parametric)
                                - "max_kl_divergence_sum": Maximize sum of symmetric KL divergence
                                - "max_mannwhitney_sum": Maximize sum of Mann-Whitney U test significance
                                - "max_wasserstein_sum": Maximize sum of Wasserstein (Earth Mover's) distances
                                Use TID3.AVAILABLE_SCORING_METHODS to see all options.
            nb_candidates (int): Number of candidate cutpoints to evaluate. Default is 100.
            duration_preference (str): Direction preference for duration comparison. Options:
                                - "two_sided": Favor any difference (two-tailed tests, default)
                                - "class1_longer": Favor class 1 having longer durations (one-tailed)
                                - "class0_longer": Favor class 0 having longer durations (one-tailed)
                                Use TID3.AVAILABLE_DURATION_PREFERENCES to see all options.
            significance_threshold (float): p-value threshold used when state_selection=True (default: None).
                                          Only relevant when state_selection=True.
            state_selection (bool): If True, after discretization each state is tested for significance
                                    (t-test between D0 and D1 duration distributions, p < significance_threshold).
                                    Non-significant states are assigned -1 in transform() instead of a state ID.
                                    Cutoff boundaries are kept unchanged. Requires significance_threshold to be set.
                                    Default: False.
        """
        self.bins = bins
        self.per_variable = per_variable
        self.min_duration_threshold = min_duration_threshold if min_duration_threshold is not None else 0
        self.max_gap = max_gap
        self.extended_output = extended_output
        self.output_path = output_path
        self.boundaries = None
        self.extended_stats = []  # Store detailed statistics for extended output
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
        # Set significance threshold and state selection
        self.significance_threshold = significance_threshold
        self.state_selection = state_selection
        self.significant_states = {}  # {tpid: set of significant 0-indexed state IDs}

    def _get_scoring_function(self):
        """
        Get the scoring function based on the selected scoring method.
        
        Returns:
            callable: The scoring function that takes (df, cutoffs, max_gap, temporal_property_id) 
                     and returns a numeric score.
        """
        if self.scoring_method == "avg_duration_diff_sum":
            return self._avg_duration_diff_sum_scoring
        elif self.scoring_method == "max_t_stat_sum":
            return self._max_t_stat_sum_scoring
        elif self.scoring_method == "max_ks_stat_sum":
            return self._max_ks_stat_sum_scoring
        elif self.scoring_method == "max_kl_divergence_sum":
            return self._max_kl_divergence_sum_scoring
        elif self.scoring_method == "max_mannwhitney_sum":
            return self._max_mannwhitney_sum_scoring
        elif self.scoring_method == "max_wasserstein_sum":
            return self._max_wasserstein_sum_scoring
        elif self.scoring_method == "self_transition_diff":
            return self._self_transition_diff_scoring
        else:
            raise ValueError(f"Scoring method '{self.scoring_method}' not implemented")
    
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
    
    def _check_directional_preference(self, mean_class1: float, mean_class0: float) -> bool:
        """
        Check if the directional preference is satisfied for non-hypothesis test methods.
        
        For methods like KL divergence and Wasserstein that don't have a built-in
        alternative parameter, this checks if the mean relationship satisfies
        the duration_preference.
        
        Parameters:
            mean_class1: Mean duration for class 1
            mean_class0: Mean duration for class 0
            
        Returns:
            bool: True if the preference is satisfied (or if two_sided), False otherwise
        """
        if self.duration_preference == "two_sided":
            return True  # Always include for two-sided
        elif self.duration_preference == "class1_longer":
            return mean_class1 > mean_class0
        elif self.duration_preference == "class0_longer":
            return mean_class0 > mean_class1
        else:
            return True

    # ======================================================================================
    # TD4C-STYLE FALLBACK SCORING (for zero-variance cases)
    # ======================================================================================
    
    def _td4c_fallback_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Fallback scoring function using TD4C-style distribution comparison.
        
        This is used when duration-based scoring fails due to zero variance (all durations identical).
        Instead of comparing durations, it compares how classes are distributed across value bins.
        
        Strategy: Maximize distance/divergence between class distributions using selected measure
        Higher distance = better separation between classes
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap (unused in this scoring, kept for compatibility)
            temporal_property_id: ID of temporal property for tracking
            
        Returns:
            float: Sum of distances/divergences between all class pairs
                   Higher score = better class separation
        """
        # Create bins using the cutoffs
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df_temp = df.copy()
        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))
        
        # Get classes
        classes = sorted(df_temp['Class'].unique())
        if len(classes) < 2:
            return 0.0
        
        nb_bins = len(bins_array) - 1
        
        # Compute class distributions over bins
        class_distribs = np.zeros((len(classes), nb_bins))
        for i, cls in enumerate(classes):
            sub = df_temp[df_temp['Class'] == cls]
            if sub.empty:
                continue
            counts = sub['Bin'].value_counts().sort_index()
            # Build a probability vector of length nb_bins
            v = np.zeros(nb_bins)
            for bin_id, count in counts.items():
                if bin_id < nb_bins:
                    v[int(bin_id)] = count
            if v.sum() > 0:
                class_distribs[i] = v / v.sum()
        
        # Calculate pairwise distances using symmetric KL divergence
        score = 0.0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                distance = symmetric_kullback_leibler(class_distribs[i], class_distribs[j])
                score += distance
        
        return score
    
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
        creating temporal "tides" using the same interval merging logic as KL generation.
        
        Parameters:
            df: DataFrame with columns [ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE, 'Class', 'Bin']
            max_gap: Maximum gap between timestamps to consider as consecutive
            
        Returns:
            dict: {class_id: [list of durations for each interval in the tide]}
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
                
                # Create intervals using the same logic as KL generation - building the tide
                intervals = []
                current_interval = None
                
                for _, row in prop_group.iterrows():
                    ts = row[TIMESTAMP]
                    state = row['Bin']
                    
                    if current_interval is None:
                        current_interval = {"start": ts, "end": ts + 1, "state": state}
                    else:
                        # If the same state and the gap is within max_gap, extend the tide interval
                        if state == current_interval["state"] and (ts - current_interval["end"]) <= max_gap:
                            current_interval["end"] = ts + 1
                        else:
                            # Save the current interval and start a new tide
                            intervals.append(current_interval)
                            current_interval = {"start": ts, "end": ts + 1, "state": state}
                
                # Don't forget the last interval in the tide
                if current_interval is not None:
                    intervals.append(current_interval)
                
                # Calculate durations from intervals - the "tide strength"
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
                
                # Create intervals using the same logic as KL generation - building the tide
                intervals = []
                current_interval = None
                
                for _, row in prop_group.iterrows():
                    ts = row[TIMESTAMP]
                    state = row['Bin']
                    
                    if current_interval is None:
                        current_interval = {"start": ts, "end": ts + 1, "state": state}
                    else:
                        # If the same state and the gap is within max_gap, extend the tide interval
                        if state == current_interval["state"] and (ts - current_interval["end"]) <= max_gap:
                            current_interval["end"] = ts + 1
                        else:
                            # Save the current interval and start a new tide
                            intervals.append(current_interval)
                            current_interval = {"start": ts, "end": ts + 1, "state": state}
                
                # Don't forget the last interval in the tide
                if current_interval is not None:
                    intervals.append(current_interval)
                
                # Calculate durations from intervals - the "tide strength"
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
    
    def _avg_duration_diff_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that evaluates cutoffs by minimizing the sum of absolute differences
        in average durations between classes for each state.
        
        This method separates durations by state and compares average durations between classes
        for each state separately, then sums the absolute differences.
        
        Respects duration_preference parameter:
        - "two_sided": Includes all absolute differences (any difference counts)
        - "class1_longer": Only includes diff when mean(class1) > mean(class0), uses signed diff
        - "class0_longer": Only includes diff when mean(class0) > mean(class1), uses signed diff
        
        Tie-breaking: When absolute differences are equal, prefers configurations with 
        lower standard deviations (more consistent duration patterns).
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Composite score combining differences (primary) and std sum (tie-breaker)
                   Lower score = better (classes have more similar and consistent duration patterns)
        """
        # Create bins using the cutoffs
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df_temp = df.copy()
        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))
        
        # Calculate durations by class and state
        durations_by_class_and_state = self._calculate_interval_durations_by_state_from_df(df_temp, max_gap)
        
        # If we don't have at least 2 classes, return a high score (bad)
        classes_with_data = list(durations_by_class_and_state.keys())
        if len(classes_with_data) < 2:
            return float('inf')  # High score = bad
        
        # For directional preference, we need class 0 and class 1
        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in classes_with_data or 1 not in classes_with_data):
            return float('inf')
        
        # Calculate sum of differences in average durations across all states
        total_diff_sum = 0.0
        total_std_sum = 0.0  # For tie-breaking: lower std is better
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate difference in average durations
        for state in states:
            state_avg_durations = {}
            
            # Calculate average and std duration for each class in this state
            for class_id in classes_with_data:
                if state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_avg_durations[class_id] = np.mean(durations)
                        total_std_sum += np.std(durations)
                    else:
                        state_avg_durations[class_id] = 0.0
                else:
                    state_avg_durations[class_id] = 0.0
            
            if is_directional:
                # Directional: only compare class 1 vs class 0
                mean_class1 = state_avg_durations.get(1, 0.0)
                mean_class0 = state_avg_durations.get(0, 0.0)
                
                if self._check_directional_preference(mean_class1, mean_class0):
                    # Use signed difference in the expected direction
                    if self.duration_preference == "class1_longer":
                        total_diff_sum += (mean_class1 - mean_class0)
                    else:  # class0_longer
                        total_diff_sum += (mean_class0 - mean_class1)
            else:
                # Two-sided: compare all pairs of classes with absolute differences
                class_ids = list(state_avg_durations.keys())
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        abs_diff = abs(state_avg_durations[class1] - state_avg_durations[class2])
                        total_diff_sum += abs_diff
        
        # Return a composite score
        # For two_sided: lower is better (minimization)
        # For directional: higher diff in preferred direction is better, so we negate
        tie_breaker_weight = 0.001
        if is_directional:
            # For directional, we want to maximize the difference in the preferred direction
            # So we negate to convert to a maximization-compatible score
            composite_score = -total_diff_sum + (tie_breaker_weight * total_std_sum)
        else:
            composite_score = total_diff_sum + (tie_breaker_weight * total_std_sum)
        
        return composite_score

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
    
    def _max_ks_stat_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that maximizes the sum of Kolmogorov-Smirnov statistics 
        for each state separately.
        
        The KS test is non-parametric and compares the cumulative distribution functions (CDFs)
        of two samples. The KS statistic ranges from 0 (identical distributions) to 1 
        (completely different distributions).
        
        Respects duration_preference parameter:
        - "two_sided": Uses two-sided KS test (any difference is good)
        - "class1_longer": Uses one-sided KS test (class1 CDF < class0 CDF, meaning class1 larger)
        - "class0_longer": Uses one-sided KS test (class0 CDF < class1 CDF, meaning class0 larger)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of KS statistics across all states
                   Higher score = better (more different duration distributions between classes)
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
            return 0.0
        
        # Calculate sum of KS statistics across all states
        total_ks_stat_sum = 0.0
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate KS statistics
        for state in states:
            state_durations_by_class = {}
            
            # Get durations for each class in this state
            for class_id in classes_with_data:
                if class_id in durations_by_class_and_state and state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_durations_by_class[class_id] = durations
            
            if is_directional:
                # Directional: only compare class 1 vs class 0
                durations_class1 = state_durations_by_class.get(1, [])
                durations_class0 = state_durations_by_class.get(0, [])
                
                if len(durations_class1) >= 1 and len(durations_class0) >= 1:
                    try:
                        # For KS test: alternative='greater' means CDF of first arg < CDF of second
                        # which means first arg tends to have larger values
                        ks_stat, p_value = ks_2samp(durations_class1, durations_class0, 
                                                    alternative=self._get_scipy_alternative())
                        total_ks_stat_sum += ks_stat
                    except Exception:
                        pass
            else:
                # Two-sided: compare all pairs of classes
                class_ids = list(state_durations_by_class.keys())
                
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        durations1 = state_durations_by_class[class1]
                        durations2 = state_durations_by_class[class2]
                        
                        if len(durations1) >= 1 and len(durations2) >= 1:
                            try:
                                ks_stat, p_value = ks_2samp(durations1, durations2)
                                total_ks_stat_sum += ks_stat
                            except Exception:
                                continue
        
        return total_ks_stat_sum

    # ======================================================================================
    # SCORING METHOD: max_kl_divergence_sum
    # ======================================================================================
    # Strategy: Maximize sum of symmetric KL divergence between class duration distributions
    # Rationale: KL divergence captures the full shape of probability distributions
    # Score interpretation: Higher is better (maximization problem)
    
    def _max_kl_divergence_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that maximizes the sum of symmetric Kullback-Leibler divergence 
        between duration distributions for each state.
        
        KL divergence measures how one probability distribution differs from another.
        We use symmetric KL (average of KL(P||Q) and KL(Q||P)) for a proper distance metric.
        
        Respects duration_preference parameter:
        - "two_sided": Includes all KL divergences (any difference is good)
        - "class1_longer": Only includes divergence when mean(class1) > mean(class0)
        - "class0_longer": Only includes divergence when mean(class0) > mean(class1)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of symmetric KL divergences across all states
                   Higher score = better (more different duration distributions between classes)
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
            return 0.0
        
        # Calculate sum of KL divergences across all states
        total_kl_sum = 0.0
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate KL divergence
        for state in states:
            state_durations_by_class = {}
            
            # Get durations for each class in this state
            for class_id in classes_with_data:
                if class_id in durations_by_class_and_state and state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_durations_by_class[class_id] = durations
            
            if is_directional:
                # Directional: only compare class 1 vs class 0
                durations_class1 = state_durations_by_class.get(1, [])
                durations_class0 = state_durations_by_class.get(0, [])
                
                if len(durations_class1) >= 2 and len(durations_class0) >= 2:
                    mean_class1 = np.mean(durations_class1)
                    mean_class0 = np.mean(durations_class0)
                    
                    # Only include if directional preference is satisfied
                    if self._check_directional_preference(mean_class1, mean_class0):
                        try:
                            all_durations = durations_class1 + durations_class0
                            min_dur, max_dur = min(all_durations), max(all_durations)
                            
                            if max_dur == min_dur:
                                continue
                            
                            n_bins = min(10, max(2, int(np.sqrt(len(all_durations)))))
                            bin_edges = np.linspace(min_dur, max_dur + 1e-10, n_bins + 1)
                            
                            hist1, _ = np.histogram(durations_class1, bins=bin_edges)
                            hist0, _ = np.histogram(durations_class0, bins=bin_edges)
                            
                            epsilon = 1e-10
                            prob1 = (hist1 + epsilon) / (hist1.sum() + epsilon * len(hist1))
                            prob0 = (hist0 + epsilon) / (hist0.sum() + epsilon * len(hist0))
                            
                            kl_divergence = symmetric_kullback_leibler(prob1, prob0)
                            
                            if not np.isnan(kl_divergence) and not np.isinf(kl_divergence):
                                total_kl_sum += kl_divergence
                        except Exception:
                            pass
            else:
                # Two-sided: compare all pairs of classes
                class_ids = list(state_durations_by_class.keys())
                
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        durations1 = state_durations_by_class[class1]
                        durations2 = state_durations_by_class[class2]
                        
                        if len(durations1) >= 2 and len(durations2) >= 2:
                            try:
                                all_durations = durations1 + durations2
                                min_dur, max_dur = min(all_durations), max(all_durations)
                                
                                if max_dur == min_dur:
                                    continue
                                
                                n_bins = min(10, max(2, int(np.sqrt(len(all_durations)))))
                                bin_edges = np.linspace(min_dur, max_dur + 1e-10, n_bins + 1)
                                
                                hist1, _ = np.histogram(durations1, bins=bin_edges)
                                hist2, _ = np.histogram(durations2, bins=bin_edges)
                                
                                epsilon = 1e-10
                                prob1 = (hist1 + epsilon) / (hist1.sum() + epsilon * len(hist1))
                                prob2 = (hist2 + epsilon) / (hist2.sum() + epsilon * len(hist2))
                                
                                kl_divergence = symmetric_kullback_leibler(prob1, prob2)
                                
                                if not np.isnan(kl_divergence) and not np.isinf(kl_divergence):
                                    total_kl_sum += kl_divergence
                            except Exception:
                                continue
        
        return total_kl_sum

    # ======================================================================================
    # SCORING METHOD: max_mannwhitney_sum
    # ======================================================================================
    # Strategy: Maximize sum of Mann-Whitney U test statistics between class duration distributions
    # Rationale: Non-parametric rank-based test, robust to outliers and non-normality
    # Score interpretation: Higher is better (maximization problem)
    
    def _max_mannwhitney_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that maximizes the sum of Mann-Whitney U test statistics 
        for each state separately.
        
        The Mann-Whitney U test is a non-parametric test that compares whether the 
        distribution of one group tends to have larger values than the other.
        We use -log(p_value) as the score to capture significance.
        
        Respects duration_preference parameter:
        - "two_sided": Uses two-sided Mann-Whitney test
        - "class1_longer": Uses one-sided test (class1 > class0)
        - "class0_longer": Uses one-sided test (class0 > class1)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of -log(p_value) from Mann-Whitney tests across all states
                   Higher score = better (more significant differences between classes)
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
            return 0.0
        
        # Calculate sum of significance scores across all states
        total_significance_sum = 0.0
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate Mann-Whitney statistics
        for state in states:
            state_durations_by_class = {}
            
            # Get durations for each class in this state
            for class_id in classes_with_data:
                if class_id in durations_by_class_and_state and state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_durations_by_class[class_id] = durations
            
            if is_directional:
                # Directional: only compare class 1 vs class 0
                durations_class1 = state_durations_by_class.get(1, [])
                durations_class0 = state_durations_by_class.get(0, [])
                
                if len(durations_class1) >= 1 and len(durations_class0) >= 1:
                    try:
                        u_stat, p_value = mannwhitneyu(durations_class1, durations_class0, 
                                                       alternative=self._get_scipy_alternative())
                        if p_value > 0:
                            significance_score = -np.log(p_value)
                        else:
                            significance_score = 100
                        total_significance_sum += significance_score
                    except Exception:
                        pass
            else:
                # Two-sided: compare all pairs of classes
                class_ids = list(state_durations_by_class.keys())
                
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        durations1 = state_durations_by_class[class1]
                        durations2 = state_durations_by_class[class2]
                        
                        if len(durations1) >= 1 and len(durations2) >= 1:
                            try:
                                u_stat, p_value = mannwhitneyu(durations1, durations2, alternative='two-sided')
                                
                                if p_value > 0:
                                    significance_score = -np.log(p_value)
                                else:
                                    significance_score = 100
                                
                                total_significance_sum += significance_score
                                
                            except Exception:
                                # If Mann-Whitney test fails, skip this comparison
                                continue
        
        return total_significance_sum

    # ======================================================================================
    # SCORING METHOD: max_wasserstein_sum
    # ======================================================================================
    # Strategy: Maximize sum of Wasserstein (Earth Mover's) distances between class duration distributions
    # Rationale: Wasserstein distance measures the "work" needed to transform one distribution into another
    # Score interpretation: Higher is better (maximization problem)
    
    def _max_wasserstein_sum_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that maximizes the sum of Wasserstein distances 
        (Earth Mover's Distance) between duration distributions for each state.
        
        The Wasserstein distance measures the minimum "work" required to transform
        one distribution into another. It's a true metric and has nice mathematical properties.
        
        Respects duration_preference parameter:
        - "two_sided": Includes all Wasserstein distances (any difference is good)
        - "class1_longer": Only includes distance when mean(class1) > mean(class0)
        - "class0_longer": Only includes distance when mean(class0) > mean(class1)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap between timestamps to consider as consecutive
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of Wasserstein distances across all states
                   Higher score = better (more different duration distributions between classes)
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
            return 0.0
        
        # Calculate sum of Wasserstein distances across all states
        total_wasserstein_sum = 0.0
        states = set()
        
        # Collect all states that exist across all classes
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # For each state, calculate Wasserstein distances
        for state in states:
            state_durations_by_class = {}
            
            # Get durations for each class in this state
            for class_id in classes_with_data:
                if class_id in durations_by_class_and_state and state in durations_by_class_and_state[class_id]:
                    durations = durations_by_class_and_state[class_id][state]
                    if len(durations) > 0:
                        state_durations_by_class[class_id] = durations
            
            if is_directional:
                # Directional: only compare class 1 vs class 0
                durations_class1 = state_durations_by_class.get(1, [])
                durations_class0 = state_durations_by_class.get(0, [])
                
                if len(durations_class1) >= 1 and len(durations_class0) >= 1:
                    mean_class1 = np.mean(durations_class1)
                    mean_class0 = np.mean(durations_class0)
                    
                    # Only include if directional preference is satisfied
                    if self._check_directional_preference(mean_class1, mean_class0):
                        try:
                            w_distance = wasserstein_distance(durations_class1, durations_class0)
                            total_wasserstein_sum += w_distance
                        except Exception:
                            pass
            else:
                # Two-sided: compare all pairs of classes
                class_ids = list(state_durations_by_class.keys())
                
                for i, class1 in enumerate(class_ids):
                    for class2 in class_ids[i+1:]:
                        durations1 = state_durations_by_class[class1]
                        durations2 = state_durations_by_class[class2]
                        
                        if len(durations1) >= 1 and len(durations2) >= 1:
                            try:
                                w_distance = wasserstein_distance(durations1, durations2)
                                total_wasserstein_sum += w_distance
                            except Exception:
                                continue
        
        return total_wasserstein_sum

    # ======================================================================================
    # SCORING METHOD: self_transition_diff
    # ======================================================================================
    # Strategy: Maximize sum of self-transition probability differences between classes.
    # Uses transition matrices built from consecutive observations (no STI/interval creation).
    # For each state x, compares P_D0(x|x) vs P_D1(x|x).
    # Higher self-transition = entity stays longer in that state = longer duration.
    # Score interpretation: Higher is better (maximization problem)
    #
    # Key advantage: No interval merging or duration computation needed — only counts
    # of consecutive state pairs. Significantly faster than duration-based methods.

    def _build_transition_counts(self, df: pd.DataFrame):
        """
        Build transition count matrices per class from consecutive observations.
        
        For each entity, walks through time-ordered observations and counts
        every consecutive pair (state_t, state_{t+1}). No gap merging or
        interval duration computation — just raw sequential transitions.
        
        Parameters:
            df: DataFrame with columns [ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, 'Class', 'Bin']
                Must already have 'Bin' column assigned.
            
        Returns:
            dict: {class_id: 2D numpy array of transition counts}
                  Shape of each array is (n_states, n_states) where n_states = max(Bin) + 1
        """
        n_states = int(df['Bin'].max()) + 1
        
        # Initialize count matrices per class
        classes = sorted(df['Class'].unique())
        counts = {cls: np.zeros((n_states, n_states), dtype=int) for cls in classes}
        
        # Group by entity, temporal property, and class, walk through time-ordered observations
        for (entity_id, tpid, class_id), group in df.groupby([ENTITY_ID, TEMPORAL_PROPERTY_ID, 'Class']):
            group = group.sort_values(TIMESTAMP)
            bins = group['Bin'].values
            
            # Count consecutive transitions
            for t in range(len(bins) - 1):
                # Skip NaN bins
                if np.isnan(bins[t]) or np.isnan(bins[t + 1]):
                    continue
                from_state = int(bins[t])
                to_state = int(bins[t + 1])
                counts[class_id][from_state, to_state] += 1
        
        return counts

    def _self_transition_diff_scoring(self, df: pd.DataFrame, cutoffs, max_gap: int = 1, temporal_property_id: int = None):
        """
        Scoring function that maximizes differences in self-transition probabilities
        between classes for each state.
        
        For each state x:
          - P_D0(x|x) = self-transition probability for class 0
          - P_D1(x|x) = self-transition probability for class 1
          - Higher P(x|x) implies longer expected duration in state x (= 1/(1-P(x|x)))
        
        No interval merging or STI creation is needed — only counts of consecutive
        state pairs from raw time-ordered observations.
        
        Respects duration_preference parameter:
        - "two_sided": Sum of |P_D0(x|x) - P_D1(x|x)| across all states
        - "class1_longer": Sum of (P_D1(x|x) - P_D0(x|x)) where P_D1(x|x) > P_D0(x|x)
        - "class0_longer": Sum of (P_D0(x|x) - P_D1(x|x)) where P_D0(x|x) > P_D1(x|x)
        
        Parameters:
            df: DataFrame with temporal data and class information
            cutoffs: List of proposed cutoff values
            max_gap: Maximum gap (unused — kept for API compatibility)
            temporal_property_id: ID of temporal property for extended output tracking
            
        Returns:
            float: Sum of self-transition probability differences across all states
                   Higher score = better (more different persistence patterns between classes)
        """
        # Create bins using the cutoffs
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df_temp = df.copy()
        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))
        
        # Drop rows where Bin is NaN (values outside range)
        df_temp = df_temp.dropna(subset=['Bin'])
        
        if df_temp.empty:
            return 0.0
        
        # Build transition count matrices per class
        transition_counts = self._build_transition_counts(df_temp)
        
        # Need at least 2 classes
        classes_with_data = list(transition_counts.keys())
        if len(classes_with_data) < 2:
            return 0.0
        
        n_states = int(df_temp['Bin'].max()) + 1
        
        # For directional preference, we need class 0 and class 1
        is_directional = self.duration_preference != "two_sided"
        if is_directional and (0 not in classes_with_data or 1 not in classes_with_data):
            return 0.0
        
        # Calculate self-transition probabilities per class per state
        self_trans_probs = {}
        for cls in classes_with_data:
            self_trans_probs[cls] = np.zeros(n_states)
            for state in range(n_states):
                row_total = transition_counts[cls][state, :].sum()
                if row_total > 0:
                    self_trans_probs[cls][state] = transition_counts[cls][state, state] / row_total
                else:
                    self_trans_probs[cls][state] = 0.0
        
        # Compute score based on duration_preference
        total_score = 0.0
        
        if is_directional:
            for state in range(n_states):
                p_d0 = self_trans_probs[0][state]
                p_d1 = self_trans_probs[1][state]
                
                if self.duration_preference == "class1_longer":
                    # Only include states where D1 stays longer (higher self-transition)
                    if p_d1 > p_d0:
                        total_score += (p_d1 - p_d0)
                elif self.duration_preference == "class0_longer":
                    # Only include states where D0 stays longer (higher self-transition)
                    if p_d0 > p_d1:
                        total_score += (p_d0 - p_d1)
        else:
            # Two-sided: compare all pairs of classes
            for i, class1 in enumerate(classes_with_data):
                for class2 in classes_with_data[i+1:]:
                    for state in range(n_states):
                        p_c1 = self_trans_probs[class1][state]
                        p_c2 = self_trans_probs[class2][state]
                        total_score += abs(p_c1 - p_c2)
        
        return total_score

    def _check_zero_variance_durations(self, df: pd.DataFrame) -> bool:
        """
        Check if all duration patterns have zero variance (all durations are identical).
        
        This is a quick check to determine if we should use TD4C fallback scoring.
        
        Parameters:
            df: DataFrame with temporal data
            
        Returns:
            bool: True if all durations are identical (zero variance), False otherwise
        """
        # Create temporary bins using quantiles for quick check
        try:
            bins_array = [-np.inf] + list(df[VALUE].quantile([0.33, 0.67]).values) + [np.inf]
            df_temp = df.copy()
            df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False, duplicates='drop'))
            
            # Calculate a few durations
            durations_by_class = self._calculate_interval_durations(df_temp, self.max_gap)
            
            # Check if all durations are identical
            all_durations = []
            for class_id, durations in durations_by_class.items():
                all_durations.extend(durations)
            
            if len(all_durations) == 0:
                return True  # No durations found, use fallback
            
            # Check variance
            variance = np.var(all_durations)
            return variance < 1e-10  # Essentially zero variance
            
        except Exception:
            # If check fails, assume we need fallback
            return True

    # ======================================================================================
    # CACHED CANDIDATE SELECTION
    # ======================================================================================
    # Optimization: When adding a new cutoff, only one existing state gets split.
    # We cache durations for unchanged states and only recalculate for the split state.
    
    def _cached_candidate_selection(self, df: pd.DataFrame, nb_bins: int, temporal_property_id: int = None):
        """
        Cached version of candidate selection that exploits the greedy algorithm structure.
        
        When a new cutoff is added, it splits exactly ONE existing state into two.
        All other states remain unchanged. This method caches durations for unchanged
        states and only recalculates durations for the split state.
        
        Parameters:
            df: DataFrame with temporal data and class information
            nb_bins: Number of bins to create (nb_bins - 1 cutoffs will be selected)
            temporal_property_id: ID of temporal property for logging
            
        Returns:
            tuple: (chosen_cutpoints, chosen_scores)
        """
        from ..utils import generate_candidate_cutpoints
        
        # Generate candidate cutpoints
        candidate_pool = generate_candidate_cutpoints(df, self.nb_candidates)
        print(f'Candidate pool: {candidate_pool}')
        chosen_cutpoints = []
        chosen_scores = []
        
        # Duration cache: {class_id: {state_id: [list of durations]}}
        # Same format as returned by _calculate_interval_durations_by_state_from_df
        durations_cache = {}
        
        # Cache the value boundaries for quick lookup of which state to split
        cached_bins = None
        
        for iteration in range(1, nb_bins):
            scores = [-np.inf] * len(candidate_pool)
            
            for i, candidate in enumerate(candidate_pool):
                # Skip if already chosen
                if len(chosen_cutpoints) > 0 and any(abs(candidate - c) < 1e-10 for c in chosen_cutpoints):
                    continue
                
                # Create suggested cutoffs
                suggested = sorted(chosen_cutpoints + [candidate])
                
                try:
                    if iteration == 1 or not durations_cache:
                        # First iteration or empty cache: calculate everything
                        bins_array = [-np.inf] + suggested + [np.inf]
                        df_temp = df.copy()
                        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))
                        durations = self._calculate_interval_durations_by_state_from_df(df_temp, self.max_gap)
                        scores[i] = self._score_from_durations_direct(durations)
                    else:
                        # Use cache: only recalculate split state
                        scores[i] = self._compute_cached_score(
                            df, cached_bins, candidate, durations_cache
                        )
                except Exception as e:
                    scores[i] = -np.inf
                    continue
            
            # Check if all scores are invalid
            if all(s == -np.inf for s in scores):
                logger.warning(f"Early termination at iteration {iteration}: All candidates produced invalid scores")
                break
            
            # Select best candidate
            best_idx = max(range(len(scores)), key=lambda x: scores[x])
            best_candidate = candidate_pool[best_idx]
            best_score = scores[best_idx]
            
            chosen_cutpoints.append(best_candidate)
            chosen_cutpoints.sort()
            chosen_scores.append(best_score)
            
            # Update cache with new state configuration
            cached_bins = [-np.inf] + chosen_cutpoints + [np.inf]
            df_temp = df.copy()
            df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=cached_bins, labels=False))
            durations_cache = self._calculate_interval_durations_by_state_from_df(df_temp, self.max_gap)
            
            # Remove chosen candidate from pool
            candidate_pool.pop(best_idx)
            candidate_pool = [c for c in candidate_pool if not any(abs(c - cp) < 1e-10 for cp in chosen_cutpoints)]
            
            if len(candidate_pool) == 0 and iteration < nb_bins - 1:
                logger.warning(f"Candidate pool exhausted at iteration {iteration}")
                break
        
        return chosen_cutpoints, chosen_scores
    
    def _compute_cached_score(self, df: pd.DataFrame, cached_bins: list, 
                               new_cutoff: float, durations_cache: dict) -> float:
        """
        Compute score using cached durations for unchanged states.
        Only recalculates durations for the state being split.
        
        Parameters:
            df: Original DataFrame
            cached_bins: Current bin edges [-inf, cutoff1, cutoff2, ..., +inf]
            new_cutoff: The new cutoff being evaluated
            durations_cache: Cached durations {class_id: {state_id: [durations]}}
            
        Returns:
            float: The computed score
        """
        # Find which state the new cutoff falls into (the state to split)
        split_state = None
        for state_id in range(len(cached_bins) - 1):
            if cached_bins[state_id] < new_cutoff < cached_bins[state_id + 1]:
                split_state = state_id
                break
        
        if split_state is None:
            return -np.inf
        
        # Create new bins with the additional cutoff
        new_cutoffs = sorted([b for b in cached_bins if b not in [-np.inf, np.inf]] + [new_cutoff])
        new_bins = [-np.inf] + new_cutoffs + [np.inf]
        
        # The split state becomes two new states: split_state and split_state + 1
        # States after split_state get shifted by +1
        
        # Filter df to only rows in the split state's value range
        # pd.cut uses left-exclusive, right-inclusive intervals: (low, high]
        # So we need to match that behavior
        split_low = cached_bins[split_state]
        split_high = cached_bins[split_state + 1]
        
        # Get rows that were in the split state using pd.cut's interval logic
        if split_state == 0:
            # First state: (-inf, high] means x <= high
            mask = df[VALUE] <= split_high
        elif split_high == np.inf:
            # Last state: (low, +inf) means x > low
            mask = df[VALUE] > split_low
        else:
            # Middle state: (low, high] means low < x <= high
            mask = (df[VALUE] > split_low) & (df[VALUE] <= split_high)
        
        df_split = df[mask].copy()
        
        # Assign new bins to split rows only
        df_split = df_split.assign(Bin=pd.cut(df_split[VALUE], bins=new_bins, labels=False))
        
        # Calculate durations for split state rows
        split_durations = self._calculate_interval_durations_by_state_from_df(df_split, self.max_gap)
        
        # Build combined durations: cached (with shifted indices) + new split durations
        combined_durations = {}
        
        for class_id in set(list(durations_cache.keys()) + list(split_durations.keys())):
            combined_durations[class_id] = {}
            
            # Copy unchanged states before split (same indices)
            if class_id in durations_cache:
                for state_id, durs in durations_cache[class_id].items():
                    if state_id < split_state:
                        combined_durations[class_id][state_id] = durs
            
            # Add split state durations (states split_state and split_state+1)
            if class_id in split_durations:
                for state_id, durs in split_durations[class_id].items():
                    if state_id in [split_state, split_state + 1]:
                        combined_durations[class_id][state_id] = durs
            
            # Copy unchanged states after split (indices shifted by +1)
            if class_id in durations_cache:
                for state_id, durs in durations_cache[class_id].items():
                    if state_id > split_state:
                        new_state_id = state_id + 1
                        combined_durations[class_id][new_state_id] = durs
        
        return self._score_from_durations_direct(combined_durations)
    
    def _score_from_durations_direct(self, durations_by_class_and_state: dict) -> float:
        """
        Compute score directly from durations using the current scoring method.
        
        Parameters:
            durations_by_class_and_state: {class_id: {state_id: [durations]}}
            
        Returns:
            float: The computed score
        """
        classes_with_data = list(durations_by_class_and_state.keys())
        if len(classes_with_data) < 2:
            return 0.0
        
        # Collect all states
        states = set()
        for class_id in classes_with_data:
            states.update(durations_by_class_and_state[class_id].keys())
        
        # Dispatch to appropriate scoring logic
        if self.scoring_method == "max_t_stat_sum":
            return self._score_t_stat_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "max_ks_stat_sum":
            return self._score_ks_stat_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "max_kl_divergence_sum":
            return self._score_kl_divergence_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "max_mannwhitney_sum":
            return self._score_mannwhitney_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "max_wasserstein_sum":
            return self._score_wasserstein_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "avg_duration_diff_sum":
            return self._score_avg_diff_sum(durations_by_class_and_state, classes_with_data, states)
        elif self.scoring_method == "self_transition_diff":
            # self_transition_diff does not use durations — this path should not be reached.
            # If it is, return 0 to avoid errors.
            return 0.0
        else:
            return 0.0
    
    def _score_t_stat_sum(self, durations, classes, states) -> float:
        """Score using sum of t-statistics. Respects duration_preference."""
        total = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        # For directional, we need class 0 and 1
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0
        
        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 2 and len(d0) >= 2:
                    if np.std(d1) == 0 and np.std(d0) == 0:
                        mean1, mean0 = np.mean(d1), np.mean(d0)
                        if self.duration_preference == "class1_longer" and mean1 > mean0:
                            total += (mean1 - mean0)
                        elif self.duration_preference == "class0_longer" and mean0 > mean1:
                            total += (mean0 - mean1)
                    else:
                        try:
                            t, _ = ttest_ind(d1, d0, alternative=self._get_scipy_alternative(), equal_var=False)
                            if self.duration_preference == "class1_longer" and t > 0:
                                total += t
                            elif self.duration_preference == "class0_longer" and t < 0:
                                total += abs(t)
                        except:
                            pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 2 and len(d2) >= 2:
                            if not (np.std(d1) == 0 and np.std(d2) == 0):
                                try:
                                    t, _ = ttest_ind(d1, d2, equal_var=False)
                                    total += abs(t)
                                except:
                                    pass
        return total
    
    def _score_ks_stat_sum(self, durations, classes, states) -> float:
        """Score using sum of KS statistics. Respects duration_preference."""
        total = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0
        
        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 1 and len(d0) >= 1:
                    try:
                        ks, _ = ks_2samp(d1, d0, alternative=self._get_scipy_alternative())
                        total += ks
                    except:
                        pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 1 and len(d2) >= 1:
                            try:
                                ks, _ = ks_2samp(d1, d2)
                                total += ks
                            except:
                                pass
        return total
    
    def _score_kl_divergence_sum(self, durations, classes, states) -> float:
        """Score using sum of KL divergences. Respects duration_preference."""
        total = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0
        
        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 2 and len(d0) >= 2:
                    mean1, mean0 = np.mean(d1), np.mean(d0)
                    if self._check_directional_preference(mean1, mean0):
                        try:
                            all_d = d1 + d0
                            if max(all_d) == min(all_d):
                                continue
                            n_bins = min(10, max(2, int(np.sqrt(len(all_d)))))
                            edges = np.linspace(min(all_d), max(all_d) + 1e-10, n_bins + 1)
                            h1, _ = np.histogram(d1, bins=edges)
                            h0, _ = np.histogram(d0, bins=edges)
                            eps = 1e-10
                            p1 = (h1 + eps) / (h1.sum() + eps * len(h1))
                            p0 = (h0 + eps) / (h0.sum() + eps * len(h0))
                            kl = symmetric_kullback_leibler(p1, p0)
                            if not np.isnan(kl) and not np.isinf(kl):
                                total += kl
                        except:
                            pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 2 and len(d2) >= 2:
                            try:
                                all_d = d1 + d2
                                if max(all_d) == min(all_d):
                                    continue
                                n_bins = min(10, max(2, int(np.sqrt(len(all_d)))))
                                edges = np.linspace(min(all_d), max(all_d) + 1e-10, n_bins + 1)
                                h1, _ = np.histogram(d1, bins=edges)
                                h2, _ = np.histogram(d2, bins=edges)
                                eps = 1e-10
                                p1 = (h1 + eps) / (h1.sum() + eps * len(h1))
                                p2 = (h2 + eps) / (h2.sum() + eps * len(h2))
                                kl = symmetric_kullback_leibler(p1, p2)
                                if not np.isnan(kl) and not np.isinf(kl):
                                    total += kl
                            except:
                                pass
        return total
    
    def _score_mannwhitney_sum(self, durations, classes, states) -> float:
        """Score using sum of Mann-Whitney significance. Respects duration_preference."""
        total = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0
        
        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 1 and len(d0) >= 1:
                    try:
                        _, p = mannwhitneyu(d1, d0, alternative=self._get_scipy_alternative())
                        total += -np.log(p) if p > 0 else 100
                    except:
                        pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 1 and len(d2) >= 1:
                            try:
                                _, p = mannwhitneyu(d1, d2, alternative='two-sided')
                                total += -np.log(p) if p > 0 else 100
                            except:
                                pass
        return total
    
    def _score_wasserstein_sum(self, durations, classes, states) -> float:
        """Score using sum of Wasserstein distances. Respects duration_preference."""
        total = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        if is_directional and (0 not in durations or 1 not in durations):
            return 0.0
        
        for state in states:
            if is_directional:
                d1 = durations.get(1, {}).get(state, [])
                d0 = durations.get(0, {}).get(state, [])
                if len(d1) >= 1 and len(d0) >= 1:
                    mean1, mean0 = np.mean(d1), np.mean(d0)
                    if self._check_directional_preference(mean1, mean0):
                        try:
                            total += wasserstein_distance(d1, d0)
                        except:
                            pass
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        d1 = durations.get(c1, {}).get(state, [])
                        d2 = durations.get(c2, {}).get(state, [])
                        if len(d1) >= 1 and len(d2) >= 1:
                            try:
                                total += wasserstein_distance(d1, d2)
                            except:
                                pass
        return total
    
    def _score_avg_diff_sum(self, durations, classes, states) -> float:
        """Score using avg duration differences. Respects duration_preference."""
        total_diff = 0.0
        total_std = 0.0
        is_directional = self.duration_preference != "two_sided"
        
        if is_directional and (0 not in durations or 1 not in durations):
            return float('-inf')
        
        for state in states:
            avgs = {}
            for c in classes:
                d = durations.get(c, {}).get(state, [])
                avgs[c] = np.mean(d) if d else 0.0
                if d:
                    total_std += np.std(d)
            
            if is_directional:
                mean1 = avgs.get(1, 0.0)
                mean0 = avgs.get(0, 0.0)
                if self._check_directional_preference(mean1, mean0):
                    if self.duration_preference == "class1_longer":
                        total_diff += (mean1 - mean0)
                    else:
                        total_diff += (mean0 - mean1)
            else:
                for i, c1 in enumerate(classes):
                    for c2 in classes[i+1:]:
                        total_diff += abs(avgs[c1] - avgs[c2])
        
        # For directional, higher diff in preferred direction is better (negate to maximize)
        # For two_sided, lower diff is better (also negate)
        if is_directional:
            return -(-total_diff + 0.001 * total_std)  # Double negate = maximize diff
        else:
            return -(total_diff + 0.001 * total_std)

    # ======================================================================================
    # STATE SELECTION
    # ======================================================================================

    def _compute_significant_states(self, cutoffs, df, temporal_property_id=None):
        """
        For each state created by the given cutoffs, test whether the duration difference
        between class 0 and class 1 is statistically significant (t-test, p < significance_threshold).

        Parameters:
            cutoffs (list): Cutoff values defining the bins
            df (pd.DataFrame): Data with 'Class' column
            temporal_property_id: Optional ID for logging

        Returns:
            set: 0-indexed state IDs that are statistically significant
        """
        if not cutoffs:
            return set()

        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df_temp = df.copy()
        df_temp = df_temp.assign(Bin=pd.cut(df_temp[VALUE], bins=bins_array, labels=False))

        durations_by_class_and_state = self._calculate_interval_durations_by_state_from_df(df_temp, self.max_gap)

        n_states = len(cutoffs) + 1
        significant = set()

        for state in range(n_states):
            d1 = durations_by_class_and_state.get(1, {}).get(state, [])
            d0 = durations_by_class_and_state.get(0, {}).get(state, [])

            if len(d1) >= 2 and len(d0) >= 2:
                if np.std(d1) == 0 and np.std(d0) == 0:
                    # Zero variance in both — significant only if means differ
                    if np.mean(d1) != np.mean(d0):
                        significant.add(state)
                else:
                    try:
                        _, p_val = ttest_ind(d1, d0, equal_var=False)
                        if p_val < self.significance_threshold:
                            significant.add(state)
                    except Exception:
                        pass
            elif len(d1) >= 2 or len(d0) >= 2:
                # Only one class has data — class-specific state, treat as significant
                significant.add(state)

        n_sig = len(significant)
        n_total = n_states
        logger.info(f"TemporalPropertyID {temporal_property_id}: {n_sig}/{n_total} states are significant (p < {self.significance_threshold})")

        return significant

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

    def _optimized_candidate_selection(self, df: pd.DataFrame, nb_bins: int, temporal_property_id: int = None):
        """
        Optimized candidate selection for t-stat scoring variants.

        Pre-sorts data once, pre-computes group boundaries and time-gap breaks,
        then evaluates candidates using vectorized numpy operations.

        Parameters:
            df: DataFrame with columns [ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE, 'Class']
            nb_bins: Number of bins to create (nb_bins - 1 cutoffs will be selected)
            temporal_property_id: ID of temporal property for logging

        Returns:
            tuple: (chosen_cutpoints, chosen_scores) — same format as candidate_selection()
        """
        from ..utils import generate_candidate_cutpoints

        # === PRE-PROCESSING (done once) ===

        # Sort data by (entity, class, tpid, timestamp)
        df_sorted = df.sort_values([ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID, TIMESTAMP])

        # Extract numpy arrays
        timestamps = df_sorted[TIMESTAMP].values.astype(np.float64)
        values = df_sorted[VALUE].values.astype(np.float64)
        classes_arr = df_sorted['Class'].values

        n_total = len(df_sorted)

        # Pre-compute group boundaries: each group is (entity, class, tpid)
        group_labels = df_sorted.groupby(
            [ENTITY_ID, 'Class', TEMPORAL_PROPERTY_ID], sort=False
        ).ngroup().values

        if n_total > 1:
            group_boundary_mask = np.diff(group_labels) != 0
            group_boundaries = np.where(group_boundary_mask)[0] + 1
            group_starts = np.concatenate([[0], group_boundaries])
            group_ends = np.concatenate([group_boundaries, [n_total]])
        else:
            group_starts = np.array([0])
            group_ends = np.array([n_total])

        num_groups = len(group_starts)
        group_classes = classes_arr[group_starts]

        # Pre-compute time-gap breaks within each group (independent of state assignment).
        # Uses the interpolation gap convention: end = ts + 1, so gap = ts_next - (ts_prev + 1).
        # Break when gap > max_gap, i.e., ts_next - ts_prev > max_gap + 1.
        time_gap_breaks = np.zeros(n_total, dtype=bool)
        for g in range(num_groups):
            s, e = group_starts[g], group_ends[g]
            if e - s > 1:
                time_gap_breaks[s + 1:e] = np.diff(timestamps[s:e]) > self.max_gap + 1

        # Handle NaN values in the data
        nan_mask = np.isnan(values)
        has_nans = np.any(nan_mask)

        # === CANDIDATE POOL GENERATION ===
        candidate_pool = generate_candidate_cutpoints(df, self.nb_candidates)
        chosen_cutpoints = []
        chosen_scores = []

        # Show progress only in debug mode
        show_progress = logger.isEnabledFor(logging.DEBUG)

        for iteration in range(1, nb_bins):
            scores = np.full(len(candidate_pool), -np.inf)

            pbar = tqdm(total=len(candidate_pool),
                       desc=f"    Evaluating candidates (bin {iteration}/{nb_bins-1})",
                       disable=not show_progress,
                       leave=False)

            for i, candidate in enumerate(candidate_pool):
                # Skip candidate if already chosen (or nearly)
                if len(chosen_cutpoints) > 0 and any(np.isclose(candidate, chosen_cutpoints)):
                    pbar.update(1)
                    continue

                # Create sorted cutoffs array for this candidate
                cutoffs = np.sort(np.append(chosen_cutpoints, candidate))

                # Assign states using searchsorted (matches pd.cut behavior):
                # np.searchsorted(cutoffs, v, side='left') returns bin index such that:
                #   v <= c1 → 0,  c1 < v <= c2 → 1,  ...,  v > c_{k-1} → k
                # This matches pd.cut(v, bins=[-inf, c1, ..., inf], labels=False).
                states = np.searchsorted(cutoffs, values, side='left')

                # Mark NaN values with a special state
                if has_nans:
                    states = states.astype(np.int64)
                    states[nan_mask] = -1

                # Compute durations by (class, state) using vectorized interval computation
                durations_by_class_and_state = {}

                for g in range(num_groups):
                    s, e = group_starts[g], group_ends[g]
                    class_id = group_classes[g]
                    n = e - s

                    if n == 0:
                        continue

                    if class_id not in durations_by_class_and_state:
                        durations_by_class_and_state[class_id] = {}

                    st_group = states[s:e]
                    ts_group = timestamps[s:e]

                    if n == 1:
                        # Single point: one interval with duration = (ts + 1) - ts = 1
                        dur = 1
                        state = int(st_group[0])
                        if dur >= self.min_duration_threshold and state != -1:
                            durations_by_class_and_state[class_id].setdefault(state, []).append(dur)
                        continue

                    # Vectorized interval break detection:
                    # Break between consecutive points i and i+1 when EITHER:
                    #   1. State changes: state[i] != state[i+1]  (e.g., s1→s2 or s2→s1)
                    #   2. Interpolation gap exceeded: ts[i+1] - (ts[i] + 1) > max_gap
                    #      equivalently: ts[i+1] - ts[i] > max_gap + 1
                    # This correctly handles cases like s1,s1,s2,s1 — the s2 interruption
                    # creates breaks on BOTH sides, producing separate s1 intervals.
                    state_changes = np.diff(st_group) != 0
                    breaks = state_changes | time_gap_breaks[s + 1:e]

                    # Compute interval boundaries from break positions
                    break_idx = np.where(breaks)[0]
                    start_indices = np.concatenate([[0], break_idx + 1])
                    end_indices = np.concatenate([break_idx, [n - 1]])

                    # Compute durations matching original code convention:
                    # Each interval: start_time = ts[first_point], end_time = ts[last_point] + 1
                    # duration = end_time - start_time = ts[last_point] + 1 - ts[first_point]
                    durations_arr = ts_group[end_indices] + 1 - ts_group[start_indices]
                    interval_states = st_group[start_indices]

                    # Filter by min_duration_threshold
                    valid = durations_arr >= self.min_duration_threshold
                    if has_nans:
                        valid = valid & (interval_states != -1)

                    durations_valid = durations_arr[valid]
                    states_valid = interval_states[valid]

                    # Accumulate durations by state
                    class_dict = durations_by_class_and_state[class_id]
                    for j in range(len(durations_valid)):
                        state = int(states_valid[j])
                        if state not in class_dict:
                            class_dict[state] = []
                        class_dict[state].append(int(durations_valid[j]))

                # Compute score using existing scoring infrastructure
                try:
                    scores[i] = self._score_from_durations_direct(durations_by_class_and_state)
                except Exception:
                    scores[i] = -np.inf

                pbar.update(1)

            pbar.close()

            # Check for valid candidates
            if len(scores) == 0:
                logger.warning(f"Early termination at iteration {iteration}: No candidate scores available")
                break
            elif np.all(np.isneginf(scores)):
                logger.warning(f"Early termination at iteration {iteration}: All {len(scores)} candidates produced invalid scores")
                break

            # Select best candidate
            best_idx = np.argmax(scores)
            best_candidate = candidate_pool[best_idx]
            best_score = scores[best_idx]

            chosen_cutpoints.append(best_candidate)
            chosen_cutpoints.sort()
            chosen_scores.append(best_score)

            # Remove chosen candidate and near-duplicates from pool
            candidate_pool.pop(best_idx)
            candidate_pool = [c for c in candidate_pool
                            if not any(np.isclose(c, chosen_cutpoints))]

            # Check if pool is exhausted
            if len(candidate_pool) == 0 and iteration < nb_bins - 1:
                logger.warning(f"Candidate pool exhausted at iteration {iteration}: achieved {len(chosen_cutpoints) + 1}/{nb_bins} bins")

        return chosen_cutpoints, chosen_scores

    def _generate_cutpoints(self, df: pd.DataFrame, temporal_property_id: int = None):
        """
        For a given DataFrame (corresponding to one variable), choose candidate cutpoints
        that maximize exploitation of tide duration patterns between classes.

        If all durations are identical (zero variance), automatically falls back to
        TD4C-style distribution-based scoring.
        """
        # Ensure class information exists; if not, create a default class (e.g., 0).
        # Get temporal property ID for tracking
        if temporal_property_id is None and len(df) > 0:
            temporal_property_id = df[TEMPORAL_PROPERTY_ID].iloc[0]

        # Handle case where all values are the same
        if df[VALUE].nunique() == 1:
            logger.warning(f"Insufficient variability for TemporalPropertyID {temporal_property_id}: only 1 unique value ({df[VALUE].iloc[0]})")
            candidates = [df[VALUE].min()] * (self.bins - 1)
            return candidates

        # Use optimized path for t-stat scoring variants
        if self.scoring_method == "max_t_stat_sum":
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

        # Log final bin analysis
        achieved_bins = len(candidates) + 1
        if achieved_bins < self.bins:
            logger.warning(f"TemporalPropertyID {temporal_property_id}: achieved {achieved_bins}/{self.bins} bins - missing {self.bins - achieved_bins} bins due to optimization constraints")

        # Collect extended statistics for the final optimal cutoffs
        if self.extended_output:
            self._collect_final_stats_for_output(candidates, df, temporal_property_id)

        return candidates

    def _collect_final_stats_for_output(self, candidates, df, temporal_property_id):
        """
        Collect statistics for the optimal cutoffs chosen for a temporal property.
        Creates one record per state for each temporal property with the new format.
        Adds LOW_CUTOFF/HIGH_CUTOFF for each state, along with duration statistics
        (AVG/STD/N per class) and t-statistics comparing duration patterns between classes.
        """
        # Build the full bins array
        bins_array = [-np.inf] + list(candidates) + [np.inf]
        n_states = len(bins_array) - 1

        # Work on a copy and assign bin labels 0..n_states-1
        df_with_bins = df.copy()
        df_with_bins = df_with_bins.assign(
            Bin=pd.cut(df_with_bins[VALUE], bins=bins_array, labels=False)
        )

        # Replace infinities with observed min/max for neatness in the CSV
        value_min = float(df_with_bins[VALUE].min())
        value_max = float(df_with_bins[VALUE].max())

        def _clean_bound(v):
            if v == -np.inf:
                return value_min
            if v == np.inf:
                return value_max
            return float(v)

        # Durations by class & state (used for AVG/STD/N per class)
        durations_by_class_and_state = self._calculate_interval_durations_by_state_from_df(
            df_with_bins, self.max_gap
        )

        # Collect all classes we've seen so we can emit zeros when a class/state is missing
        all_classes = set(durations_by_class_and_state.keys())

        # Ensure deterministic ordering of class IDs in the CSV (0,1,2,...)
        all_classes = sorted(all_classes)

        # Emit one row per state, even if the state had no durations at all
        for state_id in range(n_states):
            low_cut = _clean_bound(bins_array[state_id])
            high_cut = _clean_bound(bins_array[state_id + 1])

            state_record = {
                'TemporalPropertyID': temporal_property_id,
                'StateID': state_id,
                'LOW_CUTOFF': low_cut,
                'HIGH_CUTOFF': high_cut,
                # Clarify interval semantics used by pd.cut with labels=False and default right=True
                # Interval is (LOW_CUTOFF, HIGH_CUTOFF] for internal binning; we expose flags explicitly:
                'LOW_INCLUSIVE': False,
                'HIGH_INCLUSIVE': True,
            }

            # Add per-class stats (AVG/STD/N). If absent, fill zeros.
            for class_id in all_classes:
                durs = []
                if class_id in durations_by_class_and_state:
                    durs = durations_by_class_and_state[class_id].get(state_id, [])
                if durs:
                    state_record[f'AVG_CLS_{class_id}'] = float(np.mean(durs))
                    state_record[f'STD_CLS_{class_id}'] = float(np.std(durs))
                    state_record[f'N_CLS_{class_id}'] = int(len(durs))
                else:
                    state_record[f'AVG_CLS_{class_id}'] = 0.0
                    state_record[f'STD_CLS_{class_id}'] = 0.0
                    state_record[f'N_CLS_{class_id}'] = 0

            # Calculate t-statistics between all class pairs for this state
            if len(all_classes) >= 2:
                for i, class1 in enumerate(all_classes):
                    for class2 in all_classes[i+1:]:
                        durations1 = []
                        durations2 = []
                        
                        # Get durations for each class in this state
                        if class1 in durations_by_class_and_state:
                            durations1 = durations_by_class_and_state[class1].get(state_id, [])
                        if class2 in durations_by_class_and_state:
                            durations2 = durations_by_class_and_state[class2].get(state_id, [])
                        
                        # Calculate t-statistic if we have enough data
                        if len(durations1) >= 2 and len(durations2) >= 2:
                            # Check if both groups have zero variance (all values identical)
                            std1 = np.std(durations1)
                            std2 = np.std(durations2)
                            
                            if std1 == 0.0 and std2 == 0.0:
                                # Both groups have identical values - no variance to test
                                # Check if means are different
                                mean1 = np.mean(durations1)
                                mean2 = np.mean(durations2)
                                if mean1 == mean2:
                                    # Identical distributions - t-stat = 0, p-value = 1
                                    state_record[f'T_STAT_CLS_{class1}_VS_{class2}'] = 0.0
                                    state_record[f'P_VALUE_CLS_{class1}_VS_{class2}'] = 1.0
                                else:
                                    # Different means but zero variance - infinite t-statistic (undefined)
                                    state_record[f'T_STAT_CLS_{class1}_VS_{class2}'] = float('nan')
                                    state_record[f'P_VALUE_CLS_{class1}_VS_{class2}'] = float('nan')
                            else:
                                try:
                                    t_stat, p_value = ttest_ind(durations1, durations2, equal_var=False)
                                    state_record[f'T_STAT_CLS_{class1}_VS_{class2}'] = float(t_stat)
                                    state_record[f'P_VALUE_CLS_{class1}_VS_{class2}'] = float(p_value)
                                except Exception as e:
                                    # If t-test fails for other reasons, set to NaN
                                    logger.debug(f"T-test failed for TemporalPropertyID {temporal_property_id}, StateID {state_id}: {e}")
                                    state_record[f'T_STAT_CLS_{class1}_VS_{class2}'] = float('nan')
                                    state_record[f'P_VALUE_CLS_{class1}_VS_{class2}'] = float('nan')
                        else:
                            # Not enough data for t-test
                            state_record[f'T_STAT_CLS_{class1}_VS_{class2}'] = float('nan')
                            state_record[f'P_VALUE_CLS_{class1}_VS_{class2}'] = float('nan')

            self.extended_stats.append(state_record)


    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the TIDE model by generating cutpoints for each variable based on 
        exploitation of interval duration tide patterns.
        """
        
        if self.per_variable:
            boundaries = {}
            temporal_properties = list(data.groupby(TEMPORAL_PROPERTY_ID).groups.keys())
            
            logger.info(f"Processing {len(temporal_properties)} temporal properties sequentially")
            
            # Sequential processing
            all_extended_stats = []
            
            # Create progress bar for sequential processing
            pbar = tqdm(total=len(temporal_properties), 
                       desc="🌊 Processing temporal properties", 
                       unit="property",
                       ncols=120)
            
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                try:
                    # Clear extended stats for this temporal property to avoid accumulation
                    if self.extended_output:
                        self.extended_stats = []
                    
                    # If a mapping of entity classes exists, merge it in
                    if 'Class' not in group.columns and hasattr(self, 'entity_class') and self.entity_class:
                        group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))
                    else:
                        logger.info(f'No entity class mapping found')
                        group = group.assign(Class=0)
                    
                    # Process this temporal property directly
                    boundaries[tpid] = self._generate_cutpoints(group, tpid)

                    # State selection: mark non-significant states as -1 in transform
                    if self.state_selection and self.significance_threshold is not None:
                        self.significant_states[tpid] = self._compute_significant_states(
                            boundaries[tpid], group, tpid
                        )

                    # Collect extended stats if available
                    if self.extended_output and hasattr(self, 'extended_stats'):
                        all_extended_stats.extend(self.extended_stats)
                    
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
                    boundaries[tpid] = []
                    pbar.update(1)
            
            pbar.close()
            
            self.boundaries = boundaries
            self.extended_stats = all_extended_stats
            
        else:
            if 'Class' not in data.columns and hasattr(self, 'entity_class') and self.entity_class:
                data = data.assign(Class=data[ENTITY_ID].map(self.entity_class))
            else:
                data = data.assign(Class=0)
            self.boundaries = self._generate_cutpoints(data)

            # State selection: mark non-significant states as -1 in transform
            if self.state_selection and self.significance_threshold is not None:
                self.significant_states[None] = self._compute_significant_states(
                    self.boundaries, data, temporal_property_id=None
                )

        # Generate extended output CSV if requested
        if self.extended_output:
            logger.info("Generating extended TIDE output CSV...")
            self._generate_extended_output()
        

    def _generate_extended_output(self):
        """
        Generate detailed CSV report with TIDE statistics for each temporal property and state.
        
        The CSV will contain columns:
        - TemporalPropertyID: ID of the temporal property
        - StateID: ID of the state (bin)
        - LOW_CUTOFF: Lower boundary of the state interval
        - HIGH_CUTOFF: Upper boundary of the state interval
        - LOW_INCLUSIVE: Whether the lower boundary is inclusive (False for pd.cut default)
        - HIGH_INCLUSIVE: Whether the upper boundary is inclusive (True for pd.cut default)
        - AVG_CLS_X: Average duration for this state in class X
        - STD_CLS_X: Standard deviation of durations for this state in class X
        - N_CLS_X: Number of duration samples used to calculate avg and std for this state in class X
        - T_STAT_CLS_X_VS_Y: T-statistic comparing durations between class X and class Y for this state
        - P_VALUE_CLS_X_VS_Y: P-value from the t-test between class X and class Y for this state
        """
        if not self.extended_stats:
            logger.warning("No extended TIDE statistics collected. Extended output will be empty.")
            return
        
        # Convert extended_stats to DataFrame
        df_extended = pd.DataFrame(self.extended_stats)
        
        # Sort by temporal property ID and state ID for better readability
        df_extended = df_extended.sort_values(['TemporalPropertyID', 'StateID'])
        
        # Round numeric columns for better readability
        numeric_columns = df_extended.select_dtypes(include=[np.number]).columns
        df_extended[numeric_columns] = df_extended[numeric_columns].round(6)
        
        # Save to CSV
        try:
            df_extended.to_csv(self.output_path, index=False)
            logger.info(f"Extended TIDE output saved to: {self.output_path}")
            logger.info(f"Report contains {len(df_extended)} state records from {df_extended['TemporalPropertyID'].nunique()} temporal properties")
        except Exception as e:
            logger.error(f"Failed to save extended TIDE output to {self.output_path}: {e}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the learned TIDE cutpoints.
        For each sample, assign a state via the common assign_state() helper.
        If state_selection=True, states not in significant_states are assigned -1.
        """
        data = data.copy()
        if self.per_variable:
            def _assign(row):
                tpid = row[TEMPORAL_PROPERTY_ID]
                state = assign_state(row[VALUE], self.boundaries.get(tpid, []))
                if self.state_selection and tpid in self.significant_states:
                    # assign_state returns 1-indexed; significant_states uses 0-indexed
                    if (state - 1) not in self.significant_states[tpid]:
                        return -1
                return state
            data["state"] = data.apply(_assign, axis=1)
        else:
            def _assign_global(value):
                state = assign_state(value, self.boundaries if self.boundaries is not None else [])
                if self.state_selection and None in self.significant_states:
                    if (state - 1) not in self.significant_states[None]:
                        return -1
                return state
            data["state"] = data[VALUE].apply(_assign_global)

        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """Return the computed TIDE boundaries."""
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
        print("Available TIDE Scoring Methods:")
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

def tid3(data: pd.DataFrame, bins: int, per_variable: bool = True, min_duration_threshold: int = 2, max_gap: int = 1, extended_output: bool = False, output_path: str = "tid3_extended_output.csv", scoring_method: str = "max_t_stat_sum", nb_candidates: int = 100, duration_preference: str = "two_sided", significance_threshold: float = None, state_selection: bool = False):
    """
    Convenience function to run TID3 (Time Interval Duration Exploitation) on a dataset.

    TID3 creates temporal intervals by exploiting duration patterns in time intervals
    to find optimal discretization boundaries for continuous prediction and FCPM.

    Parameters:
      data: Input DataFrame.
      bins: Number of bins desired.
      per_variable: Whether to fit each variable separately.
      min_duration_threshold: Minimum consecutive state occurrences to consider for duration analysis.
      max_gap: Maximum gap between timestamps to consider as consecutive states.
      extended_output: If True, generates detailed CSV report with TID3 statistics.
      output_path: Path for the extended output CSV file.
      scoring_method: Scoring method to use. Available methods:
          - "max_t_stat_sum": Maximize sum of t-statistics for each state separately (default)
          - "avg_duration_diff_sum": Minimize sum of absolute differences in average durations
          - "max_ks_stat_sum": Maximize sum of Kolmogorov-Smirnov statistics (non-parametric)
          - "max_kl_divergence_sum": Maximize sum of symmetric KL divergence
          - "max_mannwhitney_sum": Maximize sum of Mann-Whitney U test significance
          - "max_wasserstein_sum": Maximize sum of Wasserstein (Earth Mover's) distances
      nb_candidates: Number of candidate cutpoints to evaluate. Default is 100.
      duration_preference: Direction preference for duration comparison. Options:
          - "two_sided": Favor any difference (two-tailed tests, default)
          - "class1_longer": Favor class 1 having longer durations (one-tailed)
          - "class0_longer": Favor class 0 having longer durations (one-tailed)
      significance_threshold: p-value threshold used when state_selection=True (default: None).
      state_selection: If True, marks non-significant states as -1 in transform output.
          Requires significance_threshold to be set. Cutoff boundaries are kept unchanged.

    Returns:
      symbolic_series: Transformed DataFrame with a "state" column (local state id).
      states: The boundaries (cutpoints) computed per variable by TID3.
    """
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]
    method_instance = TID3(bins, per_variable, min_duration_threshold=min_duration_threshold, max_gap=max_gap, extended_output=extended_output, output_path=output_path, scoring_method=scoring_method, nb_candidates=nb_candidates, duration_preference=duration_preference, significance_threshold=significance_threshold, state_selection=state_selection)
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states



# if __name__ == "__main__":
#     from ta_package import TemporalAbstraction


#     # data = pd.read_csv("/sise/robertmo-group/AyeletH/CPM_dataset/icu_ayelet/icu_no_zeros_filtered.csv")
#     data = pd.read_csv(r"C:\Users\eldar\PycharmProjects\Events_Extraction_Project\output_data_new2\stg1_progression_minAGE18_minTIME6_fromStage1_enhanced_metavision_carevue\final_formatted_data.csv")
#     ta = TemporalAbstraction(data)
#     result, states = ta.apply(method="tide", bins=3, per_variable=True, max_gap=120, extended_output=True,
#      output_path="C:\\Users\\eldar\\PycharmProjects\\HugoBot2\\ta_output\\tide_extended_output.csv", min_duration_threshold=None, scoring_method="max_t_stat_sum")
#     print(states)