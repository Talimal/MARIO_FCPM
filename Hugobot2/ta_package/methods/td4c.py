# File: ta_package/methods/td4c.py
import numpy as np
import pandas as pd
from scipy.stats import entropy
from .base import TAMethod
from ..utils import assign_state, candidate_selection, symmetric_kullback_leibler
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class TD4C(TAMethod):
    # Available distance measures - easily extensible
    AVAILABLE_DISTANCE_MEASURES = {
        "cosine": "Use cosine similarity",
        "kullback_leibler": "Use symmetric Kullback–Leibler divergence",
        "entropy": "Use the absolute difference of entropies"
    }
    
    def __init__(self, bins: int, per_variable: bool = True, distance_measure: str = "cosine"):
        """
        Parameters:
            bins (int): Desired number of bins (discretization intervals).
            per_variable (bool): If True, each TemporalPropertyID is fitted independently.
            distance_measure (str): Determines which distance function to use. Options:
                - "kullback_leibler": Use symmetric Kullback–Leibler divergence.
                - "entropy": Use the absolute difference of entropies.
                - "cosine": Use cosine similarity (default).
                Use TD4C.AVAILABLE_DISTANCE_MEASURES to see all options.
        """
        self.bins = bins
        self.per_variable = per_variable
        self.boundaries = None
        # Validate and set distance measure
        if distance_measure not in self.AVAILABLE_DISTANCE_MEASURES:
            raise ValueError(f"Unknown distance measure '{distance_measure}'. Available measures: {list(self.AVAILABLE_DISTANCE_MEASURES.keys())}")
        self.distance_measure = distance_measure
        if distance_measure == "kullback_leibler":
            self._distance_measure = symmetric_kullback_leibler
        elif distance_measure == "entropy":
            self._distance_measure = lambda p, q: abs(entropy(p) - entropy(q))
        elif distance_measure == "cosine":
            self._distance_measure = lambda p, q: 1.0 - (np.dot(p, q) / np.sqrt(np.dot(p, p) * np.dot(q, q)))

    def _generate_cutpoints(self, df: pd.DataFrame):
        """
        For a given DataFrame (corresponding to one variable), choose candidate cutpoints via candidate_selection.
        The scoring function compares class distributions across bins using the chosen distance measure.
        """
        # Ensure class information exists; if not, create a default class (e.g., 0).
        if 'Class' not in df.columns:
            df = df.assign(Class=0)
        # candidate_selection returns (candidates, scores); we only use the candidates.
        candidates, scores = candidate_selection(
            df,
            self.bins,
            lambda d, cutoffs: self._ddm_scoring_function(d, cutoffs)
        )
        

        # if all values in df[TemporalPropertyValue] are the same return candidates in the len of self.bins -1 with the value
        if df[VALUE].nunique() == 1:
            print(f"Warning: Not enough variability in for TemporalPropertyID {df[TEMPORAL_PROPERTY_ID].max()} for cutpoints. Using default candidates.")
            candidates = [df[VALUE].min()] * (self.bins - 1)
        return candidates

    def _ddm_scoring_function(self, df: pd.DataFrame, cutoffs):
        """
        Given a DataFrame and a list of cutoffs, compute a score.
        The score is calculated by first discretizing df[VALUE] based on the cutoffs,
        then for each class (from df['Class']) computing the distribution over bins and finally
        summing pairwise distances between the class distributions.
        
        Note: Epsilon smoothing is applied to avoid zero probabilities which cause
        infinite KL divergence when comparing distributions with non-overlapping support.
        """
        bins_array = [-np.inf] + list(cutoffs) + [np.inf]
        df = df.assign(Bin=pd.cut(df[VALUE], bins=bins_array, labels=False))
        classes = sorted(df['Class'].unique())
        nb_bins = len(bins_array) - 1
        class_distribs = np.zeros((len(classes), nb_bins))
        
        # Epsilon for smoothing to avoid zero probabilities (prevents inf in KL divergence)
        epsilon = 1e-10
        
        for i, cls in enumerate(classes):
            sub = df[df['Class'] == cls]
            if sub.empty:
                continue
            counts = sub['Bin'].value_counts().sort_index()
            # Build a probability vector of length nb_bins.
            # Use bin_id as index to place counts correctly (like TIDE),
            # avoiding the bug where missing bins would scramble the vector.
            v = np.zeros(nb_bins)
            for bin_id, count in counts.items():
                if bin_id < nb_bins:
                    v[int(bin_id)] = count
            if v.sum() > 0:
                # Apply epsilon smoothing to avoid zero probabilities
                v_smooth = v + epsilon
                class_distribs[i] = v_smooth / v_smooth.sum()
        
        score = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                score += self._distance_measure(class_distribs[i], class_distribs[j])
        return score

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the TD4C model by generating cutpoints for each variable.
        In per_variable mode, fit each TemporalPropertyID independently.
        """
        if self.per_variable:
            boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                # If a mapping of entity classes exists, merge it in.
                if 'Class' not in group.columns and hasattr(self, 'entity_class') and self.entity_class:
                    group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))
                else:
                    group = group.assign(Class=0)
                # if len(group) <= self.bins:
                    
                #     print(f"Warning: Not enough data for TemporalPropertyID {tpid} to generate cutpoints.")
                #     boundaries[tpid] = [-np.inf, np.inf]  # Default boundaries
                #     continue
                boundaries[tpid] = self._generate_cutpoints(group)
            self.boundaries = boundaries
        else:
            if 'Class' not in data.columns and hasattr(self, 'entity_class') and self.entity_class:
                data = data.assign(Class=data[ENTITY_ID].map(self.entity_class))
            else:
                data = data.assign(Class=0)
            # if len(data) <= self.bins:
            #     print(f"Warning: Not enough data for TemporalPropertyID {tpid} to generate cutpoints.")
            #     boundaries = [-np.inf, np.inf]  # Default boundaries
            self.boundaries = self._generate_cutpoints(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the learned cutpoints.
        For each sample, assign a state via the common assign_state() helper.
        """
        data = data.copy()
        if self.per_variable:
            data["state"] = data.apply(
                lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])),
                axis=1
            )
        else:
            data["state"] = data[VALUE].apply(
                lambda v: assign_state(v, self.boundaries if self.boundaries is not None else [])
            )
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """Return the computed boundaries."""
        return self.boundaries

def td4c(data: pd.DataFrame, bins: int, per_variable: bool = True, distance_measure: str = "kullback_leibler"):
    """
    Convenience function to run TD4C on a dataset.
    Parameters:
      data: Input DataFrame.
      bins: Number of bins desired.
      per_variable: Whether to fit each variable separately.
      distance_measure: Which distance measure to use ("kullback_leibler", "entropy", or "cosine").
    Returns:
      symbolic_series: Transformed DataFrame with a "state" column (local state id).
      states: The boundaries (cutpoints) computed per variable.
    """
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]
    method_instance = TD4C(bins, per_variable, distance_measure=distance_measure)
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states