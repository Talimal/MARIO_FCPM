# File: ta_package/methods/persist.py
import numpy as np
import pandas as pd
from collections import Counter
from .base import TAMethod
from ..utils import assign_state, candidate_selection, symmetric_kullback_leibler
from ..constants import TEMPORAL_PROPERTY_ID, VALUE

class Persist(TAMethod):
    def __init__(self, bins: int, per_variable: bool = True):
        """
        Parameters:
            bins (int): Number of bins desired.
            per_variable (bool): If True, process each variable separately.
        """
        self.bins = bins
        self.per_variable = per_variable
        self.boundaries = None

    @staticmethod
    def _transition_matrix(discrete_vals, nb_bins):
        """Joint counts T[i, j] = #(prev=i, cur=j), normalized by total transitions."""
        T = np.zeros((nb_bins, nb_bins))
        if len(discrete_vals) < 2:
            return T
        for prev, cur in zip(discrete_vals[:-1], discrete_vals[1:]):
            T[int(prev), int(cur)] += 1
        return T / (len(discrete_vals) - 1)

    @staticmethod
    def _state_probabilities(discrete_vals, nb_bins):
        c = Counter(discrete_vals)
        probs = np.array([c.get(i, 0) for i in range(nb_bins)])
        if probs.sum() == 0:
            return probs
        return probs / float(probs.sum())

    @staticmethod
    def _all_states_persistence(discrete_vals, nb_bins):
        state_probs = Persist._state_probabilities(discrete_vals, nb_bins)
        T = Persist._transition_matrix(discrete_vals, nb_bins)
        # Paper Sec. 3: smooth zero probabilities with n^-1 to keep KL finite.
        eps = 1.0 / max(2, len(discrete_vals))
        scores = []
        for i in range(nb_bins):
            row_sum = T[i, :].sum()
            # Conditional self-transition A(i,i) = P(cur=i | prev=i) per Mörchen & Ultsch 2005 Sec. 3.
            a_ii = T[i, i] / row_sum if row_sum > 0 else 0.0
            s = state_probs[i]
            a_ii = min(max(a_ii, eps), 1 - eps)
            s = min(max(s, eps), 1 - eps)
            skl = symmetric_kullback_leibler([a_ii, 1 - a_ii], [s, 1 - s])
            sign = 1.0 if a_ii >= s else -1.0
            scores.append(sign * skl)
        if np.any(np.isinf(scores)):
            return -np.inf
        return float(np.mean(scores))

    def _generate_cutpoints(self, df: pd.DataFrame):
        """
        Use candidate_selection to choose cutpoints.
        The scoring function returns the persistence score calculated over all states,
        and rejects any candidate cut that would produce a bin covering less than 5% of
        the points (per Mörchen & Ultsch 2005 Sec. 4).
        """
        min_count = max(1, int(np.ceil(0.05 * len(df))))

        def scoring(d, cutoffs):
            bin_sizes = d.groupby('Bin').size()
            if bin_sizes.empty or bin_sizes.min() < min_count:
                return -np.inf
            return Persist._all_states_persistence(d['Bin'].values, len(cutoffs) + 1)

        candidates, _ = candidate_selection(df, self.bins, scoring)
        return candidates

    def fit(self, data: pd.DataFrame) -> None:
        if self.per_variable:
            boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                boundaries[tpid] = self._generate_cutpoints(group)
            self.boundaries = boundaries
        else:
            self.boundaries = self._generate_cutpoints(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if self.per_variable:
            data["state"] = data.apply(lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID])), axis=1)
        else:
            data["state"] = data[VALUE].apply(lambda v: assign_state(v, self.boundaries))
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        return self.boundaries

def persist(data: pd.DataFrame, bins: int, per_variable: bool = True):
    method_instance = Persist(bins, per_variable)
    symbolic_series = method_instance.fit_transform(data)
    states = method_instance.get_states()
    return symbolic_series, states
