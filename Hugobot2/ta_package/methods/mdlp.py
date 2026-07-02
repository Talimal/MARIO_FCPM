# File: ta_package/methods/mdlp.py
import numpy as np
import pandas as pd
from .base import TAMethod
from ..utils import assign_state
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, VALUE


class MDLP(TAMethod):
    """
    Supervised discretization using entropy-minimizing splits in the style of
    Fayyad & Irani's MDLP, but with the MDL stopping criterion replaced by a
    fixed bin budget: recursion always continues until ``bins - 1`` cutpoints
    have been chosen (or no admissible split remains).

    Per-variable: ``self.boundaries`` is a dict ``{TemporalPropertyID: [sorted cutoffs]}``,
    same shape as TD4C / TID3.
    """

    def __init__(self, bins: int, per_variable: bool = True):
        self.bins = bins
        self.per_variable = per_variable
        self.boundaries = None

    @staticmethod
    def _entropy(classes: np.ndarray) -> float:
        if classes.size == 0:
            return 0.0
        _, counts = np.unique(classes, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _candidate_cutpoints(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Fayyad-Irani boundary points: midpoints between consecutive sorted-by-value
        rows whose class label differs. Falls back to midpoints between all
        consecutive distinct values if the class signal is degenerate.
        """
        order = np.argsort(values, kind="mergesort")
        v_sorted = values[order]
        c_sorted = classes[order]

        boundary_mids = []
        for i in range(1, v_sorted.size):
            if v_sorted[i] != v_sorted[i - 1] and c_sorted[i] != c_sorted[i - 1]:
                boundary_mids.append((v_sorted[i] + v_sorted[i - 1]) / 2.0)
        if boundary_mids:
            return np.unique(np.asarray(boundary_mids, dtype=float))

        distinct = np.unique(v_sorted)
        if distinct.size < 2:
            return np.empty(0, dtype=float)
        return (distinct[:-1] + distinct[1:]) / 2.0

    def _best_split(self, values: np.ndarray, classes: np.ndarray,
                    candidates: np.ndarray):
        """
        Return (best_cut, best_gain). ``best_cut`` is None when no candidate
        improves entropy or no candidates remain.
        """
        if candidates.size == 0 or values.size < 2:
            return None, 0.0
        parent_ent = self._entropy(classes)
        total = values.size
        best_cut = None
        best_gain = -np.inf
        for cut in candidates:
            left_mask = values < cut
            n_left = int(left_mask.sum())
            if n_left == 0 or n_left == total:
                continue
            ent_left = self._entropy(classes[left_mask])
            ent_right = self._entropy(classes[~left_mask])
            weighted = (n_left / total) * ent_left + ((total - n_left) / total) * ent_right
            gain = parent_ent - weighted
            if gain > best_gain:
                best_gain = gain
                best_cut = float(cut)
        if best_cut is None:
            return None, 0.0
        return best_cut, float(best_gain)

    def _generate_cutpoints(self, group: pd.DataFrame) -> list:
        values = group[VALUE].to_numpy(dtype=float)
        classes = group["Class"].to_numpy()
        target = max(self.bins - 1, 0)

        if target == 0:
            return []

        if np.unique(values).size < 2:
            tpid = group[TEMPORAL_PROPERTY_ID].iloc[0] if len(group) else "?"
            print(f"Warning: Not enough variability for TemporalPropertyID {tpid}; "
                  f"duplicating value as cutoff.")
            fill = float(values[0]) if values.size else 0.0
            return [fill] * target

        # Best-first: each partition tracks its rows. Candidates are recomputed
        # per-partition so that pure-class subsets still pick up the
        # all-distinct-midpoints fallback (needed for the "force exact bins"
        # semantics: we keep splitting even when entropy gain is zero).
        initial_cands = self._candidate_cutpoints(values, classes)
        if initial_cands.size == 0:
            tpid = group[TEMPORAL_PROPERTY_ID].iloc[0] if len(group) else "?"
            print(f"Warning: No candidate cutpoints for TemporalPropertyID {tpid}; "
                  f"duplicating value as cutoff.")
            fill = float(np.median(values))
            return [fill] * target

        partitions = [(values, classes)]
        best_per_partition = [self._best_split(values, classes, initial_cands)]
        cutpoints: list = []

        for _ in range(target):
            best_idx = -1
            best_gain = -np.inf
            best_cut = None
            for i, (cut, gain) in enumerate(best_per_partition):
                if cut is None:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_cut = cut
                    best_idx = i
            if best_idx == -1:
                tpid = group[TEMPORAL_PROPERTY_ID].iloc[0] if len(group) else "?"
                print(f"Warning: MDLP exhausted admissible splits for "
                      f"TemporalPropertyID {tpid} after {len(cutpoints)} cutoffs; "
                      f"padding with duplicates.")
                fill = cutpoints[-1] if cutpoints else float(np.median(values))
                while len(cutpoints) < target:
                    cutpoints.append(fill)
                break

            cutpoints.append(best_cut)
            v, c = partitions[best_idx]
            left_mask = v < best_cut
            v_left, c_left = v[left_mask], c[left_mask]
            v_right, c_right = v[~left_mask], c[~left_mask]
            cands_left = self._candidate_cutpoints(v_left, c_left)
            cands_right = self._candidate_cutpoints(v_right, c_right)

            partitions[best_idx] = (v_left, c_left)
            best_per_partition[best_idx] = self._best_split(v_left, c_left, cands_left)
            partitions.append((v_right, c_right))
            best_per_partition.append(self._best_split(v_right, c_right, cands_right))

        cutpoints.sort()
        return cutpoints

    def fit(self, data: pd.DataFrame) -> None:
        if not getattr(self, "entity_class", None):
            raise ValueError("MDLP requires an entity_class mapping (supervised method).")
        if self.per_variable:
            boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                group = group.assign(Class=group[ENTITY_ID].map(self.entity_class))
                boundaries[tpid] = self._generate_cutpoints(group)
            self.boundaries = boundaries
        else:
            data = data.assign(Class=data[ENTITY_ID].map(self.entity_class))
            self.boundaries = self._generate_cutpoints(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if self.per_variable:
            data["state"] = data.apply(
                lambda row: assign_state(row[VALUE],
                                         self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])),
                axis=1,
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
        return self.boundaries


def mdlp(data: pd.DataFrame, bins: int, per_variable: bool = True,
         entity_class: dict = None):
    """
    Convenience function to run MDLP on a dataset.

    Parameters:
      data: Input DataFrame.
      bins: Number of bins desired (forces exactly bins-1 cutpoints per variable).
      per_variable: Whether to fit each variable separately.
      entity_class: Mapping {EntityID: class_label}. Required (supervised method).

    Returns:
      symbolic_series: Transformed DataFrame with a "state" column.
      states: Boundaries dict {TemporalPropertyID: [cutoffs]}.
    """
    data = data[data[TEMPORAL_PROPERTY_ID] != -1]
    inst = MDLP(bins=bins, per_variable=per_variable)
    if entity_class is not None:
        inst.entity_class = entity_class
    symbolic_series = inst.fit_transform(data)
    states = inst.get_states()
    return symbolic_series, states
