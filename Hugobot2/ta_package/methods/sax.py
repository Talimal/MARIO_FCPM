import pandas as pd
from scipy.stats import norm
from .base import TAMethod
from ..utils import assign_state, paa_transform
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class SAX(TAMethod):
    def __init__(self, bins: int, per_variable: bool = True, paa_method: str = None, paa_window: int = None):
        """
        Parameters:
          bins (int): Number of bins.
          per_variable (bool): Process each TemporalPropertyID separately (default True).
          paa_method (str): Optional PAA method ('mean', 'min', or 'max'); default is None (no PAA).
          paa_window (int): Optional window size for PAA; default is None.
        """
        self.bins = bins
        self.boundaries = None
        self.mean = None
        self.std = None
        self.per_variable = per_variable
        self.paa_method = paa_method
        self.paa_window = paa_window

    def fit(self, data: pd.DataFrame) -> None:
        if self.paa_method is not None and self.paa_window is not None:
            data = paa_transform(data, self.paa_window, agg_method=self.paa_method)
        if self.per_variable:
            self.boundaries = {}
            self.mean = {}
            self.std = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                m = group[VALUE].mean()
                s = group[VALUE].std() or 1
                self.mean[tpid] = m
                self.std[tpid] = s
                # Compute boundaries using the inverse CDF of the normal distribution.
                rv = norm(loc=m, scale=s)
                self.boundaries[tpid] = [rv.ppf(i/self.bins) for i in range(1, self.bins)]
        else:
            self.mean = data[VALUE].mean()
            self.std = data[VALUE].std() or 1
            # Compute boundaries using the inverse CDF of the normal distribution.
            rv = norm(loc=self.mean, scale=self.std)
            self.boundaries = [rv.ppf(i/self.bins) for i in range(1, self.bins)]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if self.paa_method is not None and self.paa_window is not None:
            data = paa_transform(data, self.paa_window, agg_method=self.paa_method)
        if self.per_variable:
            data["state"] = data.apply(lambda row: assign_state(row[VALUE], self.boundaries.get(row[TEMPORAL_PROPERTY_ID], [])), axis=1)
        else:
            data["state"] = data[VALUE].apply(lambda v: assign_state(v, self.boundaries if self.boundaries is not None else []))
        return data

    def get_states(self):
        return self.boundaries

# Method-level function for convenience.
def sax(data: pd.DataFrame, bins: int, per_variable: bool = True, paa_method: str = None, paa_window: int = None):
    sax_method = SAX(bins, per_variable=per_variable, paa_method=paa_method, paa_window=paa_window)
    symbolic_series = sax_method.fit_transform(data)
    # symbolic_series = symbolic_series.rename(columns={"state": "StateId"})
    states = sax_method.get_states()
    return symbolic_series, states
