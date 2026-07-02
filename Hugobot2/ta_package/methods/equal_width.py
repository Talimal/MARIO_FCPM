import pandas as pd
from .base import TAMethod
from ..utils import assign_state, paa_transform
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class EqualWidth(TAMethod):
    def __init__(self, bins: int, per_variable: bool = True, paa_method: str = None, paa_window: int = None):
        self.bins = bins
        self.boundaries = None
        self.per_variable = per_variable
        self.paa_method = paa_method
        self.paa_window = paa_window

    def fit(self, data: pd.DataFrame) -> None:
        if self.paa_method is not None and self.paa_window is not None:
            data = paa_transform(data, self.paa_window, agg_method=self.paa_method)
        if self.per_variable:
            self.boundaries = {}
            for tpid, group in data.groupby(TEMPORAL_PROPERTY_ID):
                min_val = group[VALUE].min()
                max_val = group[VALUE].max()
                self.boundaries[tpid] = [min_val + i * (max_val - min_val) / self.bins for i in range(1, self.bins)]
        else:
            min_val = data[VALUE].min()
            max_val = data[VALUE].max()
            self.boundaries = [min_val + i * (max_val - min_val) / self.bins for i in range(1, self.bins)]

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

def equal_width(data: pd.DataFrame, bins: int, per_variable: bool = True, paa_method: str = None, paa_window: int = None):
    ew = EqualWidth(bins, per_variable=per_variable, paa_method=paa_method, paa_window=paa_window)
    symbolic_series = ew.fit_transform(data)
    # symbolic_series = symbolic_series.rename(columns={"state": "StateId"})
    states = ew.get_states()
    return symbolic_series, states
