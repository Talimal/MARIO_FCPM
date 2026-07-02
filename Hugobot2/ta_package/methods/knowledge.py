import pandas as pd
import numpy as np
from .base import TAMethod
from ..utils import assign_state
from ..constants import ENTITY_ID, TEMPORAL_PROPERTY_ID, TIMESTAMP, VALUE

class KnowledgeBased(TAMethod):
    def __init__(self, states: dict, per_variable: bool = True):
        """
        Parameters:
          states (dict or pd.DataFrame): Either a dictionary of pre-determined boundaries (cutoffs) 
                         for each variable, or a DataFrame with CSV format containing StateID, 
                         TemporalPropertyID, BinID, BinLow, BinHigh columns.
                         Dict format example: { state_id: [cutoff1, cutoff2, ...],
                                             "default": [default_cutoff1, default_cutoff2, ...] }
          per_variable (bool): If True, each TemporalPropertyID is processed separately (default True).
        """
        self.states = states
        self.per_variable = per_variable
        self.original_csv = None  # Store original CSV if provided
        self.state_mapping = {}  # Mapping from (tpid, value) to StateID

    def fit(self, data: pd.DataFrame) -> None:
        """
        In the knowledge-based method no learning is done—the boundaries are provided by the user.
        """
        # if state is df, store it and create mapping
        if isinstance(self.states, pd.DataFrame):
            self.original_csv = self.states.copy()
            df = self.states.copy()
            df['BinLow'] = pd.to_numeric(df['BinLow'], errors='coerce')
            df['BinHigh'] = pd.to_numeric(df['BinHigh'], errors='coerce')
            
            # Build state mapping: for each row, map (TemporalPropertyID, value_range) -> StateID
            # We'll use this for efficient lookup during transform
            for _, row in df.iterrows():
                tpid = row['TemporalPropertyID']
                state_id = row['StateID']
                bin_low = row['BinLow']
                bin_high = row['BinHigh']
                
                if tpid not in self.state_mapping:
                    self.state_mapping[tpid] = []
                self.state_mapping[tpid].append({
                    'StateID': state_id,
                    'BinLow': bin_low,
                    'BinHigh': bin_high
                })
            
            # Sort by BinLow for each variable to ensure correct ordering
            for tpid in self.state_mapping:
                self.state_mapping[tpid].sort(key=lambda x: x['BinLow'] if not np.isinf(x['BinLow']) else -np.inf)
            
            # Also create boundaries dict for backwards compatibility (used in composite mode)
            df_filtered = df[(df['BinLow'] != -np.inf) & (df['BinHigh'] != np.inf)]
            result = df_filtered.groupby('TemporalPropertyID').agg({
                'BinLow': 'min',
                'BinHigh': 'max'
            })
            final_dict = result.apply(lambda row: [row['BinLow'], row['BinHigh']], axis=1).to_dict()
            self.boundaries = final_dict
        else:
            # If states is already a dictionary, use it directly
            self.boundaries = self.states
            self.state_mapping = {}  # No CSV mapping available

    def transform(self, data: pd.DataFrame, method_config = None) -> pd.DataFrame:
        """
        Using the provided boundaries or CSV mapping, assign a state id for each sample.
        If CSV was provided, use StateID from the CSV directly.
        Otherwise, calculate state number from boundaries.
        """
        data = data.copy()

        if method_config is not None:
            for cfg in method_config:
                method_name = cfg.get("method")
                data = data[data[TEMPORAL_PROPERTY_ID] == method_name]

        # If we have CSV mapping, use StateID directly from CSV
        if self.state_mapping:
            def get_state_from_csv(row):
                tpid = row[TEMPORAL_PROPERTY_ID]
                value = row[VALUE]
                
                if tpid not in self.state_mapping:
                    return -1  # Unknown variable
                
                # Find which bin this value falls into
                for state_info in self.state_mapping[tpid]:
                    bin_low = state_info['BinLow']
                    bin_high = state_info['BinHigh']
                    
                    # Handle infinity values
                    if np.isinf(bin_low) and bin_low < 0:  # -inf
                        if value < bin_high:
                            return state_info['StateID']
                    elif np.isinf(bin_high) and bin_high > 0:  # +inf
                        if value >= bin_low:
                            return state_info['StateID']
                    else:
                        # Normal range: bin_low <= value < bin_high
                        if bin_low <= value < bin_high:
                            return state_info['StateID']
                
                return -1  # Value doesn't fall in any bin
            
            data["state"] = data.apply(get_state_from_csv, axis=1)
        else:
            # Original behavior: use boundaries to calculate state number
            if self.per_variable:
                # For each row, retrieve the boundaries for its variable;
                # if not found, use the boundaries under the key "default" (or an empty list if neither exists).
                data["state"] = data.apply(
                    lambda row: assign_state(
                        row[VALUE],
                        self.boundaries.get(row[TEMPORAL_PROPERTY_ID], self.boundaries.get("default", []))
                    ),
                    axis=1
                )
            else:
                default_boundaries = self.boundaries.get("default", [])
                data["state"] = data[VALUE].apply(
                    lambda v: assign_state(v, default_boundaries)
                )
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def get_states(self):
        """
        Return the states. If original CSV was provided, return it.
        Otherwise, return the boundaries dict.
        """
        if self.original_csv is not None:
            return self.original_csv
        return self.states 

# Method-level convenience function.
def knowledge(data: pd.DataFrame, states: dict, per_variable: bool = True):
    kb = KnowledgeBased(states, per_variable=per_variable)
    symbolic_series = kb.fit_transform(data)
    return symbolic_series, kb.get_states()