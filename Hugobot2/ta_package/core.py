
import os
import math
import pandas as pd
from .utils import paa_transform, generate_KL_content, save_entity_ids, remove_na, parse_td4c_config
from .constants import ENTITY_ID, VALUE, TEMPORAL_PROPERTY_ID, TIMESTAMP
# import List
from typing import List

class TemporalAbstraction:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame containing the time series.
        Expected columns: EntityID, TemporalPropertyID, TimeStamp, TemporalPropertyValue.
        
        Note: Rows with TemporalPropertyID == -1 are reserved for class assignment.
              Their TemporalPropertyValue indicates the class (e.g., 0 or 1) for that EntityID.
        """
        self.data = data
        self.entity_class = {}  # Mapping: {EntityID: class}
        self.method_config = None

    def apply(self, method: str = None, method_config: dict = None,
              paa: str = None, paa_window: int = None,
              per_entity: bool = False,
              split_test: bool = False,
              save_output: bool = True, output_dir: str = None, train_ratio: float = 0.7,
              max_gap: int = 1, train_states = None, skip_var: List[int] = None, **kwargs):
        """
        Apply temporal abstraction and (optionally) split into train and test sets.
        
        There are two modes:
          - Single-method mode (if method_config is None)
          - Composite mode (if method_config is provided)
        
        Additionally, the user can choose to apply composite discretization per entity by setting
        per_entity=True. In that case, the algorithm will first group the data by ENTITY_ID and then,
        for each entity, group by TemporalPropertyID and apply the discretization methods specified in
        the configuration.
        
        Before processing, rows with TemporalPropertyID == -1 (class assignment) are extracted and then dropped.
        In train–test mode, boundaries are learned from the training set only and then applied to the test set.
        
        Parameters:
          method (str): Discretization method name for single-method mode (e.g., "equal_width", "sax", "gradient", "td4c", "persist").
          method_config (dict): Dictionary mapping TemporalPropertyID (or for per_entity mode, ENTITY_ID) to configuration(s)
                                for composite mode. For per_entity mode, keys can be individual entity IDs or groups.
          paa (str): Optional global PAA method ("mean", "min", or "max").
          paa_window (int): Global window size for PAA.
          per_entity (bool): If True, perform composite discretization separately for each entity.
          split_test (bool): If True, split data by unique ENTITY_ID into train and test sets.
          save_output (bool): Whether to save outputs (default True).
          output_dir (str): Output directory; if None, defaults to "ta_output".
          max_gap (int): Maximum gap for merging intervals in the KL file.
          kwargs: Additional parameters for single-method mode.
        
        Returns:
          In single-method mode: (final_result, final_states)
          In composite mode with split_test: ((final_train, final_test), final_states)
        """
        data_to_use = self.data.copy()
        if skip_var is not None:
            print(f"Skipping variables: {skip_var}")
            skip_rows = data_to_use[data_to_use[TEMPORAL_PROPERTY_ID].isin(skip_var)] # check if temporalPropertyID is in the list skip_var
            # remove skip rows from data_to_use
            data_to_use = data_to_use[~data_to_use.index.isin(skip_rows.index)]
        
        
        # --- Extract Class Assignment Rows ---
        class_rows = data_to_use[data_to_use[TEMPORAL_PROPERTY_ID] == -1].copy()
        if not class_rows.empty:
            for _, row in class_rows.iterrows():
                ent = row[ENTITY_ID]
                self.entity_class[ent] = int(float(row[VALUE]))
        data_to_use = data_to_use[data_to_use[TEMPORAL_PROPERTY_ID] != -1]
        # drop na
        data_to_use = remove_na(data_to_use)

        # Apply global PAA pre-processing if specified.
        if paa is not None and paa_window is not None:
            data_to_use = paa_transform(data_to_use, paa_window, agg_method=paa)
        
        if split_test:
            train_data, test_data = self._split_train_test(data_to_use, train_ratio=train_ratio)
        else:
            train_data = data_to_use.copy()
            test_data = None

        ta_method = None

        if method_config is not None:
            # --- Composite Mode ---
            if per_entity:
                # New per-entity composite processing:
                if split_test:
                    final_train = self._composite_fit_transform_entity(train_data, method_config)
                    final_test = self._composite_transform_entity(test_data, method_config)
                    final_result = (final_train, final_test)
                else:
                    final_train = self._composite_fit_transform_entity(train_data, method_config)
                    final_result = final_train
                    final_states = self._entity_state_mapping  # (this is built inside _composite_fit_transform_entity)
            else:
                # Composite mode as before (grouping by TemporalPropertyID)
                if split_test:
                    final_train, final_states = self._composite_fit_transform(train_data, method_config)
                    final_test = self._composite_transform(test_data, method_config)
                    final_result = (final_train, final_test)
                else:
                    final_train, final_states = self._composite_fit_transform(train_data, method_config)
                    final_result = final_train
        else:
            # --- Single-Method Mode (unchanged) ---
            if split_test:
                if method == "equal_width":
                    from .methods.equal_width import EqualWidth
                    ta_method = EqualWidth(**kwargs)
                elif method == "equal_frequency":
                    from .methods.equal_frequency import EqualFrequency
                    ta_method = EqualFrequency(**kwargs)
                elif method == "sax":
                    from .methods.sax import SAX
                    ta_method = SAX(**kwargs)
                elif method == "gradient":
                    from .methods.gradient import Gradient
                    ta_method = Gradient(**kwargs)
                elif method == "td4c" or (isinstance(method, str) and method.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_kwargs = kwargs.copy()
                    if method.startswith("td4c_") and "distance_measure" not in td4c_kwargs:
                        distance_measure = parse_td4c_config(method)
                        td4c_kwargs["distance_measure"] = distance_measure
                    ta_method = TD4C(**td4c_kwargs)
                elif method == "persist":
                    from .methods.persist import Persist
                    ta_method = Persist(**kwargs)
                elif method == "tid3":
                    from .methods.tid3 import TID3
                    ta_method = TID3(**kwargs)
                    ta_method.entity_class = self.entity_class  # Pass entity-class mapping
                elif method == "mdlp":
                    from .methods.mdlp import MDLP
                    ta_method = MDLP(**kwargs)
                    ta_method.entity_class = self.entity_class  # Pass entity-class mapping
                else:
                    raise ValueError(f"Method '{method}' is not supported.")
                ta_method.fit(train_data)
                final_train = ta_method.transform(train_data)
                final_test = ta_method.transform(test_data)
                final_result = (final_train, final_test)
                final_states = ta_method.get_states()
            else:
                if method == "equal_width":
                    from .methods.equal_width import EqualWidth
                    ta_method = EqualWidth(**kwargs)
                elif method == "equal_frequency":
                    from .methods.equal_frequency import EqualFrequency
                    ta_method = EqualFrequency(**kwargs)
                elif method == "sax":
                    from .methods.sax import SAX
                    ta_method = SAX(**kwargs)
                elif method == "gradient":
                    from .methods.gradient import Gradient
                    ta_method = Gradient(**kwargs)
                elif method == "td4c" or (isinstance(method, str) and method.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_kwargs = kwargs.copy()
                    if method.startswith("td4c_") and "distance_measure" not in td4c_kwargs:
                        distance_measure = parse_td4c_config(method)
                        td4c_kwargs["distance_measure"] = distance_measure
                    ta_method = TD4C(**td4c_kwargs)
                elif method == "persist":
                    from .methods.persist import Persist
                    ta_method = Persist(**kwargs)
                elif method == "knowledge":
                    from .methods.knowledge import KnowledgeBased
                    if train_states is None:
                        raise ValueError("train_states parameter is required for knowledge-based method")
                    ta_method = KnowledgeBased(states=train_states, **kwargs)
                elif method == "tid3":
                    from .methods.tid3 import TID3
                    ta_method = TID3(**kwargs)
                    ta_method.entity_class = self.entity_class  # Pass entity-class mapping
                elif method == "mdlp":
                    from .methods.mdlp import MDLP
                    ta_method = MDLP(**kwargs)
                    ta_method.entity_class = self.entity_class  # Pass entity-class mapping
                else:
                    raise ValueError(f"Method '{method}' is not supported.")
                final_result = ta_method.fit_transform(train_data)
                final_states = ta_method.get_states()
        
        if skip_var is not None:
            # concat final_result with skip rows
            final_result = pd.concat([final_result, skip_rows], ignore_index=True)
        
        if save_output:
            if output_dir is None:
                output_dir = "ta_output"
            if split_test:
                self._save_results(os.path.join(output_dir, "train"), final_train, final_states, max_gap)
                self._save_results(os.path.join(output_dir, "test"), final_test, final_states, max_gap)
                if ta_method is not None and hasattr(ta_method, 'save_scores_summary') and ta_method.final_scores_per_tpid:
                    ta_method.save_scores_summary(os.path.join(output_dir, "train"))
                elif getattr(self, '_tid3_scores_per_tpid', {}):
                    self._save_tid3_scores_csv(os.path.join(output_dir, "train"))
            else:
                self._save_results(output_dir, final_result, final_states, max_gap)
                if ta_method is not None and hasattr(ta_method, 'save_scores_summary') and ta_method.final_scores_per_tpid:
                    ta_method.save_scores_summary(output_dir)
                elif getattr(self, '_tid3_scores_per_tpid', {}):
                    self._save_tid3_scores_csv(output_dir)
                
        if split_test:
            return (final_train, final_test), final_states
        else:
            return final_result, final_states

    def _split_train_test(self, data: pd.DataFrame, train_ratio: float = 0.7):
        unique_ids = data[ENTITY_ID].unique()
        unique_ids = sorted(unique_ids)
        cutoff = int(len(unique_ids) * train_ratio)
        train_ids = unique_ids[:cutoff]
        test_ids = unique_ids[cutoff:]
        train = data[data[ENTITY_ID].isin(train_ids)]
        test = data[data[ENTITY_ID].isin(test_ids)]
        return train, test

    # New helper functions for per_entity composite mode:
    def _composite_fit_transform_entity(self, train_data: pd.DataFrame, method_config: dict):
        """
        Composite mode fitting on training data, grouping first by ENTITY_ID.
        For each entity, the data is processed (grouped by TemporalPropertyID) using the configurations
        specified for that entity (or using the "default" configuration).
        The results for each entity are concatenated together.
        """
        entity_results = []
        entity_state_mapping = []  # This will accumulate state mapping rows for all entities.
        # Loop over each entity.
        for ent, ent_group in train_data.groupby(ENTITY_ID):
            # Look up configuration for this entity.
            # For per_entity mode, the method_config keys are entity IDs (or a default).
            if ent in method_config:
                cfg = method_config[ent]
            else:
                cfg = method_config.get("default")
            # If configuration is not a list, wrap it in a list.
            if not isinstance(cfg, list):
                cfg = [cfg]
            # Process this entity using a helper function:
            ent_result, ent_states = self._composite_fit_transform_entity_single(ent_group, cfg)
            entity_results.append(ent_result)
            entity_state_mapping.extend(ent_states)
        # Optionally, update a class attribute with per-entity state mapping.
        self._entity_state_mapping = entity_state_mapping
        final_entity_data = pd.concat(entity_results, ignore_index=True)
        return final_entity_data, entity_state_mapping

    def _composite_fit_transform_entity_single(self, data: pd.DataFrame, config):
        """
        Process composite mode discretization for a single entity's data.
        Data here is already filtered for one ENTITY_ID.
        We group by TemporalPropertyID and apply the methods (possibly multiple) as in the standard composite mode.
        Returns the discretized data for this entity and a list of state mapping rows.
        """
        global_mapping = {}  # local mapping for this entity
        global_states_rows = []  # list of state mapping rows for this entity
        composite_results = []
        # Process per variable (TemporalPropertyID) as before.
        for tpid, subset in data.groupby(TEMPORAL_PROPERTY_ID):
            # For each variable, look up configuration.
            if tpid in config:
                cfgs = config[tpid]
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            local_global_states = {}
            for cfg in cfgs:
                method_name = cfg.get("method")
                params = cfg.copy()
                params.pop("method", None)
                # For per-entity mode, if method-specific PAA is provided, apply it:
                if "paa_method" in cfg and "paa_window" in cfg:
                    subset_method = paa_transform(subset.copy(), cfg["paa_window"], agg_method=cfg["paa_method"])
                else:
                    subset_method = subset.copy()
                # Dispatch the discretization method.
                if method_name == "equal_width":
                    from .methods.equal_width import equal_width
                    local_result, local_states = equal_width(subset_method, **params, per_variable=True)
                elif method_name == "equal_frequency":
                    from .methods.equal_frequency import equal_frequency
                    local_result, local_states = equal_frequency(subset_method, **params, per_variable=True)
                elif method_name == "sax":
                    from .methods.sax import sax
                    local_result, local_states = sax(subset_method, **params, per_variable=True)
                elif method_name == "gradient":
                    from .methods.gradient import gradient
                    local_result, local_states = gradient(subset_method, **params, per_variable=True)
                elif method_name == "td4c" or (isinstance(method_name, str) and method_name.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_params = params.copy()
                    if method_name.startswith("td4c_") and "distance_measure" not in td4c_params:
                        distance_measure = parse_td4c_config(method_name)
                        td4c_params["distance_measure"] = distance_measure
                    td4c_inst = TD4C(**td4c_params, per_variable=True)
                    local_result = td4c_inst.fit_transform(subset_method)
                    local_states = td4c_inst.get_states()
                elif method_name == "persist":
                    from .methods.persist import Persist
                    persist_inst = Persist(**params, per_variable=True)
                    local_result = persist_inst.fit_transform(subset_method)
                    local_states = persist_inst.get_states()
                elif method_name == "tid3":
                    from .methods.tid3 import TID3
                    tid3_inst = TID3(**params)
                    tid3_inst.entity_class = self.entity_class  # Pass entity-class mapping
                    local_result = tid3_inst.fit_transform(subset_method)
                    local_states = tid3_inst.get_states()
                else:
                    raise ValueError(f"Unsupported method: {method_name} for variable {tpid}")
                if isinstance(local_states, dict):
                    boundaries = local_states.get(tpid)
                else:
                    boundaries = local_states
                col_name = f"state_{method_name}"
                local_result = local_result.copy()
                local_result[col_name] = local_result.apply(
                    lambda row: self._map_local_state_ex(tpid, method_name, boundaries,
                                                          int(row["state"]), global_mapping, global_states_rows),
                    axis=1
                )
                local_global_states[method_name] = local_result[col_name]
            # Concatenate the results of different methods for this variable.
            var_result = pd.concat(
                [subset.assign(StateID=local_global_states[m], MethodName=m)
                 for m in sorted(local_global_states.keys())],
                ignore_index=True
            )
            composite_results.append(var_result)
        final_data = pd.concat(composite_results, ignore_index=True)
        return final_data, global_states_rows

    def _composite_fit_transform_entity_single(self, data: pd.DataFrame, config):
        """
        Process composite mode discretization for a single entity's data.
        Data here is already filtered for one ENTITY_ID.
        We group by TemporalPropertyID and apply the methods (possibly multiple) as in the standard composite mode.
        Returns the discretized data for this entity and a list of state mapping rows.
        """
        global_mapping = {}  # local mapping for this entity
        global_states_rows = []  # list of state mapping rows for this entity
        composite_results = []
        # Process per variable (TemporalPropertyID) as before.
        for tpid, subset in data.groupby(TEMPORAL_PROPERTY_ID):
            # For each variable, look up configuration.
            if tpid in config:
                cfgs = config[tpid]
            else:
                cfgs = config.get("default")
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            local_global_states = {}
            for cfg in cfgs:
                method_name = cfg.get("method")
                params = cfg.copy()
                params.pop("method", None)
                # For per-entity mode, if method-specific PAA is provided, apply it:
                if "paa_method" in cfg and "paa_window" in cfg:
                    subset_method = paa_transform(subset.copy(), cfg["paa_window"], agg_method=cfg["paa_method"])
                else:
                    subset_method = subset.copy()
                # Dispatch the discretization method.
                if method_name == "equal_width":
                    from .methods.equal_width import equal_width
                    local_result, local_states = equal_width(subset_method, **params, per_variable=True)
                elif method_name == "equal_frequency":
                    from .methods.equal_frequency import equal_frequency
                    local_result, local_states = equal_frequency(subset_method, **params, per_variable=True)
                elif method_name == "sax":
                    from .methods.sax import sax
                    local_result, local_states = sax(subset_method, **params, per_variable=True)
                elif method_name == "gradient":
                    from .methods.gradient import gradient
                    local_result, local_states = gradient(subset_method, **params, per_variable=True)
                elif method_name == "td4c" or (isinstance(method_name, str) and method_name.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_params = params.copy()
                    if method_name.startswith("td4c_") and "distance_measure" not in td4c_params:
                        distance_measure = parse_td4c_config(method_name)
                        td4c_params["distance_measure"] = distance_measure
                    td4c_inst = TD4C(**td4c_params, per_variable=True)
                    local_result = td4c_inst.fit_transform(subset_method)
                    local_states = td4c_inst.get_states()
                elif method_name == "persist":
                    from .methods.persist import Persist
                    persist_inst = Persist(**params, per_variable=True)
                    local_result = persist_inst.fit_transform(subset_method)
                    local_states = persist_inst.get_states()
                elif method_name == "tid3":
                    from .methods.tid3 import TID3
                    tid3_inst = TID3(**params)
                    tid3_inst.entity_class = self.entity_class  # Pass entity-class mapping
                    local_result = tid3_inst.fit_transform(subset_method)
                    local_states = tid3_inst.get_states()
                else:
                    raise ValueError(f"Unsupported method: {method_name} for variable {tpid}")
                if isinstance(local_states, dict):
                    boundaries = local_states.get(tpid)
                else:
                    boundaries = local_states
                col_name = f"state_{method_name}"
                local_result = local_result.copy()
                local_result[col_name] = local_result.apply(
                    lambda row: self._map_local_state_ex(tpid, method_name, boundaries,
                                                          int(row["state"]), global_mapping, global_states_rows),
                    axis=1
                )
                local_global_states[method_name] = local_result[col_name]
            # Concatenate the results of different methods for this variable.
            var_result = pd.concat(
                [subset.assign(StateID=local_global_states[m], MethodName=m)
                 for m in sorted(local_global_states.keys())],
                ignore_index=True
            )
            composite_results.append(var_result)
        final_data = pd.concat(composite_results, ignore_index=True)
        return final_data, global_states_rows
    
    # def _map_local_state_ex(self, tpid, method_name, boundaries, local_state, global_mapping, global_states_rows):
    #     key = (tpid, method_name, local_state)
    #     if key not in global_mapping:
    #         num_bins = len(boundaries) + 1 if boundaries is not None else 1
    #         if boundaries is None:
    #             bin_low = None
    #             bin_high = None
    #         else:
    #             if local_state == 1:
    #                 bin_low = -math.inf
    #                 bin_high = boundaries[0]
    #             elif local_state == num_bins:
    #                 bin_low = boundaries[-1]
    #                 bin_high = math.inf
    #             else:
    #                 bin_low = boundaries[local_state - 2]
    #                 bin_high = boundaries[local_state - 1]
    #         global_id = len(global_mapping) + 1
    #         global_mapping[key] = global_id
    #         global_states_rows.append({
    #             "StateID": global_id,
    #             "TemporalPropertyID": tpid,
    #             "MethodName": method_name,
    #             "BinId": local_state,
    #             "BinLow": round(bin_low, 5) if bin_low is not None else None,
    #             "BinHigh": round(bin_high, 5) if bin_high is not None else None,
    #         })
    #     return global_mapping[key]

    def _composite_fit_transform(self, train_data: pd.DataFrame, method_config: dict):
        """
        Composite mode fitting on training data only.
        For each variable and each method configuration, fit and transform the training subset.
        Returns (final_train, final_states).
        """
        global_mapping = {}
        global_states_rows = []
        composite_results = []
        default_config = method_config.get("default", None)
        unique_vars = train_data[TEMPORAL_PROPERTY_ID].unique()
        
        # STEP 1: Pre-process ALL knowledge method states across ALL variables FIRST
        # This ensures all knowledge StateIDs are registered before other methods start
        knowledge_results = {}  # Store knowledge results per variable to use later
        
        for tpid in unique_vars:
            subset = train_data[train_data[TEMPORAL_PROPERTY_ID] == tpid]
            
            if isinstance(method_config, dict) and tpid in method_config:
                cfgs = method_config[tpid]
            else:
                cfgs = method_config.get("default")
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            
            # Look for knowledge method
            for config in cfgs:
                if config.get("method") == "knowledge":
                    method_name = "knowledge"
                    params = config.copy()
                    params.pop("method", None)
                    
                    # Run knowledge method
                    from .methods.knowledge import knowledge
                    local_result, local_states = knowledge(subset, **params, per_variable=True)
                    
                    # If it's a CSV DataFrame, add ALL its states to global_states_rows NOW
                    if isinstance(local_states, pd.DataFrame):
                        for _, row in local_states[local_states['TemporalPropertyID'] == tpid].iterrows():
                            state_id = int(row['StateID'])
                            bin_id = row.get('BinID', row.get('BinId', 0))
                            
                            global_states_rows.append({
                                "StateID": state_id,
                                "TemporalPropertyID": row['TemporalPropertyID'],
                                "TemporalPropertyName": row.get('TemporalPropertyName', ''),
                                "MethodName": method_name,
                                "BinId": bin_id,
                                "BinLabel": row.get('BinLabel', ''),
                                "BinLow": row['BinLow'],
                                "BinHigh": row['BinHigh'],
                            })
                        
                        # Store the result to use when building final composite results
                        knowledge_results[tpid] = (local_result, local_states)
                    break  # Only one knowledge method per variable
        
        # Accumulators for TID3 scores summary CSV
        self._tid3_scores_per_tpid = {}
        self._tid3_cutoffs_per_tpid = {}

        # STEP 1.5: Pre-process TID3 with multivariate refinement (needs all variables at once)
        # The per-variable loop below creates a separate TID3 per variable, so multivariate
        # refinement (which requires >= 2 variables) would never run. We handle it here by
        # fitting a single TID3 instance on ALL variables, then reusing its results in the loop.
        tid3_mv_inst = None
        _default_cfgs = method_config.get("default", [])
        if not isinstance(_default_cfgs, list):
            _default_cfgs = [_default_cfgs]
        for _cfg in _default_cfgs:
            if _cfg.get("method") == "tid3" and _cfg.get("multivariate_refinement", False):
                from .methods.tid3 import TID3
                _params = _cfg.copy()
                _params.pop("method", None)
                tid3_mv_inst = TID3(**_params)
                tid3_mv_inst.entity_class = self.entity_class
                tid3_mv_inst.fit(train_data)
                self.tid3_runtime_phase1_s = tid3_mv_inst.runtime_phase1_s
                self.tid3_runtime_phase2_s = tid3_mv_inst.runtime_phase2_s
                self._tid3_scores_per_tpid.update(tid3_mv_inst.final_scores_per_tpid)
                self._tid3_cutoffs_per_tpid.update(tid3_mv_inst.final_cutoffs_per_tpid)
                break

        # STEP 2: Now process each variable with ALL methods (including knowledge from cache)
        for tpid in unique_vars:
            subset = train_data[train_data[TEMPORAL_PROPERTY_ID] == tpid]
            # Here, we assume that for composite per-variable mode, method_config is a dict keyed by tpid.
            if isinstance(method_config, dict) and tpid in method_config:
                cfgs = method_config[tpid]
            else:
                cfgs = method_config.get("default")
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            local_global_states = {}
            
            for config in cfgs:
                method_name = config.get("method")
                params = config.copy()
                params.pop("method", None)
                
                # Check if this is knowledge method and we already processed it
                if method_name == "knowledge" and tpid in knowledge_results:
                    # Use cached knowledge result
                    local_result, local_states = knowledge_results[tpid]
                    col_name = f"state_{method_name}"
                    local_result = local_result.copy()
                    # The state column already contains the correct StateID from the CSV
                    local_result[col_name] = local_result["state"]
                    local_global_states[method_name] = local_result[col_name]
                    continue  # Skip to next method
                
                # Process other methods normally
                if method_name == "equal_width":
                    from .methods.equal_width import equal_width
                    local_result, local_states = equal_width(subset, **params, per_variable=True)
                elif method_name == "equal_frequency":
                    from .methods.equal_frequency import equal_frequency
                    local_result, local_states = equal_frequency(subset, **params, per_variable=True)
                elif method_name == "sax":
                    from .methods.sax import sax
                    local_result, local_states = sax(subset, **params, per_variable=True)
                elif method_name == "gradient":
                    from .methods.gradient import gradient
                    local_result, local_states = gradient(subset, **params, per_variable=True)
                elif method_name == "td4c" or (isinstance(method_name, str) and method_name.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_params = params.copy()
                    if method_name.startswith("td4c_") and "distance_measure" not in td4c_params:
                        distance_measure = parse_td4c_config(method_name)
                        td4c_params["distance_measure"] = distance_measure
                    td4c_inst = TD4C(**td4c_params, per_variable=True)
                    local_result = td4c_inst.fit_transform(subset)
                    local_states = td4c_inst.get_states()
                elif method_name == "persist":
                    from .methods.persist import Persist
                    persist_inst = Persist(**params, per_variable=True)
                    local_result = persist_inst.fit_transform(subset)
                    local_states = persist_inst.get_states()
                elif method_name == "tid3":
                    if tid3_mv_inst is not None and params.get("multivariate_refinement", False):
                        # Use pre-computed multivariate TID3 (boundaries already refined)
                        local_result = tid3_mv_inst.transform(subset)
                        local_states = tid3_mv_inst.get_states()
                    else:
                        from .methods.tid3 import TID3
                        tid3_inst = TID3(**params)
                        tid3_inst.entity_class = self.entity_class
                        local_result = tid3_inst.fit_transform(subset)
                        local_states = tid3_inst.get_states()
                        # Accumulate per-variable phase 1 timing for non-MV TID3
                        if not hasattr(self, 'tid3_runtime_phase1_s'):
                            self.tid3_runtime_phase1_s = 0.0
                            self.tid3_runtime_phase2_s = 0.0
                        self.tid3_runtime_phase1_s += getattr(tid3_inst, 'runtime_phase1_s', 0.0)
                        self._tid3_scores_per_tpid.update(tid3_inst.final_scores_per_tpid)
                        self._tid3_cutoffs_per_tpid.update(tid3_inst.final_cutoffs_per_tpid)
                elif method_name == "mdlp":
                    from .methods.mdlp import MDLP
                    mdlp_inst = MDLP(**params, per_variable=True)
                    mdlp_inst.entity_class = self.entity_class
                    local_result = mdlp_inst.fit_transform(subset)
                    local_states = mdlp_inst.get_states()
                elif method_name == "knowledge":
                    # Knowledge method not in cache (dict-based, not CSV)
                    from .methods.knowledge import knowledge
                    local_result, local_states = knowledge(subset, **params, per_variable=True)
                else:
                    raise ValueError(f"Unsupported method: {method_name} for variable {tpid}")

                # Variable was root-skipped (e.g. TID3 min_mean_gap dropped it): no rows to process.
                if local_result.empty:
                    continue

                # Normal handling for all non-cached methods
                if isinstance(local_states, dict):
                    boundaries = local_states.get(tpid)
                else:
                    boundaries = local_states

                col_name = f"state_{method_name}"
                local_result = local_result.copy()
                local_result[col_name] = local_result.apply(
                    lambda row: self._map_local_state_ex(tpid, method_name, boundaries,
                                                          int(row["state"]), global_mapping, global_states_rows),
                    axis=1
                )
                local_global_states[method_name] = local_result[col_name]
            if not local_global_states:
                continue
            var_result = pd.concat(
                [subset.assign(StateID=local_global_states[m], MethodName=m)
                 for m in sorted(local_global_states.keys())],
                ignore_index=True
            )
            composite_results.append(var_result)
        final_train = pd.concat(composite_results, ignore_index=True)
        final_states = global_states_rows
        return final_train, final_states

    def _composite_transform(self, test_data: pd.DataFrame, method_config: dict):
        """
        Transform test data in composite mode using the same configurations as the training phase.
        Returns final_test: DataFrame with global StateID and a MethodName column.
        """
        composite_results = []
        default_config = method_config.get("default", None)
        unique_vars = test_data[TEMPORAL_PROPERTY_ID].unique()
        for tpid in unique_vars:
            subset = test_data[test_data[TEMPORAL_PROPERTY_ID] == tpid]
            if isinstance(method_config, dict) and tpid in method_config:
                cfgs = method_config[tpid]
            else:
                cfgs = method_config.get("default")
            if not isinstance(cfgs, list):
                cfgs = [cfgs]
            method_results = []
            for config in cfgs:
                method_name = config.get("method")
                params = config.copy()
                params.pop("method", None)
                if method_name == "equal_width":
                    from .methods.equal_width import equal_width
                    local_result, _ = equal_width(subset, **params, per_variable=True)
                elif method_name == "equal_frequency":
                    from .methods.equal_frequency import equal_frequency
                    local_result, _ = equal_frequency(subset, **params, per_variable=True)
                elif method_name == "sax":
                    from .methods.sax import sax
                    local_result, _ = sax(subset, **params, per_variable=True)
                elif method_name == "gradient":
                    from .methods.gradient import gradient
                    local_result, _ = gradient(subset, **params, per_variable=True)
                elif method_name == "td4c" or (isinstance(method_name, str) and method_name.startswith("td4c_")):
                    from .methods.td4c import TD4C
                    # Parse TD4C config if method name contains distance measure
                    td4c_params = params.copy()
                    if method_name.startswith("td4c_") and "distance_measure" not in td4c_params:
                        distance_measure = parse_td4c_config(method_name)
                        td4c_params["distance_measure"] = distance_measure
                    td4c_inst = TD4C(**td4c_params, per_variable=True)
                    local_result = td4c_inst.fit_transform(subset)
                    _ = td4c_inst.get_states()
                elif method_name == "persist":
                    from .methods.persist import Persist
                    persist_inst = Persist(**params, per_variable=True)
                    local_result = persist_inst.fit_transform(subset)
                    _ = persist_inst.get_states()
                elif method_name == "tid3":
                    from .methods.tid3 import TID3
                    tid3_inst = TID3(**params)
                    tid3_inst.entity_class = self.entity_class  # Pass entity-class mapping
                    local_result = tid3_inst.fit_transform(subset)
                    _ = tid3_inst.get_states()
                elif method_name == "mdlp":
                    from .methods.mdlp import MDLP
                    mdlp_inst = MDLP(**params, per_variable=True)
                    mdlp_inst.entity_class = self.entity_class  # Pass entity-class mapping
                    local_result = mdlp_inst.fit_transform(subset)
                    _ = mdlp_inst.get_states()
                elif method_name == "knowledge":
                    from .methods.knowledge import KnowledgeBased
                    local_result, _ = KnowledgeBased(**params, per_variable=True).fit_transform(subset), KnowledgeBased(**params, per_variable=True).get_states()
                else:
                    raise ValueError(f"Unsupported method: {method_name} for variable {tpid}")
                temp_mapping = {}
                temp_states_rows = []
                col_name = f"state_{method_name}"
                local_result = local_result.copy()
                local_result[col_name] = local_result.apply(
                    lambda row: self._map_local_state_ex(tpid, method_name, None, int(row["state"]),
                                                         temp_mapping, temp_states_rows),
                    axis=1
                )
                local_result = local_result.assign(MethodName=method_name)
                method_results.append(local_result[[TEMPORAL_PROPERTY_ID, "MethodName", col_name]])
            var_result = pd.concat(
                [subset.assign(StateID=mr[col_name], MethodName=mr["MethodName"])
                 for mr in method_results],
                ignore_index=True
            )
            composite_results.append(var_result)
        final_test = pd.concat(composite_results, ignore_index=True)
        return final_test

    def _map_local_state_ex(self, tpid, method_name, boundaries, local_state, global_mapping, global_states_rows):
        key = (tpid, method_name, local_state)
        if key not in global_mapping:
            num_bins = len(boundaries) + 1 if boundaries is not None else 1
            if local_state == -1:
                return -1
            if boundaries is None:
                bin_low = None
                bin_high = None
            else:
                if local_state == 1:
                    bin_low = -math.inf
                    bin_high = boundaries[0]
                elif local_state == num_bins:
                    bin_low = boundaries[-1]
                    bin_high = math.inf
                else:
                    bin_low = boundaries[local_state - 2]
                    bin_high = boundaries[local_state - 1]
            
            # Calculate next StateID: should be max(existing StateIDs) + 1
            if global_states_rows:
                max_state_id = max(row["StateID"] for row in global_states_rows)
                global_id = max_state_id + 1
            else:
                global_id = len(global_mapping) + 1
            
            global_mapping[key] = global_id
            global_states_rows.append({
                "StateID": global_id,
                "TemporalPropertyID": tpid,
                "MethodName": method_name,
                "BinId": local_state,
                "BinLow": round(bin_low, 3),
                "BinHigh": round(bin_high, 3),
            })
        return global_mapping[key]

    def _save_tid3_scores_csv(self, output_dir: str):
        """Write tid3_scores_summary.csv from accumulated composite-mode TID3 scores."""
        scores = getattr(self, '_tid3_scores_per_tpid', {})
        cutoffs = getattr(self, '_tid3_cutoffs_per_tpid', {})
        if not scores:
            return
        rows = []
        for tpid in sorted(scores.keys()):
            cuts = cutoffs.get(tpid, [])
            rows.append({
                'TemporalPropertyID': tpid,
                'NumCutoffs': len(cuts),
                'CutoffValues': str([round(c, 4) for c in cuts]),
                'FinalTStatScore': round(scores[tpid], 6),
            })
        rows.append({
            'TemporalPropertyID': 'TOTAL',
            'NumCutoffs': '-',
            'CutoffValues': '-',
            'FinalTStatScore': round(sum(scores.values()), 6),
        })
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'tid3_scores_summary.csv'), index=False)

    def _save_results(self, output_dir: str, symbolic_series: pd.DataFrame, states, max_gap: int):
        os.makedirs(output_dir, exist_ok=True)
        
        # Define a helper function for sorting keys:
        def sort_key(x):
            if isinstance(x, int):
                return (0, x)
            else:
                return (1, str(x))
        
        # Check if states is already a DataFrame (from knowledge method CSV)
        if isinstance(states, pd.DataFrame):
            # Knowledge method provided a CSV - save it as-is
            states_df = states.copy()
        elif isinstance(states, list) and states and "MethodName" in states[0]:
            # Composite mode with multiple methods
            states_df = pd.DataFrame(states)
        else:
            # Single method with dict boundaries
            global_mapping = {}
            states_rows = []
            for tpid in sorted(states.keys(), key=sort_key):
                boundaries = states[tpid]
                num_bins = len(boundaries) + 1
                for local_bin in range(1, num_bins + 1):
                    if local_bin == 1:
                        bin_low = -math.inf
                        bin_high = boundaries[0]
                    elif local_bin == num_bins:
                        bin_low = boundaries[-1]
                        bin_high = math.inf
                    else:
                        bin_low = boundaries[local_bin - 2]
                        bin_high = boundaries[local_bin - 1]
                    global_mapping[(tpid, local_bin)] = len(global_mapping) + 1
                    states_rows.append({
                        "StateID": global_mapping[(tpid, local_bin)],
                        "TemporalPropertyID": tpid,
                        "BinId": local_bin,
                        "BinLow": round(bin_low, 5),
                        "BinHigh": round(bin_high, 5),
                    })
            states_df = pd.DataFrame(states_rows)
        states_file = os.path.join(output_dir, "states.csv")
        states_df.to_csv(states_file, index=False)
        
        updated_series = symbolic_series.copy().reset_index(drop=True)
        if "MethodName" not in updated_series.columns:
            # Single method mode
            if isinstance(states, pd.DataFrame):
                # Knowledge method with CSV - state column already contains correct StateIDs
                updated_series = updated_series.rename(columns={"state": "StateID"})
            else:
                # Other methods with dict boundaries
                global_mapping = {}
                for tpid in sorted(states.keys(), key=sort_key):
                    boundaries = states[tpid]
                    num_bins = len(boundaries) + 1
                    for local_bin in range(1, num_bins + 1):
                        global_mapping[(tpid, local_bin)] = len(global_mapping) + 1
                def map_state(row):
                    tpid = row[TEMPORAL_PROPERTY_ID]
                    local_state = int(row["state"])
                    return global_mapping.get((tpid, local_state), local_state)
                updated_series["state"] = updated_series.apply(map_state, axis=1)
                updated_series = updated_series.rename(columns={"state": "StateID"})
        symbolic_file = os.path.join(output_dir, "symbolic_time_series.csv")
        updated_series.to_csv(symbolic_file, index=False)
        
        kl_content = generate_KL_content(updated_series, max_gap)
        kl_file = os.path.join(output_dir, "KL.txt")
        with open(kl_file, "w") as f:
            f.write(kl_content)
        
        if self.entity_class:
            updated_series["EntityClass"] = updated_series[ENTITY_ID].map(self.entity_class)
            for cls in sorted(set(self.entity_class.values())):
                subset = updated_series[updated_series["EntityClass"] == cls]
                kl_content_cls = generate_KL_content(subset, max_gap)
                kl_file_cls = os.path.join(output_dir, f"KL-class-{float(cls)}.txt")
                with open(kl_file_cls, "w") as f:
                    f.write(kl_content_cls)

        save_entity_ids(self.entity_class, output_dir)

        print(f"Results saved in directory: {output_dir}")