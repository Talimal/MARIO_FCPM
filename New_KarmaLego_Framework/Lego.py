import multiprocessing as mp

import pandas as pd

from New_KarmaLego_Framework.Tirp_new import TIRP


class Lego(object):
    def __init__(self, karma, label, incremental_output=False, path=None, max_tirp_length=10, need_one_sized=False,
                 print_instances=True):
        self._karma = karma
        self.frequent_tirps = []
        self._max_tirp_length = max_tirp_length
        # self.tirps_tree_root = LegoTreeNode(None)
        self.path = path
        self.incremental_output = incremental_output
        self.need_one_sized = need_one_sized
        self.print_instances = print_instances
        self.label = label

    def fit(self, index_same=False, skip_followers=False, processes_num=None):
        """
        creating two-sized tirps and send them to extend-tirp
        finally all frequent tirps are created
        :return: None
        """
        if processes_num:
            # mp.freeze_support()
            self.processes_num = processes_num
            print(self.processes_num)
            # Use context manager for better resource management
            with mp.Pool(processes=processes_num) as pool:
                nodes_processes = []
                for symbol1 in self._karma.get_symbols_map().keys():
                    if self._karma.get_symbol_vertical_support(symbol1) < self._karma.get_min_vertical_support():
                        continue
                    if self.need_one_sized:
                        instances = self._karma.get_one_size_tirp_instances(symbol1)
                        one_sized_tirp = TIRP.get_one_sized_tirp(symbol1, instances)
                        self.frequent_tirps.append(one_sized_tirp)
                    
                    for symbol2 in self._karma.get_symbols_map().keys():
                        for rel in range(self._karma.num_relations):
                            if self._karma.get_vertical_support_of_sym_to_sym_rel(
                                    symbol1, symbol2, rel) < \
                                    self._karma.get_min_vertical_support():
                                continue
                            if not index_same and symbol1 == symbol2:
                                continue
                            tirp = self._karma.get_two_sized_tirp_for_symbols_and_rel(symbol1, symbol2, rel)
                            nodes_processes.append((tirp, index_same, skip_followers))

                # Dynamic chunk size based on available memory and work complexity
                optimal_chunk_size = max(1, min(len(nodes_processes) // (processes_num * 2), 50))
                
                # Process in chunks to reduce memory pressure
                results = pool.starmap(self._extend_tirp_wrapper, nodes_processes, chunksize=optimal_chunk_size)
                
                # Collect results efficiently
                for result_list in results:
                    if result_list:
                        self.frequent_tirps.extend(result_list)
        else:
            # Sequential processing
            for symbol1 in self._karma.get_symbols_map().keys():
                if self._karma.get_symbol_vertical_support(symbol1) < self._karma.get_min_vertical_support():
                    continue
                if self.need_one_sized:
                    instances = self._karma.get_one_size_tirp_instances(symbol1)
                    one_sized_tirp = TIRP.get_one_sized_tirp(symbol1, instances)
                    self.frequent_tirps.append(one_sized_tirp)
                
                for symbol2 in self._karma.get_symbols_map().keys():
                    for rel in range(self._karma.num_relations):
                        if self._karma.get_vertical_support_of_sym_to_sym_rel(
                                symbol1, symbol2, rel) < \
                                self._karma.get_min_vertical_support():
                            continue
                        if not index_same and symbol1 == symbol2:
                            continue
                        tirp = self._karma.get_two_sized_tirp_for_symbols_and_rel(symbol1, symbol2, rel)
                        self.extend_tirp(tirp, index_same, skip_followers)
        print('Lego fit complete')

    def _extend_tirp_wrapper(self, tirp, index_same, skip_followers):
        """
        Wrapper method for multiprocessing to handle memory management better
        """
        try:
            # Clear any local variables to reduce memory footprint
            found_tirps = self.extend_tirp(tirp, index_same, skip_followers)
            return found_tirps
        except Exception as e:
            print(f"Error processing TIRP: {e}")
            return []

    def extend_tirp(self, tirp, index_same, skip_followers):
        """
         extending recursively the tirp and appending the frequent tirps to self.frequent_tirps
         :param tirp: TIRP, the tirp to extend
         :param index_same: Boolean, prevents multiple instances of same symbol in a tirp if true
         :param skip_followers: Boolean, prevents multiple instances of same var in a tirp if true
         :return: List of frequent TIRPs found
        """
        # root_node = LegoTreeNode(tirp)
        found_tirps = [tirp]  # T - collect TIRPs found in this call
        
        # Only append to frequent_tirps if running sequentially (not in multiprocessing)
        if not hasattr(self, 'processes_num') or self.processes_num is None:
            self.frequent_tirps.append(tirp)
        
        for symbol in self._karma.get_symbols_map().keys():
            if not index_same and symbol in tirp._symbols:
                continue
            for seed_relation in range(self._karma.num_relations):
                if (self._karma.get_vertical_support_of_sym_to_sym_rel(
                        tirp._symbols[tirp.size - 1], symbol,
                        seed_relation) >= self._karma.get_min_vertical_support()):
                    candidates = self.generate_candidates(tirp, seed_relation)
                    for candidate_relations in candidates:
                        new_tirp = self.extend_single_tirp(tirp, symbol, candidate_relations, skip_followers)
                        if new_tirp.get_vertical_support() >= self._karma.get_min_vertical_support():
                            # new_tirp._support_discovery = round(
                            #     len(new_tirp._supporting_sequences_by_entity) / len(self._karma._entities), 2)
                            # if self.incremental_output:
                            #     tirp.print_tirp(self.path,
                            #                         self._karma.get_entities_vector(), self._karma._num_relations,
                            #                         Karma.CALC_OFFSETS, self._karma.entities_times,
                            #                         self.print_instances)
                            if new_tirp.size < self._max_tirp_length:
                                recursive_tirps = self.extend_tirp(new_tirp, index_same, skip_followers)
                                found_tirps.extend(recursive_tirps)

        # tirp_file_name = tirp.get_tirp_file_name(self._karma._num_relations)
        # tirp.print_tirp(self.path,
        #                     self._karma.get_entities_vector(), self._karma._num_relations,
        #                     Karma.CALC_OFFSETS, self._karma.entities_times,
        #                     self.print_instances)
        return found_tirps

    def generate_candidates(self, tirp, seed_relation):
        column_size = tirp.size
        top_cnd_rel_index = 0
        btm_rel_index = column_size - 2
        candidates_list = []
        candidate = []
        for i in (range(column_size)):
            candidate.append(i)
        candidate[column_size - 1] = seed_relation
        candidates_list.append(candidate)
        rng = list(range(top_cnd_rel_index, btm_rel_index + 1))
        rng.reverse()
        for rel_index_to_set in rng:
            left_tirp_index = int(((rel_index_to_set + 1) * rel_index_to_set / 2) + rel_index_to_set)
            below_cnd_index = rel_index_to_set + 1
            cand_list_size = len(candidates_list)
            for cand_index in range(cand_list_size):
                candidate = candidates_list[cand_index]
                transitivity_list = self._karma._relation_handler_obj. \
                    get_transitivity_list(tirp._tirp_matrix.get_relations()[left_tirp_index],
                                          candidate[below_cnd_index])
                for rel in transitivity_list:
                    if rel > transitivity_list[0]:
                        new_candidate = []
                        for i in (range(column_size)):
                            new_candidate.append(i)
                        tmp_rng = list(range(rel_index_to_set + 1, column_size))
                        tmp_rng.reverse()
                        for r_index in tmp_rng:
                            new_candidate[r_index] = candidate[r_index]
                        new_candidate[rel_index_to_set] = rel
                        candidates_list.append(new_candidate)
                    else:
                        candidate[rel_index_to_set] = rel
        return candidates_list


    def join_with_relationship_table(self, base_table, relationship_table, base_symbol, new_symbol, how=''):
        """
        Memory-efficient join between base TIRP table and relationship table.
        """
        # Early exit for empty tables to save memory
        if base_table.empty or relationship_table.empty:
            return None
        
        # Use categorical data types for memory efficiency if strings are involved
        if base_table[str(base_symbol)].dtype == 'object':
            base_table = base_table.copy()
            base_table[str(base_symbol)] = base_table[str(base_symbol)].astype('category')
        
        if relationship_table[str(base_symbol)].dtype == 'object':
            relationship_table = relationship_table.copy()
            relationship_table[str(base_symbol)] = relationship_table[str(base_symbol)].astype('category')
        
        # Perform memory-efficient join
        if how == 'left':
            merged = base_table.merge(
                relationship_table,
                on=["EntityID", str(base_symbol)],
                how=how,
                copy=False  # Avoid unnecessary copying
            )
        else:
            merged = base_table.merge(
                relationship_table,
                on=["EntityID", str(base_symbol), str(new_symbol)],
                how='inner',
                copy=False  # Avoid unnecessary copying
            )

        # Early memory check - exit if insufficient entities
        entity_support_count = merged["EntityID"].nunique()
        if entity_support_count < self._karma.get_min_vertical_support():
            del merged  # Explicitly free memory
            return None

        return merged

    def extend_single_tirp(self, base_tirp, new_symbol, new_relations, skip_followers):
        """
        Memory-efficient TIRP extension using optimized DataFrame operations.
        """
        # Create empty DataFrame template for early returns
        empty_columns = ["EntityID"] + base_tirp._symbols + [new_symbol]
        empty_df = pd.DataFrame(columns=empty_columns)

        # Pre-validate conditions to avoid unnecessary processing
        join_info = []
        for i, existing_symbol in enumerate(base_tirp._symbols):
            if skip_followers:
                abs_difference = abs(int(new_symbol) - int(existing_symbol))
                same_variable = (
                        self._karma.symbol_to_variable_map.get(new_symbol) ==
                        self._karma.symbol_to_variable_map.get(existing_symbol)
                )
                if abs_difference == 1 and same_variable:
                    return base_tirp.extend_tirp(new_symbol, new_relations, empty_df)

            relation = new_relations[i]
            
            # Early validation of vertical support
            if (self._karma.get_vertical_support_of_sym_to_sym_rel(existing_symbol, new_symbol, relation) < 
                    self._karma.get_min_vertical_support()):
                return base_tirp.extend_tirp(new_symbol, new_relations, empty_df)

            # Get relationship table
            relationship_table = self._karma.get_instances_table(existing_symbol, new_symbol, relation)
            if relationship_table is None or relationship_table.empty:
                return base_tirp.extend_tirp(new_symbol, new_relations, empty_df)

            join_info.append({
                "existing_symbol": existing_symbol,
                "relationship_table": relationship_table,
                "relation": relation,
                "size": len(relationship_table)
            })

        # Sort by table size for optimal join order (smallest first)
        join_info.sort(key=lambda x: x["size"])

        # Perform joins with memory management
        current_table = base_tirp.instances
        
        for i, info in enumerate(join_info):
            relationship_table = info["relationship_table"]
            existing_symbol = info["existing_symbol"]
            
            # Use left join only for first join, then inner joins
            join_type = 'left' if i == 0 else 'inner'
            
            # Perform the join
            new_table = self.join_with_relationship_table(
                current_table, relationship_table, existing_symbol, 
                new_symbol=new_symbol, how=join_type
            )
            
            if new_table is None:
                return base_tirp.extend_tirp(new_symbol, new_relations, empty_df)
            
            # Memory management: delete previous table if it's not the original
            if i > 0 and current_table is not base_tirp.instances:
                del current_table
            
            current_table = new_table

        # Create and return new TIRP
        return base_tirp.extend_tirp(new_symbol, new_relations, current_table)


    def print_frequent_tirps(self, path):
        """
        printing all the frequent TIRPs into a file
         :param path: the output path to print the TIRP
        :return: None - a file with the TIRPs
        """
        for tirp, i in zip(self.frequent_tirps, range(len(self.frequent_tirps))):
            # st = time.time_ns()
            # tirp_file_name = tirp.get_tirp_file_name(self._karma._num_relations)
            # if i+1 < len(self.frequent_tirps) and \
            #         os.path.exists(path + self.frequent_tirps[i+1].get_tirp_file_name(self._karma._num_relations)):
            #     continue
            tirp.print_tirp(path, self._karma.num_relations)
            # print('Finish Writing TIRP: ({}) {}. Time: {} min'.format(i, tirp_file_name, g.get_time_passed(st)))
