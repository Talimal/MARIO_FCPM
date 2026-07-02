from SKL.Karma_new import Karma
from SKL.Tirp_new import TIRP
from copy import copy
import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import groupby


def process_group(group, detector):
    """
    Process a group of TIRPs using the given detector.
    :param group: List of TIRPs in the group.
    :param detector: TIRPDetector object to process the group.
    :return: List of processed TIRPs.
    """
    res = [detector.process_single_tirp(tirp) for tirp in group]
    return res

class TIRPDetector:
    def __init__(self, time_intervals_path, num_relations, max_gap, epsilon=0, output_path=None, print_instances=True,one_size_tirp=False):
        """
        Initialize the TIRPDetector with paths, parameters, and configurations.

        :param time_intervals_path: Path to the temporal data file.
        :param relationship_mode: Number of relationships to consider (3 or 7).
        :param max_gap: Maximum allowed gap between intervals.
        :param epsilon: Allowable difference between intervals to still be considered overlapping.
        :param output_path: Path to save output results.
        :param print_instances: If true, print instances of the TIRPs.
        :param one_size_tirp: Whether to include one-size TIRPs (default is False).
        """
        self.time_intervals_path = time_intervals_path
        self.num_relations = num_relations
        self.max_gap = max_gap
        self.epsilon = epsilon
        self.output_path = output_path
        self.print_instances = print_instances
        self.karma = Karma(num_relations=num_relations, epsilon=epsilon, max_gap=max_gap)
        self.expanded_tirps = {}
        self.one_size_tirp = one_size_tirp

    def get_relevant_symbols_from_tirps(self, input_tirps):
        """
        Extracts a set of unique symbols used within the input TIRPs.

        :param input_tirps: List of TIRP objects to filter symbols from.
        :return: Set of relevant symbols found in the TIRPs.
        """
        relevant_symbols = set()
        for tirp in input_tirps:
            for symbol in tirp._symbols:
                relevant_symbols.add(symbol)
                relevant_symbols.add(str(symbol))
        return relevant_symbols

    def get_relevant_relations_from_tirps(self, input_tirps):
        """
        Extract a list of relevant relations from the input TIRPs.
        Each relation is represented as a key in the format: 'symbol1_symbol2_relation'.

        :param input_tirps: List of TIRP objects.
        :return: Set of relevant relation keys.
        """
        relevant_relations = set()
        for tirp in input_tirps:
            symbols = tirp._symbols
            relations = tirp._tirp_matrix.get_relations()

            for i in range(len(symbols)):
                for j in range(0, i):
                    row_index = j
                    column_index = i - 1

                    # Calculate relation index in the half-matrix
                    relation_index = int(((1 + column_index) * column_index) / 2 + row_index)
                    relation_name = self.karma.relationship_mapping[int(relations[relation_index])]
                    key = f"{symbols[j]}_{symbols[i]}_{relation_name}"
                    # Initialize DataFrame if key is not present
                    if key not in self.karma.index:
                        self.karma.index[key] = pd.DataFrame(columns=["EntityID", str(symbols[j]), str(symbols[i])])

        return relevant_relations


    def initialize_karma(self, selected_symbols=None):
        """
        Initialize the Karma object and index the time intervals data.
        :param selected_symbols: List of symbols to filter.
        """
        self.karma.run_karma(self.time_intervals_path, selected_symbols=selected_symbols)

    def join_with_relationship_table(self, base_table, relationship_table, base_symbol,new_symbol, how=''):
        """
        Perform a join between the base TIRP table and a relationship table on EntityID and base_symbol.
        :param base_table: DataFrame of the current TIRP's instances.
        :param relationship_table: DataFrame for the relationship between two symbols.
        :param base_symbol: The symbol from the base TIRP involved in the relationship.
        :param new_symbol: The new symbol being added to the TIRP.
        :return: DataFrame with the joined result.
        """
        # Perform the join on EntityID and base_symbol
        if how == 'left':
            merged = base_table.merge(
                relationship_table,
                on=["EntityID", str(base_symbol)],how=how
            )
        else:
            merged = base_table.merge(
                relationship_table,
                on=["EntityID", str(base_symbol),str(new_symbol)],how='inner'
            )
        return merged

    def extend_tirp(self, base_tirp, new_symbol, new_relations):
        """
        Extend a TIRP by joining its table with the relationship tables for the new symbol.
        :param base_tirp: Current TIRP object being extended.
        :param new_symbol: The new symbol being added to the TIRP.
        :param new_relations: List of relationships between the new symbol and existing symbols.
        :return: Extended TIRP object.
        """
        tirp_key = (tuple(base_tirp._symbols + [new_symbol]), tuple(base_tirp._tirp_matrix.get_relations() + new_relations))

        # Check if TIRP has already been expanded
        if tirp_key in self.expanded_tirps:
            return self.expanded_tirps[tirp_key]

        base_table = base_tirp.instances

        for i, existing_symbol in enumerate(base_tirp._symbols):
            relation = new_relations[i]
            relationship_table = self.karma.get_instances_table(existing_symbol, new_symbol, relation)

            if relationship_table is None:
                return None

            if i == 0:
                base_table = self.join_with_relationship_table(
                    base_table, relationship_table, existing_symbol, new_symbol=new_symbol ,how='left'
                )
            else:
                base_table = self.join_with_relationship_table(
                    base_table, relationship_table, existing_symbol, new_symbol=new_symbol
                )

        # Create the new TIRP and save it
        new_tirp = base_tirp.extend_tirp(new_symbol,new_relations,base_table)
        self.expanded_tirps[tirp_key] = new_tirp

        return new_tirp

    def process_single_tirp(self, input_tirp):
        """
        Process a single TIRP by expanding it iteratively and detecting its instances.
        :param input_tirp: TIRP object to process.
        :return: The fully expanded TIRP with detected instances.
        """
        if self.one_size_tirp and input_tirp.size == 1:
            symbol = input_tirp._symbols[0]
            instances = self.karma.get_one_size_tirp_instances(symbol)
            current_tirp = TIRP.get_one_sized_tirp(symbol,instances)
            return current_tirp

        decomposition = self.decompose_tirp(input_tirp)
        initial_symbols, initial_relation = decomposition[0]
        initial_table = self.karma.get_instances_table(initial_symbols[0], initial_symbols[1], initial_relation)
        current_tirp = TIRP(initial_symbols[0], initial_symbols[1], initial_relation, initial_table)

        for new_symbol, new_relations in decomposition[1:]:
            current_tirp = self.extend_tirp(current_tirp, new_symbol, new_relations)
            if current_tirp is None:
                break

        return current_tirp

    def run_detection(self, input_tirps, use_parallel=False):
        """
        Detect instances for a list of input TIRPs.
        :param input_tirps: List of TIRP objects to process.
        :param use_parallel: Whether to use multiprocessing for detection.
        :return: List of detected TIRPs.
        """
        # Extract relevant symbols and initialize Karma
        self.get_relevant_relations_from_tirps(input_tirps)
        relevant_symbols = self.get_relevant_symbols_from_tirps(input_tirps)
        self.initialize_karma(selected_symbols=relevant_symbols)

        # Group TIRPs by their first symbol
        sorted_tirps = sorted(input_tirps, key=lambda tirp: tirp._symbols[0])
        grouped_tirps = {key: list(group) for key, group in groupby(sorted_tirps, key=lambda tirp: tirp._symbols[0])}

        if self.one_size_tirp:
            # Add one-size TIRP to each group
            for symbol in grouped_tirps.keys():
                one_size_tirp = TIRP.get_one_sized_tirp(new_symbol=symbol)  # Create one-size TIRP
                grouped_tirps[symbol].insert(0, one_size_tirp)  # Insert at the beginning of the group

        results = []
        if use_parallel:
            with Pool(cpu_count()) as pool:
                group_results = pool.starmap(process_group, [(group, self) for group in grouped_tirps.values()])
                for group in group_results:
                    results.extend(group)
        else:
            for group in grouped_tirps.values():
                results.extend(process_group(group, self))

        # Save results
        if self.output_path and self.print_instances:
            with open(self.output_path, "w") as f:
                for tirp in results:
                    f.write(str(tirp) + "\n")

        # print("Detection complete. Results saved to:", self.output_path)
        return results

    def decompose_tirp(self, input_tirp):
        """
        Decomposes an input TIRP into its initial size-2 TIRP and a series of expansions.
        :param input_tirp: The full TIRP to decompose.
        :return: List of tuples, each containing a symbol and relations for sequential expansion.
        """
        decomposed_tirp = []
        # Start with the first two symbols and their relation as the initial size-2 TIRP
        initial_symbols = input_tirp._symbols[:2]
        initial_relation = input_tirp._tirp_matrix.get_relation(0, 1)
        decomposed_tirp.append((initial_symbols, initial_relation))

        # Sequentially add each symbol and its relations for expansion
        for i in range(2, len(input_tirp._symbols)):
            column_index = i - 1
            offset = int(((1 + column_index) * column_index) / 2)
            new_symbol = input_tirp._symbols[i]
            relations = input_tirp._tirp_matrix.get_relations()[offset:offset + i]
            decomposed_tirp.append((new_symbol, relations))
        # print(decomposed_tirp)

        return decomposed_tirp
