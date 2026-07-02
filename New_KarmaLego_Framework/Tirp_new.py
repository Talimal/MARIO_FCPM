from copy import copy
from typing import Tuple
import pickle
import numpy as np 

import networkx as nx
import pandas as pd

from New_KarmaLego_Framework.RelationHandler import RelationHandler


class TIRP:
    def __init__(self, first_symbol=None, second_symbol=None, relation=None, instances=None):
        if first_symbol is not None and second_symbol is not None and relation is not None:
            self.size = 2
            self._symbols = [first_symbol, second_symbol]
            self._tirp_matrix = TirpMatrix(relation)
        else:
            self.size = 0
            self._symbols = []
            self._tirp_matrix = {}

        self.instances = instances if instances is not None else pd.DataFrame()

    def extend_tirp(self, new_symbol, new_relations, instances):
        new_tirp = self.copy_tirp()
        new_tirp._symbols.append(new_symbol)
        new_tirp._tirp_matrix.extend(new_relations)
        new_tirp.size = self.size + 1
        new_tirp.instances = instances
        return new_tirp

    @staticmethod
    def get_one_sized_tirp(new_symbol, instances=None):
        new_tirp = TIRP()
        new_tirp._symbols = [new_symbol]
        new_tirp._tirp_matrix = TirpMatrix()
        new_tirp.size = 1
        new_tirp.instances = instances if instances is not None else pd.DataFrame()
        return new_tirp

    def compute_tiep_order(self, relations_type=7):
        """
        Returns a list of tuples, where each tuple is a set of coinciding endpoints
        in the format "symbol+" or "symbol-". If multiple endpoints share a time,
        they appear in the same tuple.

        tirp: Object that has:
            - tirp._symbols (list of symbols, which can be strings or ints)
            - tirp._tirp_matrix (an object implementing get_relation(i, j))

        relations_type: integer, either 3 or 7, to decide which relation_mapping to use.
        """

        # relation mapping, can be adapted as needed
        relation_mapping = {
            3: {"before": 0, "overlap": 1, "contain": 2},
            7: {
                "before": 0,
                "meet": 1,
                "overlap": 2,
                "finishby": 3,
                "contain": 4,
                "starts": 5,
                "equal": 6
            }
        }
        rel_map = relation_mapping[relations_type]
        inv_map = {v: k for k, v in rel_map.items()}

        symbols = self._symbols
        tmatrix = self._tirp_matrix

        G = nx.DiGraph()

        # Create endpoints as (symbol, '+') or (symbol, '-')
        endpoints = [(sym, '+') for sym in symbols] + [(sym, '-') for sym in symbols]
        G.add_nodes_from(endpoints)

        # Simple union-find to unify endpoints under 'meet'/'equal'
        parent = {ep: ep for ep in endpoints}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                for k, v in list(parent.items()):
                    if v == rb:
                        parent[k] = ra
                parent[rb] = ra

        def add_edge(a, b):
            A, B = find(a), find(b)
            if A != B:
                G.add_edge(A, B)

        # (Optional) Enforce start <= end for each symbol
        # If you want each symbol's start to be before its end:
        for sym in symbols:
            add_edge((sym, '+'), (sym, '-'))

        # Build constraints from the TIRP relations
        for i in range(len(symbols) - 1):
            for j in range(i + 1, len(symbols)):
                rel_id = tmatrix.get_relation(i, j)
                rel_name = inv_map[rel_id]

                Aplus, Aminus = (symbols[i], '+'), (symbols[i], '-')
                Bplus, Bminus = (symbols[j], '+'), (symbols[j], '-')

                if rel_name == 'before':
                    add_edge(Aminus, Bplus)
                elif rel_name == 'meet':
                    union(Aminus, Bplus)
                elif rel_name == 'overlap':
                    add_edge(Aplus, Bplus)
                    add_edge(Bplus, Aminus)
                    add_edge(Aminus, Bminus)
                elif rel_name == 'finishby':
                    union(Aminus, Bminus)
                    add_edge(Aplus, Bplus)
                elif rel_name == 'contain':
                    add_edge(Aplus, Bplus)
                    add_edge(Bminus, Aminus)
                elif rel_name == 'starts':
                    union(Aplus, Bplus)
                    add_edge(Aminus, Bminus)
                elif rel_name == 'equal':
                    union(Aplus, Bplus)
                    union(Aminus, Bminus)

        # Group endpoints by their representative
        reps = {}
        for ep in endpoints:
            reps.setdefault(find(ep), set()).add(ep)

        # Build a compressed graph
        CG = nx.DiGraph()
        for rep in reps:
            CG.add_node(rep)
        for u, v in G.edges():
            ru, rv = find(u), find(v)
            if ru != rv:
                CG.add_edge(ru, rv)

        # Topological sort on the compressed graph
        topo = list(nx.topological_sort(CG))

        # # Transform each representative node into a tuple of string endpoints like "10+" or "A-"
        # # sorted by the endpoint string for consistency
        symbol_to_index = {symbol: i for i, symbol in enumerate(self._symbols)}

        result = []
        for r_rep in topo: # r_rep is a representative endpoint tuple e.g. ('A', '+')
            # Get all original endpoints that belong to this representative's equivalence class
            # This was stored in reps_map
            group_endpoints = reps[r_rep]
            
            # Sort the endpoints within the group:
            # Primary key: original index of the symbol in self._symbols
            # Secondary key: sign ('+' or '-'), '+' usually comes before '-' in string sort
            # but it's ('symbol', '+') vs ('symbol', '-'), so x[1] is the sign.
            # Standard sort: '-' < '+'
            group = sorted(list(group_endpoints), key=lambda x: (symbol_to_index[x[0]], x[1]))
            
            group_strings = tuple(str(sym_val) + sign for (sym_val, sign) in group)
            result.append(group_strings)

        return result

    def get_vertical_support(self):
        """
        Calculate the vertical support of the TIRP.

        Vertical support is the number of unique entities that have at least one instance of the TIRP.

        :return: The number of entities with at least one instance of the TIRP.
        """
        if self.instances.empty:
            return 0

        # Count the unique entities in the 'EntityID' column
        vertical_support = self.instances["EntityID"].nunique()

        return vertical_support

    def calculate_mean_horizontal_support(self):
        """
        Calculate the mean horizontal support based on the instances.
        Horizontal support is defined as the number of occurrences of the TIRP in an entity.
        Mean horizontal support is the average number of repetitions among entities where the TIRP appears.
        """
        if self.instances.empty:
            self.mean_horizontal_support = 0.0
            return self.mean_horizontal_support

        # Count occurrences of the TIRP per entity
        occurrences_per_entity = self.instances['EntityID'].value_counts()

        # Calculate the mean for entities with at least one occurrence
        self.mean_horizontal_support = occurrences_per_entity.mean()
        return self.mean_horizontal_support

    def get_horizontal_support_by_entity(self, entity_id):
        """
        Calculate the horizontal support for a specific entity.
        Horizontal support is the number of occurrences of the entity in the instances.

        :param entity_id: The ID of the entity to calculate the horizontal support for.
        :return: The horizontal support of the given entity.
        """
        if self.instances.empty:
            return 0

        # Filter instances for the given entity ID
        entity_instances = self.instances[self.instances['EntityID'] == entity_id]

        # Horizontal support is the number of occurrences for this entity
        return len(entity_instances)


    def calculate_mean_mean_duration(self):
        """
        Calculate the mean continuation duration across all entities efficiently using Pandas.
        Continuation duration is calculated as the total span (from the start of the first interval
        to the end of the last interval) for each entity's supporting instances.
        """
        if self.instances.empty:
            self.mean_mean_duration = 0.0
            return 0.0

        # Get the list of columns corresponding to symbols (assuming they are column names)
        # Convert symbols to strings if they are column names
        symbol_cols = [str(s) for s in self._symbols]

        # Define a function to calculate duration for a single row (instance)
        def _calculate_instance_duration(row):
            # Select only the interval data for this row
            intervals = [row[col] for col in symbol_cols if isinstance(row[col], tuple)]
            
            if not intervals:
                return np.nan # Return Not a Number if no valid intervals found

            start_times = [interval[0] for interval in intervals]
            end_times = [interval[1] for interval in intervals]

            if not start_times or not end_times:
                 return np.nan # Should not happen if intervals is not empty, but safe check

            return max(end_times) - min(start_times)

        # Apply the function row-wise to calculate duration for each instance
        # Create a temporary column 'instance_duration'
        self.instances['instance_duration'] = self.instances.apply(_calculate_instance_duration, axis=1)

        # Calculate the mean duration per entity
        # This results in a Series where the index is EntityID and value is the mean duration for that entity
        mean_duration_per_entity = self.instances.groupby('EntityID')['instance_duration'].mean()

        # Calculate the mean of these entity-specific mean durations
        # Use skipna=True to ignore entities that might have had NaN durations (if any)
        overall_mean_duration = mean_duration_per_entity.mean(skipna=True)

        # Clean up the temporary column
        self.instances.drop(columns=['instance_duration'], inplace=True)

        # Handle case where no entities had valid durations
        if pd.isna(overall_mean_duration):
             overall_mean_duration = 0.0

        self.mean_mean_duration = overall_mean_duration
        return self.mean_mean_duration

    def get_mean_duration_by_entity(self, entity_id):
        """
        Calculate the mean continuation duration for a specific entity.
        Continuation duration is calculated as the total span (from the start of the first interval
        to the end of the last interval) for each supporting instance of the specified entity.

        :param entity_id: The ID of the entity for which to calculate the mean duration.
        :return: The mean duration for the given entity.
        """
        if self.instances.empty:
            return 0.0

        # Filter instances for the specific entity
        entity_instances = self.instances[self.instances['EntityID'] == entity_id]

        if entity_instances.empty:
            return 0.0

        # Process supporting instances for the entity
        durations = []
        for _, row in entity_instances.iterrows():
            start_times = [interval[0] for interval in row[1:] if isinstance(interval, tuple)]
            end_times = [interval[1] for interval in row[1:] if isinstance(interval, tuple)]
            if start_times and end_times:
                durations.append(max(end_times) - min(start_times))

        # Calculate and return the mean duration for the entity
        if durations:
            return sum(durations) / len(durations)
        return 0.0

    def calculate_feature_matrices_score_by_entity(self, entity_id):
        """
        Calculate feature matrices scores for a specific entity.

        :param entity_id: The ID of the entity for which to calculate the scores.
        :return: A tuple containing BIN, HS, and MD:
            - BIN: Binary variable indicating whether the TIRP appears in the entity (1 if present, 0 otherwise).
            - HS: Horizontal support value for the entity.
            - MD: Mean duration for the entity.
        """
        # Check if the TIRP appears in the entity
        entity_instances = self.instances[self.instances['EntityID'] == entity_id]
        BIN = 1 if not entity_instances.empty else 0

        # Calculate horizontal support
        HS = self.get_horizontal_support_by_entity(entity_id)

        # Calculate mean duration
        MD = self.get_mean_duration_by_entity(entity_id)

        return BIN, HS, MD

    def copy_tirp(self):
        """
        Create new TIRP and copy without instances.
        :return: TIRP, copy of this TIRP.
        """
        new_tirp = TIRP()
        new_tirp.size = self.size
        new_tirp._symbols = copy(self._symbols)
        new_tirp._tirp_matrix = self._tirp_matrix.copy()
        return new_tirp

    def to_string(self):
        ans = f'{self.size}-'
        for symbol in self._symbols:
            ans += str(symbol) + '_'
        ans += self._tirp_matrix.to_string()
        return ans

    def print_tirp(self, path, num_relations):
        """
        Converts the TIRP instances to the desired custom output format.

        Returns:
            str: A formatted string with intervals in the desired format.
        """

        rel_object = RelationHandler(num_relations)
        tirp_string = str(len(self._symbols))
        tirp_string = tirp_string + " "
        for sym in self._symbols:
            tirp_string = tirp_string + str(sym) + "-"
        tirp_string = tirp_string + " "
        for rel in self._tirp_matrix.get_relations():
            tirp_string = tirp_string + rel_object.get_short_description(rel) + "."
        tirp_string = tirp_string + " "
        tirp_string = tirp_string + str(self.get_vertical_support()) + " "
        tirp_string = tirp_string + str(self.calculate_mean_horizontal_support()) + " "
        formatted_output = [tirp_string]

        for _, row in self.instances.iterrows():
            entity_id = row['EntityID']
            intervals = []

            # Loop through the symbols to extract intervals
            for symbol in self._symbols:
                symbol = str(symbol)
                if symbol in row:
                    start, end = row[symbol]
                    intervals.append(f"[{start}-{end}]")

            # Combine entity ID and intervals
            formatted_output.append(f"{entity_id} {''.join(intervals)}")

        tirp_string = ' '.join(formatted_output)
        with open(path, 'a') as output_file:
            output_file.write(tirp_string + "\n")

        # return ' '.join(formatted_output)

    def save_tirp_to_csv(self, folder_path):
        """
        Save the TIRP to a CSV file.

        :param folder_path: Path to the folder where the TIRP instances should be saved as a CSV.
        """
        if self.instances.empty:
            print(f"for {self.to_string()} No instances to save.")
            return

        # Generate the filename based on the TIRP's string representation
        file_name = f"{self.to_string()}.csv".replace(" ", "_").replace(":", "-")
        file_path = f"{folder_path}/{file_name}"

        # Save the instances to the CSV file
        self.instances.to_csv(file_path, index=False, header=True)
        # print(f"TIRP saved to {file_path}")

    def save_tirp_object(self, folder_path):
        """
        Save a copy of the TIRP object without instances but including tiep_representation.

        :param folder_path: Path to the folder where the TIRP object should be saved.
        """

        # Create a copy of the TIRP without instances
        new_tirp = self.copy_tirp()

        # Generate the filename based on the TIRP's string representation
        file_name = f"{self.to_string()}.pkl".replace(" ", "_").replace(":", "-")
        file_path = f"{folder_path}/{file_name}"

        # Save the copied TIRP object to a Pickle file
        with open(file_path, "wb") as file:
            pickle.dump(new_tirp, file)

        # print(f"TIRP object saved successfully to {file_path}")

    @staticmethod
    def load_tirp_from_csv(file_path: str, epsilon: int, max_gap: int, num_relations: int):
        """
        Load a TIRP object from a CSV file.

        :param file_path: Path to the CSV file containing the TIRP's instances.
        :param epsilon: Allowable difference between intervals to still be considered overlapping.
        :param max_gap: Maximum allowable gap between intervals.
        :param num_relations: Specifies whether to compute relationships using 3 or 7 relationship modes.
        :return: A TIRP object reconstructed from the CSV file.
        """
        # Read the CSV file
        instances = pd.read_csv(file_path)

        # Ensure EntityID column is of type 'object'
        if "EntityID" in instances.columns:
            instances["EntityID"] = instances["EntityID"].astype(str)

        # Ensure all interval columns are tuples of (start, end)
        for col in instances.columns:
            if col != "EntityID":
                instances[col] = instances[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                instances[col] = instances[col].apply(lambda x: tuple(map(int, x)) if isinstance(x, tuple) else x)

        # Extract symbols from the columns (exclude "EntityID")
        symbols = [col for col in instances.columns if col != "EntityID"]

        # Calculate the size of the TIRP
        size = len(symbols)
        tirp_matrix = TirpMatrix()

        for i, symbol1 in enumerate(symbols):
            # Compute relationships for the new column
            relation_column = []
            for j, symbol2 in enumerate(symbols[:i]):  # Skip self-relationships
                # Compute the relationship between symbol1 and symbol2
                interval1 = instances[symbol2].iloc[0]  # Use the first instance
                interval2 = instances[symbol1].iloc[0]  # Use the first instance
                relationship = compute_relationship(interval1, interval2, epsilon, max_gap, num_relations)
                if relationship == -1:
                    raise ValueError(f"Relationship between {symbol2} and {symbol1} is undefined.")
                relation_column.append(relationship)

            # Extend the matrix with the computed column
            if relation_column:
                tirp_matrix.extend(relation_column)

        # Create and return the TIRP object
        tirp = TIRP()
        tirp._symbols = [int(symbol) if symbol.isdigit() else symbol for symbol in symbols]  # Normalize symbol types
        tirp._tirp_matrix = tirp_matrix
        tirp.size = size
        tirp.instances = instances

        return tirp


class TirpMatrix(object):
    """
        Class represents logic of "half TIRP matrix"
        holds the size of the symbol array corresponding to this half matrix.
        holds and array of relations.
        Half matrix logic definition:
        - the actual data needed for the relations is (|symbols| * (|symbols| -1)) / 2.
        - matrix row is all the symbols excluding the last one.
        - matrix column is all the symbols excluding the first one.
        - the matrix represented full size is (|symbols|-1)^2
        definition example:
            for symbols array = [A, B, C]
            and relations [r1, r2, r3], |relations| = 3*2/2
            the half matrix is

                | B | C
              -----------
              A | r1| r2
              B |   | r3
        self._size define as |symbols|-1
    """

    def __init__(self, relation=None):
        if relation is not None:
            self._size = 1
            self._relations = [relation]
        else:
            self._size = 0
            self._relations = []

    def get_relation(self, first_index, second_index):
        """
            Get the relation corresponds to the given indices from the representing symbols array.

            since the matrix logic as defined, then the second index must be subtracted by one.

            Constrains:
             - second_index >= first_index
             - self._size > first_index >= 0
             - self._size > second_index >= 0

             use the function n(a1 + an)/2 to calculate the offset in the relations array to be in the first place in
             the corresponding row data.

             subtract the "empty space" in the matrix is a function of row number and column index.

            for symbols array = [A, B, C]  and relations [r1, r2, r3]
            the half matrix is

                | B | C
              -----------
              A | r1| r2
              B |   | r3

              A as first index will be 0
              C as second index will be 2, but the index in the represented is 2-1 = 1.

        :param first_index: the first index of the symbol in the corresponding symbols array for the current relations
        :param second_index: the second index of the symbol in the corresponding symbols array for the current relations
        :return: the relation between symbols[first_index] and symbols[second_index]
        """
        row_index = first_index
        column_index = second_index - 1
        # offset = (row_index * (self._size + self._size - (row_index - 1))) / 2
        # index = int(offset + column_index - row_index)
        index = int(((1 + column_index) * column_index) / 2 + row_index)
        return self._relations[index]

    def extend(self, relation_column):
        """
            Extends the current matrix to support new symbol column.
            matrix size scales by 1 symbol representation.

            Example:
                Given
                relation_column = [r4,r5,r6]

            Current representation:

                relations = [r1, r2, r3]
                representing some [A, B, C] symbols order.

                | B | C  |
              --|---|----|
              A | r1| r2 |
              --|---|----|
              B |   | r3 |
              --|---|----|

            Becomes:

                relations = [r1, r2, r4, r3, r5, r6]
                representing some [A, B, C, D] symbols order.

                | B | C | D |
              --|---|---|---|
              A | r1| r2| r4|
              --|---|---|---|
              B |   | r3| r5|
              --|---|---|---|
              C |   |   | r6|

        :param relation_column: the column of relations as the new column in the "matrix"
        """
        for index in range(1, len(relation_column) + 1):
            # self.relations.insert(self.size * index, relation_column[index - 1])
            self._relations.append(relation_column[index - 1])
        self._size += 1

    def copy(self):
        ans = TirpMatrix()
        ans._size = self._size
        ans._relations = copy(self._relations)
        return ans

    def get_all_direct_relations(self):
        """
        extract the last relation column in the TirpMatrix
        :return: ans: list of int, the last relation column
        """
        return self._relations[len(self._relations) - self._size:]

    def get_relations(self):
        return self._relations

    def to_string(self):
        ans = ''
        for rel in self._relations:
            ans += str(rel) + '_'
        ans = ans[0: -1]
        return ans


def compute_relationship(interval1: Tuple[int, int], interval2: Tuple[int, int], epsilon: int, max_gap: int,
                         num_relations: int) -> int:
    """
    Compute the temporal relationship between two intervals with respect to epsilon and max_gap.
    Return the relationship as a numerical value directly.
    :param interval1: The first interval as a tuple (start, end).
    :param interval2: The second interval as a tuple (start, end).
    :param epsilon: Allowable difference between intervals to be considered overlapping.
    :param max_gap: Maximum allowable gap between intervals.
    :param num_relations: Mode for the number of relationships (3 or 7).
    :return: Numerical representation of the relationship or -1 if not defined.
    """
    start1, end1 = map(int, interval1[:2])
    start2, end2 = map(int, interval2[:2])

    s2_minus_e1 = start2 - end1

    # Check maximum gap
    if s2_minus_e1 > max_gap or start1 > start2:
        return -1  # Not defined

    e1_minus_s2 = end1 - start2
    s2_minus_s1 = start2 - start1
    e1_minus_e2 = end1 - end2
    e2_minus_e1 = end2 - end1

    # Compute relationships based on 3 or 7 relationship modes
    if num_relations == 3:
        if epsilon < s2_minus_e1 < max_gap:
            return 0  # "before"
        elif s2_minus_s1 > epsilon >= abs(e1_minus_s2) and e1_minus_e2 < epsilon:
            return 0  # "before"
        elif s2_minus_s1 > epsilon < e1_minus_s2 and e1_minus_e2 < epsilon:
            return 1  # "overlap"
        elif abs(s2_minus_s1) <= epsilon and abs(e1_minus_e2) <= epsilon:
            return 2  # "contain"
        elif s2_minus_s1 > epsilon and e1_minus_e2 > epsilon:
            return 2  # "contain"
        elif abs(s2_minus_s1) <= epsilon < e2_minus_e1:
            return 2  # "contain"
        elif s2_minus_s1 > epsilon >= abs(e1_minus_e2):
            return 2  # "contain"
    elif num_relations == 7:
        if epsilon < s2_minus_e1 < max_gap:
            return 0  # "before"
        elif s2_minus_s1 > epsilon >= abs(e1_minus_s2) and e1_minus_e2 < epsilon:
            return 1  # "meet"
        elif s2_minus_s1 > epsilon < e1_minus_s2 and e1_minus_e2 < epsilon:
            return 2  # "overlap"
        elif abs(s2_minus_s1) <= epsilon and abs(e1_minus_e2) <= epsilon:
            return 6  # "equal"
        elif s2_minus_s1 > epsilon and e1_minus_e2 > epsilon:
            return 4  # "contain"
        elif abs(s2_minus_s1) <= epsilon < e2_minus_e1:
            return 5  # "starts"
        elif s2_minus_s1 > epsilon >= abs(e1_minus_e2):
            return 3  # "finishby"
    return -1  # Not defined
