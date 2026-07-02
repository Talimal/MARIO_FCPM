import time
from typing import Tuple

import pandas as pd

from New_KarmaLego_Framework.RelationHandler import RelationHandler
from New_KarmaLego_Framework.Tirp_new import TIRP


class Karma:
    def __init__(self, min_ver_support, num_relations: int = 7, epsilon: int = 0, max_gap: int = float('inf'), 
                 index_same: bool = True, skip_followers: bool = False, skip_same_variable: bool = False):
        """
        Initialize the Karma object.
        :param relationship_mode: Choose between 3 or 7 relationships (default is 7).
        :param epsilon: Allowable difference between intervals to still be calculated as overlapping.
        :param max_gap: Maximum time gap for relationships to be considered.
        :param index_same: Boolean, whether to index relationships between same symbols.
        :param skip_followers: Boolean, whether to skip indexing relationships between consecutive symbols from same variable.
        :param skip_same_variable: Boolean, whether to skip indexing relationships between any symbols from same variable.
        """
        self.index = {}
        self.symbols_map = {}
        self.num_relations = num_relations
        self.epsilon = epsilon
        self.max_gap = max_gap
        self._min_ver_support = min_ver_support
        self._relation_handler_obj = RelationHandler(num_relations)
        self.index_same = index_same
        self.skip_followers = skip_followers
        self.skip_same_variable = skip_same_variable

        # Define mappings for 3 and 7 relationships
        self.relation_mapping = {
            3: {
                "before": 0,
                "overlap": 1,
                "contain": 2
            },
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
        self.relationship_mapping = {v: k for k, v in self.relation_mapping[self.num_relations].items()}

    def parse_input(self, file_path: str) -> dict:
        """
        Parse the updated TXT file format where every two lines contain an entity's data.
        :param file_path: Path to the TXT file.
        :return: A dictionary where each key is an EntityID and the value is a list of intervals.
        """
        data = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Skip metadata lines
        lines = lines[2:]

        # Read every two lines for each entity
        for i in range(0, len(lines), 2):
            entity_id_line = lines[i].strip()
            intervals_line = lines[i + 1].strip()

            # Entity ID
            entity_id = entity_id_line.split(';')[0]

            # Parse intervals
            intervals = [
                tuple(item.split(',')) for item in intervals_line.split(';') if item
            ]

            if intervals:
                data[entity_id] = intervals

        self._min_ver_support = self._min_ver_support * len(data.keys())
        self._num_of_entities = len(data.keys())
        return data

    def compute_relationship(self, interval1: Tuple[int, int], interval2: Tuple[int, int]) -> str:
        """
        Compute the temporal relationship between two intervals with respect to epsilon and max_gap.
        """
        start1, end1 = map(int, interval1[:2])
        start2, end2 = map(int, interval2[:2])

        s2_minus_e1 = start2 - end1

        # Check maximum gap
        if s2_minus_e1 > self.max_gap or start1 > start2:
            return "not_defined"
        e1_minus_s2 = end1 - start2
        s2_minus_s1 = start2 - start1
        e1_minus_e2 = end1 - end2
        e2_minus_e1 = end2 - end1

        # Compute relationships based on 3 or 7 relationship modes
        if self.num_relations == 3:
            if self.epsilon < s2_minus_e1 < self.max_gap:
                return "before"
            elif s2_minus_s1 > self.epsilon >= abs(e1_minus_s2) and e1_minus_e2 < self.epsilon:
                return "before"
            elif s2_minus_s1 > self.epsilon < e1_minus_s2 and e1_minus_e2 < self.epsilon:
                return "overlap"
            elif abs(s2_minus_s1) <= self.epsilon and abs(e1_minus_e2) <= self.epsilon:
                return "contain"
            elif s2_minus_s1 > self.epsilon and e1_minus_e2 > self.epsilon:
                return "contain"
            elif abs(s2_minus_s1) <= self.epsilon < e2_minus_e1:
                return "contain"
            elif s2_minus_s1 > self.epsilon >= abs(e1_minus_e2):
                return "contain"
        elif self.num_relations == 7:
            if self.epsilon < s2_minus_e1 < self.max_gap:
                return "before"
            elif s2_minus_s1 > self.epsilon >= abs(e1_minus_s2) and e1_minus_e2 < self.epsilon:
                return "meet"
            elif s2_minus_s1 > self.epsilon < e1_minus_s2 and e1_minus_e2 < self.epsilon:
                return "overlap"
            elif abs(s2_minus_s1) <= self.epsilon and abs(e1_minus_e2) <= self.epsilon:
                return "equal"
            elif s2_minus_s1 > self.epsilon and e1_minus_e2 > self.epsilon:
                return "contain"
            elif abs(s2_minus_s1) <= self.epsilon < e2_minus_e1:
                return "starts"
            elif s2_minus_s1 > self.epsilon >= abs(e1_minus_e2):
                return "finishby"
        return "not_defined"

    def index_relationships(self, data: dict, selected_symbols=None):
        """
        Index all relationships between intervals and store them in the dictionary.
        Also, index occurrences of single symbols.
        :param data: Dictionary of entity intervals.
        :param selected_symbols: List of symbol IDs to include. If None, include all symbols.
        """
        # Temporary storage for faster appends
        temp_index = {}

        # Initialize the symbol-to-variable mapping
        self.symbol_to_variable_map = {}

        for entity_id, intervals in data.items():
            for i, interval1 in enumerate(intervals):
                symbol1 = interval1[2]  # Symbol ID
                variable1 = interval1[3]  # Variable

                # Add symbol-to-variable mapping
                if symbol1 not in self.symbol_to_variable_map:
                    self.symbol_to_variable_map[symbol1] = variable1

                # Skip intervals not in the selected_symbols list
                if selected_symbols and symbol1 not in selected_symbols:
                    continue
                # Add single-symbol occurrence to symbols_map
                if symbol1 not in self.symbols_map:
                    self.symbols_map[symbol1] = []
                self.symbols_map[symbol1].append((entity_id, (int(interval1[0]), int(interval1[1]))))

                for j, interval2 in enumerate(intervals):
                    if i >= j:
                        continue

                    symbol2 = interval2[2]  # Symbol ID
                    variable2 = interval2[3]  # Variable

                    # Apply index_same constraint: skip if symbols are same and index_same is False
                    if not self.index_same and symbol1 == symbol2:
                        continue

                    # Apply skip_same_variable constraint: skip if symbols are from same variable
                    if self.skip_same_variable:
                        same_variable = (variable1 == variable2)
                        if same_variable:
                            continue

                    # Apply skip_followers constraint: skip if symbols are consecutive from same variable
                    if self.skip_followers:
                        abs_difference = abs(int(symbol1) - int(symbol2))
                        same_variable = (variable1 == variable2)
                        if abs_difference == 1 and same_variable:
                            continue

                    # Break early if max_gap condition is exceeded
                    if int(interval2[0]) - int(interval1[1]) > self.max_gap:
                        break

                    # Skip intervals not in the selected_symbols list
                    if selected_symbols and symbol2 not in selected_symbols:
                        continue

                    relationship = self.compute_relationship(interval1, interval2)

                    if relationship == "not_defined":
                        continue

                    key = f"{symbol1}_{symbol2}_{relationship}"

                    if key not in temp_index:
                        temp_index[key] = []

                    # Append the record as a tuple
                    temp_index[key].append(
                        (entity_id, (int(interval1[0]), int(interval1[1])), (int(interval2[0]), int(interval2[1])))
                    )

        # Convert lists in temp_index to DataFrames and store in self.index
        for key, records in temp_index.items():
            if records:  # Only convert if there are records
                self.index[key] = pd.DataFrame(records, columns=["EntityID", key.split("_")[0], key.split("_")[1]])

        # Convert symbols_map to DataFrames
        for symbol, records in self.symbols_map.items():
            if records:  # Only convert if there are records
                self.symbols_map[symbol] = pd.DataFrame(records, columns=["EntityID", symbol])

    def get_symbols_map(self):
        return self.symbols_map

    def get_min_vertical_support(self):
        return self._min_ver_support

    def get_symbol_vertical_support(self, symbol) -> int:
        """
        Get the vertical support of a symbol.
        Vertical support is the number of unique entities in which the symbol appears.

        :param symbol: The symbol for which to calculate vertical support.
        :return: The number of unique entities that have at least one occurrence of the symbol.
        """

        key = str(symbol)
        if key not in self.symbols_map:
            print(f"Symbol {key} not found in the index.")
            return 0

        symbol_df = self.symbols_map[key]  # Get the DataFrame for the symbol
        vertical_support = symbol_df["EntityID"].nunique() if not symbol_df.empty else 0  # Count unique entities
        return vertical_support

    def get_vertical_support_of_sym_to_sym_rel(self, symbol1: str, symbol2: str, relationship: str) -> int:
        """
        Get the vertical support of a specific relationship between two symbols.
        Vertical support is the number of unique entities in which the relationship between the two symbols appears.

        :param symbol1: The first symbol in the relationship.
        :param symbol2: The second symbol in the relationship.
        :param relationship: The relationship between the two symbols (e.g., "before", "meet", "overlap").
        :return: The number of unique entities supporting the specified relationship.
        """
        # Validate the relationship number
        if relationship not in self.relationship_mapping:
            print(f"Invalid relationship number: {relationship}")
            return None

        # Convert number to relationship name
        relationship_name = self.relationship_mapping[relationship]
        key = f"{symbol1}_{symbol2}_{relationship_name}"

        # Check if the key exists in the index
        if key not in self.index:
            # print(f"Relationship {relationship} between {symbol1} and {symbol2} not found.")
            return 0

        # Count the number of unique entities that support this relationship
        vertical_support = self.index[key]["EntityID"].nunique()

        return vertical_support

    def get_two_sized_tirp_for_symbols_and_rel(self, symbol1: str, symbol2: str, relationship: int):
        """
        Get a TIRP object of size 2 for the specified symbols and numerical relationship.

        :param symbol1: The first symbol.
        :param symbol2: The second symbol.
        :param relationship: The numerical representation of the relationship (e.g., 0 for "before", 1 for "meet").
        :return: A TIRP object of size 2 corresponding to the symbols and relationship, or None if not found.
        """
        # Convert the numerical relationship to its string representation
        if relationship not in self.relationship_mapping:
            print(f"Invalid relationship number: {relationship}")
            return None

        relationship_name = self.relationship_mapping[relationship]

        # Construct the key for the index lookup
        key = f"{symbol1}_{symbol2}_{relationship_name}"

        # Check if the key exists in the index
        if key not in self.index:
            print(f"No relationship '{relationship_name}' found between {symbol1} and {symbol2}.")
            return None

        # Retrieve the DataFrame corresponding to the relationship
        relation_df = self.index[key]

        if relation_df.empty:
            print(f"No instances for relationship '{relationship_name}' between {symbol1} and {symbol2}.")
            return None

        # Create a TIRP of size 2
        tirp = TIRP(first_symbol=symbol1, second_symbol=symbol2, relation=relationship, instances=relation_df)

        return tirp

    def get_instances_table(self, symbol1: str, symbol2: str, relationship_number: int):
        """
        Retrieve the table of TIRPs of size 2 for the given symbols and numerical relationship.
        :param symbol1: The first symbol (e.g., "1").
        :param symbol2: The second symbol (e.g., "2").
        :param relationship_number: The numerical representation of the relationship (e.g., 0 for "before").
        :return: A Pandas DataFrame containing the corresponding table, or None if not found.
        """
        # Reverse mapping: number to name
        # relationship_mapping = {v: k for k, v in self.relation_mapping[self.num_relations].items()}

        # Validate the relationship number
        if relationship_number not in self.relationship_mapping:
            print(f"Invalid relationship number: {relationship_number}")
            return None

        # Convert number to relationship name
        relationship_name = self.relationship_mapping[relationship_number]
        key = f"{symbol1}_{symbol2}_{relationship_name}"

        # Retrieve the corresponding TIRP table
        if key in self.index:
            return self.index[key]
        else:
            # print(f"No TIRP table found for key: {key}")
            return None

    def get_one_size_tirp_instances(self, symbol) -> pd.DataFrame:
        """
        Get the DataFrame of instances for a single-symbol TIRP.

        :param symbol: The symbol (string) to retrieve instances for.
        :return: A DataFrame containing the instances for the given symbol, or an empty DataFrame if no instances are found.
        """
        # Check if the symbol exists in the symbols_map
        key = str(symbol)
        if key in self.symbols_map:
            return self.symbols_map[key]  # Return the DataFrame of instances for the symbol
        else:
            print(f"No instances found for symbol: {symbol}")
            return pd.DataFrame(columns=["EntityID", key])  # Return an empty DataFrame with expected columns

    def run_karma(self, file_path: str, selected_symbols=None):
        """
        Main function to run the Karma indexing process.
        :param file_path: Path to the input TXT file.
        :param selected_symbols: List of symbol IDs to include. If None, include all symbols.
        """
        start_time = time.time()
        # print(selected_symbols)
        data = self.parse_input(file_path)
        self.index_relationships(data, selected_symbols=selected_symbols)
        end_time = time.time()
        print(f'done karma in {end_time - start_time:.2f} seconds')
