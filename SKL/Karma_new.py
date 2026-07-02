import pandas as pd
from typing import List, Tuple
import time


class Karma:
    def __init__(self, num_relations: int = 7, epsilon: int = 0, max_gap: int = float('inf')):
        """
        Initialize the Karma object.
        :param relationship_mode: Choose between 3 or 7 relationships (default is 7).
        :param epsilon: Allowable difference between intervals to still be calculated as overlapping.
        :param max_gap: Maximum time gap for relationships to be considered.
        """
        self.index = {}
        self.symbols_map = {}
        self.num_relations = num_relations
        self.epsilon = epsilon
        self.max_gap = max_gap

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
        temp_index = {key: [] for key in self.index.keys()}  # Initialize only for existing keys

        for entity_id, intervals in data.items():
            for i, interval1 in enumerate(intervals):
                symbol1 = interval1[2]  # Symbol ID
                # Add single-symbol occurrence to symbols_map
                if symbol1 not in self.symbols_map:
                    self.symbols_map[symbol1] = []
                self.symbols_map[symbol1].append((entity_id, (int(interval1[0]), int(interval1[1]))))

                # Skip intervals not in the selected_symbols list
                if selected_symbols and symbol1 not in selected_symbols:
                    continue

                for j, interval2 in enumerate(intervals):
                    if i >= j:
                        continue

                    # Break early if max_gap condition is exceeded
                    if int(interval2[0]) - int(interval1[1]) > self.max_gap:
                        break

                    # Skip intervals not in the selected_symbols list
                    if selected_symbols and interval2[2] not in selected_symbols:
                        continue

                    relationship = self.compute_relationship(interval1, interval2)

                    if relationship == "not_defined":
                        continue

                    key = f"{symbol1}_{interval2[2]}_{relationship}"

                    # Skip if the key does not exist in the initialized index
                    if key not in self.index:
                        continue

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
        relationship_number = int(relationship_number)

        # Validate the relationship number
        if relationship_number not in self.relationship_mapping.keys():
            # print(self.relationship_mapping.keys())
            print(f"Invalid relationship number: {relationship_number}")
            return None

        # Convert number to relationship name
        relationship_name = self.relationship_mapping[relationship_number]
        key = f"{symbol1}_{symbol2}_{relationship_name}"

        # Retrieve the corresponding TIRP table
        if key in self.index:
            return self.index[key]
        else:
            print(f"No TIRP table found for key: {key}")
            return None

    def get_one_size_tirp_instances(self, symbol: str) -> pd.DataFrame:
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


