from copy import deepcopy
from itertools import combinations
import pickle
from New_KarmaLego_Framework.Tirp_new import TIRP



#########################################
# 1) Full Allen Transition Table (Table 1)
#########################################
ALLEN_STIS_TRANSITION_TABLE = {
    2: {
        2: [2],
        3: [2],
        4: [2, 3, 4],
        6: [2],
        5: [2]
    },
    3: {
        2: [2],
        3: [3],
        4: [4],
        6: [3],
        5: [2]
    },
    4: {
        2: [2, 3, 4],
        3: [4],
        4: [4],
        6: [4],
        5: [2, 3, 4]
    },
    6: {
        2: [2],
        3: [3],
        4: [4],
        6: [6],
        5: [5]
    },
    5: {
        2: [2],
        3: [2],
        4: [2, 3, 4],
        6: [5],
        5: [5]
    }
}


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
        new_mat = TirpMatrix()
        new_mat._size = self._size
        new_mat._relations = self._relations.copy()
        return new_mat

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


class TIRPPrefix:
    def __init__(self,symbols, finished_symbols, unfinished_symbols, relations_matrix):
        self.finished_symbols = finished_symbols
        self.unfinished_symbols = unfinished_symbols
        self.symbols = symbols
        self.relations_matrix = relations_matrix

    def copy(self):
        return TIRPPrefix(
            deepcopy(self.symbols),
            deepcopy(self.finished_symbols[:]),
            deepcopy(self.unfinished_symbols),
            deepcopy(self.relations_matrix.copy())
        )

    def to_string(self):
        ans = f'{len(self.symbols)}-'
        for symbol in self.symbols:
            # Check if the symbol appears in unfinished_symbols (the first element of the tuple)
            if any(t[0] == symbol for t in self.unfinished_symbols):
                ans += f'{symbol}*_'
            else:
                ans += f'{symbol}_'
        ans += self.relations_matrix.to_string()
        return ans

    def __str__(self):
        return f"finished_symbols: {self.finished_symbols}\nunfinished_symbols: {self.unfinished_symbols}\nRelations: {self.relations_matrix.to_string()}"

#########################################
# 3) Helper functions
#########################################
def getRel(candidate: TIRPPrefix, i, j):
    return candidate.relations_matrix.get_relation(i, j)

def updateRel(candidate: TIRPPrefix, i, j, relation):
    newCandidate = candidate.copy()
    # Update the half-matrix
    if i > j:
        i, j = j, i
    row_index = i
    col_index = j - 1
    idx = int(((1 + col_index) * col_index) / 2 + row_index)
    # print(f'for update rel between symbol_index {row_index} and {j} - rel index {idx}')
    newCandidate.relations_matrix._relations[idx] = relation
    return newCandidate

#########################################
# 4) The expandTIRPCand function (Algorithm 4)
#########################################
def expandTIRPCand(c: TIRPPrefix, unfSTIsLen: int, adjJump: int, i: int):
    fnlCands = set()
    if (c.symbols[i] in c.finished_symbols) or (c.symbols[i+1] in c.finished_symbols) or (c.symbols[i+adjJump] in c.finished_symbols):
        rel = getRel(c, i, i+adjJump)
        if rel == 'temp-equals':
            inferredRels = [5,6]
        elif rel == 'temp-finished-by':
            inferredRels = [2,3,4] 
        else:
            # If the relation is not a temporary one, we assume it is already a final relation
            inferredRels = [rel]
    else:       
        # Step 2: fstRel = getRel(c, i, i+1)
        fstRel = getRel(c, i, i+1)
        # Step 3: scdRel = getRel(c, i+1, i+adjJump)
        scdRel = getRel(c, i+1, i+adjJump)

        # Step 4: inferredRels = transitionTable(fstRel, scdRel)
        # We'll map 'temp-equals' -> 'o' or 'fi' as needed if you have placeholders,
        # but here we assume the half-matrix already has 'o','fi','c','=', or 's'.
        # If you still have placeholders like 'temp-equals', you'll need a separate function
        # to map them to 'o' or 's' or so. For now, we assume your matrix is using these 5 relations directly.
        inferredRels = ALLEN_STIS_TRANSITION_TABLE.get(fstRel, {}).get(scdRel, [])

    # Steps 5-12
    for rel in inferredRels:
        # print(f'i={i} , adjump={adjJump}')
        extC = updateRel(c, i, i+adjJump, rel)

        if (i+1) < (unfSTIsLen - adjJump):
            # Step 8: expandTIRPCand(extC, unfSTIsLen, adjJump, i+1)
            fnlCands = fnlCands.union(expandTIRPCand(extC, unfSTIsLen, adjJump, i+1))
        elif (adjJump+1) < unfSTIsLen:
            # Step 10: expandTIRPCand(extC, unfSTIsLen, adjJump+1, i)
            fnlCands = fnlCands.union(expandTIRPCand(extC, unfSTIsLen, adjJump+1,0))
        else:
            # Step 12: fnlCands.union(extC)
            fnlCands = fnlCands.union({extC})

    return fnlCands


#########################################
# Helper: update_matrix_relation
#########################################
def update_matrix_relation(matrix: TirpMatrix, first_index, second_index, relation):
    if first_index > second_index:
        first_index, second_index = second_index, first_index
    row_index = first_index
    col_index = second_index - 1
    idx = int(((1 + col_index) * col_index) / 2 + row_index)
    matrix._relations[idx] = relation


def initGenAdjacentUnfSTIs(prefix: TIRPPrefix, epsilon=0):
    """
    Part 1: For every pair of unfinished STIs, set a temporary relation:
        - 'temp-equals' if abs(start_i - start_j) <= epsilon
        - 'temp-finished-by' otherwise
    Part 2: For each adjacent pair among the unfinished STIs (in lex order),
            expand the temporary relation to final relations.
            For 'temp-equals', we produce {equals, starts, swapped->starts}
            For 'temp-finished-by', we produce {overlaps, finished-by, contains}
    """
    # --- Part 1: set the temporary relations among all pairs of unfinished STIs ---
    new_mat = prefix.relations_matrix.copy()
    nFinished = len(prefix.finished_symbols)
    unfinished = prefix.unfinished_symbols
    symbols = prefix.symbols

    for (sym_i, sym_j) in combinations(unfinished, 2):
        sym_name_i, start_i = sym_i
        sym_name_j, start_j = sym_j
        # Find the actual indices of these symbols in the main prefix.symbols list
        try:
            i_idx = symbols.index(sym_name_i)
            j_idx = symbols.index(sym_name_j)
        except ValueError:
            # This should not happen if unfinished_symbols are consistent with symbols
            continue
        if abs(start_i - start_j) <= epsilon:
            temp_relation = 'temp-equals'
        else:
            temp_relation = 'temp-finished-by'
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        row_index = i_idx
        col_index = j_idx - 1
        idx = int(((1 + col_index) * col_index) / 2 + row_index)
        new_mat._relations[idx] = temp_relation

    base_prefix = TIRPPrefix(symbols,prefix.finished_symbols, unfinished, new_mat)

    # --- Part 2: expand each adjacent pair of unfinished STIs ---
    sorted_unfinished_tuples = sorted(unfinished, key=lambda x: (x[1], x[0]))
    candidate_list = [base_prefix]
    
    # If there are 2 or fewer unfinished STIs, we iterate over adjacent pairs directly.
    if len(sorted_unfinished_tuples) <= 2:
        # Iterate over pairs of unfinished symbols that are adjacent *in the sorted unfinished list*
        for k in range(len(sorted_unfinished_tuples) - 1):
            new_candidates = []
            # Get the two adjacent unfinished symbols (names and start times)
            sym_i = sorted_unfinished_tuples[k]
            sym_j = sorted_unfinished_tuples[k+1]

            sym_name_i, _ = sym_i
            sym_name_j, _ = sym_j

            # Find their actual indices in the `symbols` of each candidate
            for cand_prefix in candidate_list:
                try:
                    # These indices are crucial: they refer to positions in cand_prefix.symbols
                    actual_idx_adj_i = symbols.index(sym_name_i)
                    actual_idx_adj_j = symbols.index(sym_name_j)
                except ValueError:
                    # If a symbol from sorted_unfinished_tuples is not in cand_prefix.symbols,
                    # this candidate might be inconsistent or already processed. Skip.
                    new_candidates.append(cand_prefix) # Or handle error
                    continue

                # Ensure the order for get_relation
                first_overall_idx = min(actual_idx_adj_i, actual_idx_adj_j)
                second_overall_idx = max(actual_idx_adj_i, actual_idx_adj_j)
                temp_rel = cand_prefix.relations_matrix.get_relation(first_overall_idx, second_overall_idx)

                if temp_rel == 'temp-equals':
                    # 1) "equals" candidate
                    eqCand = cand_prefix.copy()
                    update_matrix_relation(eqCand.relations_matrix, first_overall_idx, second_overall_idx, 6)
                    new_candidates.append(eqCand)

                    # 2) "starts" candidate (same order)
                    stCand = cand_prefix.copy()
                    update_matrix_relation(stCand.relations_matrix, first_overall_idx, second_overall_idx, 5)
                    new_candidates.append(stCand)

                    # # 3) swapped candidate -> "starts"
                    # #    effectively implementing what would have been "started-by"
                    swappedCand = swapSymbolsInPrefix(cand_prefix, first_overall_idx, second_overall_idx)
                    update_matrix_relation(swappedCand.relations_matrix, first_overall_idx, second_overall_idx, 5)
                    new_candidates.append(swappedCand)


                elif temp_rel == 'temp-finished-by':
                    # Expand normally, e.g. { 'overlaps', 'finished-by', 'contains' }
                    expansions = [2, 3, 4]
                    for final_rel in expansions:
                        new_cand = cand_prefix.copy()
                        update_matrix_relation(new_cand.relations_matrix, first_overall_idx, second_overall_idx, final_rel)
                        new_candidates.append(new_cand)

                else:
                    # If it's not a known temporary relation, keep it as-is.
                    new_candidates.append(cand_prefix)

            candidate_list = new_candidates
    else:
        # If there are more than 2 unfinished STIs, we need to iterate over all pairs of indices
        # in the unfinished symbols list, not just adjacent ones.
        for i in range(len(base_prefix.symbols)-1):
            new_candidates = []

            # Find their actual indices in the `symbols` of each candidate
            for cand_prefix in candidate_list:
                temp_rel = cand_prefix.relations_matrix.get_relation(i, i+1)

                if temp_rel == 'temp-equals':
                    # 1) "equals" candidate
                    eqCand = cand_prefix.copy()
                    update_matrix_relation(eqCand.relations_matrix, i, i+1, 6)
                    new_candidates.append(eqCand)

                    # 2) "starts" candidate (same order)
                    stCand = cand_prefix.copy()
                    update_matrix_relation(stCand.relations_matrix, i, i+1, 5)
                    new_candidates.append(stCand)

                    # 3) swapped candidate -> "starts"
                    #    effectively implementing what would have been "started-by"
                    swappedCand = swapSymbolsInPrefix(cand_prefix, i, i+1)
                    update_matrix_relation(swappedCand.relations_matrix, i, i+1, 5)
                    new_candidates.append(swappedCand)

                elif temp_rel == 'temp-finished-by':
                    # Expand normally, e.g. { 'overlaps', 'finished-by', 'contains' }
                    expansions = [2, 3, 4]
                    for final_rel in expansions:
                        new_cand = cand_prefix.copy()
                        update_matrix_relation(new_cand.relations_matrix, i, i+1, final_rel)
                        new_candidates.append(new_cand)

                else:
                    # If it's not a known temporary relation, keep it as-is.
                    new_candidates.append(cand_prefix)

            candidate_list = new_candidates
    return candidate_list


def swapSymbolsInPrefix(cand: TIRPPrefix, i_idx: int, j_idx: int) -> TIRPPrefix:
    """
    Swap the symbols at overall indices i_idx and j_idx in the candidate.
    This function updates both the overall candidate.symbols list and, if both indices
    fall into the unfinished portion, the candidate.unfinished_symbols list.
    """
    newCand = cand.copy()
    # Swap in the overall symbol list.
    newCand.symbols[i_idx], newCand.symbols[j_idx] = newCand.symbols[j_idx], newCand.symbols[i_idx]

    # Determine the number of finished symbols.
    f = len(newCand.finished_symbols)
    # If both indices are in the unfinished portion (i.e. >= f), swap them in the unfinished_symbols list too.
    if i_idx >= f and j_idx >= f:
        off_i = i_idx - f
        off_j = j_idx - f
        newCand.unfinished_symbols[off_i], newCand.unfinished_symbols[off_j] = newCand.unfinished_symbols[off_j], \
        newCand.unfinished_symbols[off_i]
    return newCand

def reduce_7_to_3_relations(prefix_list):
    """
    Input: prefix_list - a list of TIRPPrefix objects.
    For each TIRPPrefix, do the following:
      1) Create a copy of the symbol list and a copy of the TirpMatrix.
      2) Replace each of the 7 Allen relations in the half-matrix with one of the 3 coarser relations:
           "before", "overlap", or "contains".
      3) Build a deduplication key from the symbol ordering and the half-matrix string.
      4) If the candidate (TIRP) has not been seen yet, create a new TIRP object with:
           new_tirp._symbols = copied symbol list,
           new_tirp._tirp_matrix = copied (and mapped) TirpMatrix,
           new_tirp.size = number of symbols.
    Return a set of unique TIRP objects in the coarser 3-relation form.
    """
    REL_7_TO_3 = {
        0:      0,
        1:       0,
        2:    1,
        5:      2,
        3: 2,
        4:    2,
        6:      2
    }

    final_tirps = set()
    visited_keys = set()

    for prefix in prefix_list:
        # Copy the symbols and the relation matrix.
        symbols_copy = prefix.symbols[:]  # shallow copy is enough for a list of strings
        relations_matrix_copy = prefix.relations_matrix.copy()

        # Map each relation from the 7-relation set to the 3-relation set.
        for idx, old_rel in enumerate(relations_matrix_copy._relations):
            if old_rel in REL_7_TO_3:
                new_rel = REL_7_TO_3[old_rel]
                relations_matrix_copy._relations[idx] = new_rel
            else:
                # If a temporary or unknown relation is encountered, handle as needed.
                pass

        # Build a key for deduplication using the symbols and the half-matrix string.
        key = (tuple(symbols_copy), relations_matrix_copy.to_string())
        if key not in visited_keys:
            visited_keys.add(key)
            # Create a new TIRP object.
            new_tirp = TIRP()
            new_tirp._symbols = symbols_copy
            new_tirp._tirp_matrix = relations_matrix_copy
            new_tirp.size = len(symbols_copy)
            final_tirps.add(new_tirp)

    return final_tirps


def from_3_to_7_relations(prefix):
    REL_3_TO_7 = {
        0:      0,
        1:       2,
        2:    4
    }
    # Copy the symbols and the relation matrix.
    symbols_copy = prefix.symbols[:]  # shallow copy is enough for a list of strings
    relations_matrix_copy = prefix.relations_matrix.copy()
    unfinished_symbols = prefix.unfinished_symbols
    finished_symbols = prefix.finished_symbols

    # Map each relation from the 7-relation set to the 3-relation set.
    for idx, old_rel in enumerate(relations_matrix_copy._relations):
        if old_rel in REL_3_TO_7:
            new_rel = REL_3_TO_7[old_rel]
            relations_matrix_copy._relations[idx] = new_rel
        else:
            # If a temporary or unknown relation is encountered, handle as needed.
            pass
    # Create a new TIRP object.
    new_prefix = TIRPPrefix(symbols_copy,finished_symbols,unfinished_symbols,relations_matrix_copy)
    return new_prefix


def generate_all_complete_tirps(prefix: TIRPPrefix, epsilon=0,num_relations=7):
    """
    Matches the structure of Algorithm 3 from the paper:
    1) fnlTIRPCnddts <- ∅
    2) unfSTIsLen <- number of unfinished STIs
    3) initTIRPCanddts <- initGenAdjacentUnfSTIs(prefix, epsilon)
    4) for each c in initTIRPCanddts:
          extTIRPCnddts <- expandTIRPCand(c, unfSTIsLen, adjJump=2, i=0)
          fnlTIRPCnddts <- fnlTIRPCnddts ∪ extTIRPCnddts
    5) return fnlTIRPCnddts
    """
    if num_relations == 3:
        prefix = from_3_to_7_relations(prefix)

    # 1) Initialize final candidates set
    finalCandidates = set()

    # 2) Number of unfinished STIs in the prefix
    unfinish = len(prefix.symbols)

    # 3) Generate the initial candidate TIRP-prefixes by enumerating adjacent unfinished STIs
    #    This function should return a list or set of TIRPPrefix objects (3^(k-1) expansions).
    initTIRPCanddts = initGenAdjacentUnfSTIs(prefix, epsilon)

    # 4) For each candidate in initTIRPCanddts, expand them fully
    for c in initTIRPCanddts:
        # print(f'temp tirp : {c.to_string()}')
        if len(c.symbols) <=2 or len(c.unfinished_symbols)<=2:
            extTIRPCnddts = {c}
        else: extTIRPCnddts = expandTIRPCand(c, unfinish, adjJump=2, i=0)
        finalCandidates |= extTIRPCnddts

    # 5) Return the final set of expanded TIRP candidates
    if num_relations == 3:
        finalCandidates = reduce_7_to_3_relations(finalCandidates)
    else:
        final_tirps = []
        for cand in finalCandidates:
            # print(f'final candidate: {cand.to_string()}')
            symbols_copy = cand.symbols[:]  # shallow copy is enough for a list of strings
            relations_matrix_copy = cand.relations_matrix.copy()
            new_tirp = TIRP()
            new_tirp._symbols = symbols_copy
            new_tirp._tirp_matrix = relations_matrix_copy
            new_tirp.size = len(symbols_copy)
            final_tirps.append(new_tirp)
        # print(f'prefix: {prefix.to_string()}')
        # print(f'length of final_tirps: {len(final_tirps)}')
        return final_tirps
    return finalCandidates


def update_matrix_after_removal(matrix, removed_index, old_symbols):
    """
    Given a TirpMatrix (matrix) built for old_symbols (list of symbols),
    return a new TirpMatrix computed only for the symbols remaining after
    removing the symbol at position removed_index.

    The new_relations list is pre-allocated, and relations are placed
    at their correct index based on TirpMatrix's storage convention.
    For a new index i in the range [0, new_n), its corresponding old index is:
        old_i = i if i < removed_index else i + 1.
    """
    new_symbols_count = len(old_symbols) - 1

    # Create a new TirpMatrix instance.
    new_tirp_matrix = TirpMatrix()

    if new_symbols_count <= 1:
        # If 0 or 1 symbol remains, there are no relations.
        return new_tirp_matrix

    # For new_symbols_count > 1
    # Calculate the total number of relations in the new matrix (N*(N-1)/2)
    num_total_new_relations = (new_symbols_count * (new_symbols_count - 1)) // 2
    new_relations_list = [None] * num_total_new_relations

    # Iterate over all unique pairs (i, j) in the new symbol list, where i < j
    for i in range(new_symbols_count):  # Index for the first symbol in the pair
        for j in range(i + 1, new_symbols_count):  # Index for the second symbol in the pair
            # Map new indices i, j to their corresponding old indices
            old_i = i if i < removed_index else i + 1
            old_j = j if j < removed_index else j + 1
            
            # Retrieve the relation from the original matrix using old indices
            relation = matrix.get_relation(old_i, old_j)

            # Calculate the flat list index for the relation between new symbol i and new symbol j.
            # which is effectively: (second_symbol_idx * (second_symbol_idx - 1) / 2) + first_symbol_idx
            target_index_in_new_list = (j * (j - 1)) // 2 + i
            
            new_relations_list[target_index_in_new_list] = relation
            
    new_tirp_matrix._relations = new_relations_list
    # The _size attribute of TirpMatrix is defined as (number of symbols - 1)
    new_tirp_matrix._size = new_symbols_count - 1
    
    return new_tirp_matrix


#########################################
# Helper: Remove symbol from TIRP-Prefix and update the matrix
#########################################
def remove_symbol_from_prefix(prefix, index):
    """
    Removes the symbol at the given index from prefix._symbols.
    Also, if the symbol is present in finished_symbols, remove it.
    And if it is in unfinished_symbols, remove it as well.
    Then, update the TirpMatrix to include only the relations for the remaining symbols.
    """
    # Copy the old symbol list for matrix reconstruction.
    old_symbols = prefix.symbols[:]
    # Remove the symbol from finished_symbols (if present) and from unfinished_symbols.
    sym = prefix.symbols[index]
    if sym in prefix.finished_symbols:
        prefix.finished_symbols.remove(sym)
    if sym in prefix.unfinished_symbols:
        prefix.unfinished_symbols.remove(sym)
    # Remove the symbol from the overall symbols list.
    prefix.symbols.pop(index)
    # Rebuild the TirpMatrix based on the remaining symbols.
    prefix.relations_matrix = update_matrix_after_removal(prefix.relations_matrix, index, old_symbols)

#########################################
# Helper: Remove from unfinished_symbols
#########################################
def remove_from_unfinished(prefix, sym):
    """
    Removes any tuple (s, time) from prefix.unfinished_symbols whose first element equals sym.
    """
    prefix.unfinished_symbols = [t for t in prefix.unfinished_symbols if t[0] != sym]


# #########################################
# # TIRP-Prefixes Revealer
# ########################################
def tirp_prefixes_revealer(tirp,num_relations=7):
    """
    Implements the TIRP-Prefixes Revealer as described in the paper.

    Input:
      tirp: a TIRP object with attributes:
            - _symbols: ordered list of symbols (e.g., ['A', 'B', 'C', 'D'])
            - size: number of symbols
            - _tirp_matrix: a TirpMatrix object storing the half-matrix of relations
            - tiep_representation: list of tuples of tiep strings in chronological order.
              For example: [('A+', 'B+', 'C+', 'D+'), ('B-'), ('D-')]

    Initially, we create a TIRPPrefix candidate with:
      finished_symbols = copy of tirp._symbols (all symbols are initially finished)
      unfinished_symbols = [] (empty)
      symbols = same as finished_symbols
    Then, we process the reversed tiep_representation (from last to first). For each tiep:
      - If the tiep ends with '-', it indicates the end of an interval.
           We remove the symbol from finished_symbols and add it to unfinished_symbols as a tuple:
           (symbol, start_time) – where start_time is determined by the order of the starting tiep in the original representation.
      - If the tiep ends with '+', it indicates the beginning of an interval.
           We remove the symbol from unfinished_symbols (if it is present) and also remove it from the overall symbols list.
           Removing from symbols also calls update_matrix_after_removal.

    After processing each tiep group, we convert the current TIRP candidate into a TIRPPrefix and prepend it to our result list.
    Finally, we return the list of TIRPPrefix candidates.
    """
    tirp = tirp.copy_tirp()
    # First, extract start times for each symbol from the original tirp's tiep_representation.
    # Here, we assign a start time equal to the group index (starting at 1) when a symbol first appears with '+'.
    start_times = {}
    tiep_representation = tirp.compute_tiep_order(num_relations)
    # print(tiep_representation)
    for group_index, group in enumerate(tiep_representation, start=1):
        for tiep in group:
            if tiep.endswith('+'):
                sym = tiep[:-1]
                if sym not in start_times:
                    start_times[sym] = group_index

    # Initialize the TIRPPrefix candidate.
    init_finished = tirp._symbols[:]  # all symbols are initially finished
    init_unfinished = []  # initially empty
    currPrefix = TIRPPrefix(tirp._symbols,init_finished, init_unfinished, tirp._tirp_matrix.copy())

    # Get the tiep groups in reverse order.
    revTieps = list(reversed(tiep_representation))
    prefix_list = []
    # print(f'revTieps: {revTieps}')

    # Process each tiep group (from latest to earliest)
    for tiep_group in revTieps[:-1]:
        # print(f'tiep_group: {tiep_group}')
        for tiep in tiep_group:
            # print(f'tiep: {tiep}')
            sym = tiep[:-1]
            try:
                idx = currPrefix.symbols.index(sym)
            except ValueError:
                continue  # already removed
            if tiep.endswith('-'):
                # Ending tiep: this marks the end of the interval.
                # Remove sym from finished_symbols (if present) and add to unfinished_symbols as (sym, start_time)
                if sym in currPrefix.finished_symbols:
                    currPrefix.finished_symbols.remove(sym)
                # Only add if not already added
                if not any(t[0] == sym for t in currPrefix.unfinished_symbols):
                    currPrefix.unfinished_symbols.append((sym, start_times.get(sym, None)))
                # Do not remove sym from the overall symbols list.
            elif tiep.endswith('+'):
                # Starting tiep: remove sym from unfinished_symbols (if present)
                remove_from_unfinished(currPrefix, sym)
                # Remove sym from overall symbols and update the matrix.
                remove_symbol_from_prefix(currPrefix, idx)
        # Before saving the candidate, sort the unfinished_symbols list by start_time, then lexicographically.
        currPrefix.unfinished_symbols.sort(key=lambda tup: (tup[1], tup[0]))
        # After processing one group, convert the current candidate into a TIRPPrefix object and store it.
        # print('currPrefix:', currPrefix.to_string())
        prefix_list.insert(0, deepcopy(currPrefix))

    return prefix_list
