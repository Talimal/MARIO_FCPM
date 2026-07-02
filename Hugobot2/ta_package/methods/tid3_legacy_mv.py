"""
Archive of the original TID3 multivariate refinement (Phase 2 joint enumeration)
and the Phase 1 beam-search branch that produced its per-variable candidate sets.

This file is NOT imported by the main pipeline. It is preserved for the journal-extension
greedy-vs-exhaustive ablation: comparing the conditional intra-variable greedy MV phase
that now lives in tid3.py against the original K^N joint enumeration.

To run the legacy algorithm:
  1. Re-introduce `self._alternative_cutoff_sets = {}` in TID3.__init__.
  2. Re-introduce the beam-search branch inside `_optimized_candidate_selection` (see
     `_OPTIMIZED_CANDIDATE_SELECTION_BEAM_BRANCH` below), and have it return a 3-tuple.
  3. Re-introduce the alternative-set capture in `_generate_cutpoints` and update the
     unpack to a 3-tuple.
  4. Add the `_multivariate_refinement_exhaustive` method below to TID3 and call it from
     `fit()` instead of `_multivariate_refinement_greedy`.

Original locations in tid3.py prior to the refactor:
  - `_multivariate_refinement`: lines 1023-1187
  - Beam-search branch in `_optimized_candidate_selection`: lines 1410-1510
  - `_alternative_cutoff_sets` init: line 151
  - Beam-search alternative-set capture in `_generate_cutpoints`: lines 1632-1634
"""

# Imports that the legacy code depended on (kept for reference).
# from joblib import Parallel, delayed
# import itertools
# import numpy as np
# from tqdm import tqdm
# from ..constants import TEMPORAL_PROPERTY_ID
# (logger from this module)


_OPTIMIZED_CANDIDATE_SELECTION_BEAM_BRANCH = '''
        # === BEAM SEARCH PATH (multivariate refinement) ===
        # Inserted inside _optimized_candidate_selection AFTER the precompute and BEFORE
        # the greedy-path block. Required `multivariate_refinement=True` to fire.
        if self.multivariate_refinement:
            K = self.top_k_per_iteration
            # Each beam: (cutoffs_list, pool_list)
            beams = [([], candidate_pool.copy())]
            # Track the per-state stats of the best (top) expansion at the last
            # iteration; that expansion produced beams[0], the winning beam.
            best_beam_per_state_stats = None

            for iteration in range(1, nb_bins):
                # Collect all (beam_idx, candidate_idx, score, candidate_value,
                # per_state_stats, zero_score_reason) across all beams.
                all_expansions = []

                for beam_idx, (beam_cutoffs, beam_pool) in enumerate(beams):
                    if len(beam_pool) == 0:
                        continue
                    scores, per_state_stats_list, zero_score_reasons = _evaluate_candidates(
                        beam_cutoffs, beam_pool,
                        f"beam {beam_idx+1}/{len(beams)}, bin {iteration}/{nb_bins-1}"
                    )

                    valid_mask = ~np.isneginf(scores)
                    valid_indices = np.where(valid_mask)[0]
                    for idx in valid_indices:
                        all_expansions.append(
                            (
                                beam_idx,
                                idx,
                                float(scores[idx]),
                                beam_pool[idx],
                                per_state_stats_list[idx],
                                zero_score_reasons[idx],
                            )
                        )

                if not all_expansions:
                    logger.warning(f"Early termination at iteration {iteration}: no valid expansions across all beams")
                    break

                # Sort by score descending, then keep top-K globally while skipping
                # expansions that would produce a cutoff set already chosen at this iteration.
                all_expansions.sort(key=lambda x: x[2], reverse=True)
                top_k_expansions = []
                seen_cutoff_sets = set()
                for exp in all_expansions:
                    beam_idx, _, _, cand_value, _, _ = exp
                    parent_cutoffs = beams[beam_idx][0]
                    new_cutoffs_key = tuple(sorted(parent_cutoffs + [cand_value]))
                    if new_cutoffs_key in seen_cutoff_sets:
                        continue
                    seen_cutoff_sets.add(new_cutoffs_key)
                    top_k_expansions.append(exp)
                    if len(top_k_expansions) == K:
                        break

                # Print iteration results
                top_str = ", ".join([f"{exp[3]:.4f} (score={exp[2]:.4f}, beam={exp[0]+1})"
                                     for exp in top_k_expansions])
                print(f"  [TID3]   Iter {iteration}/{nb_bins-1}: top-{len(top_k_expansions)} beams = [{top_str}]")
                if top_k_expansions and top_k_expansions[0][2] == 0.0:
                    print(
                        f"  [TID3]   Zero-score reason for TPID {temporal_property_id}, "
                        f"beam candidate {top_k_expansions[0][3]:.4f}: {top_k_expansions[0][5]}"
                    )

                # Build new beams
                new_beams = []
                for beam_idx, cand_idx, score, cand_value, _, _ in top_k_expansions:
                    old_cutoffs, old_pool = beams[beam_idx]
                    new_cutoffs = sorted(old_cutoffs + [cand_value])
                    # Remove chosen candidate and near-duplicates from pool
                    new_pool = [c for c in old_pool
                                if not any(np.isclose(c, new_cutoffs))]
                    new_beams.append((new_cutoffs, new_pool))

                beams = new_beams

                # top_k_expansions[0] is the highest-scoring (non-duplicate) expansion
                # this iteration, so it produced beams[0] — the eventual winning beam.
                if top_k_expansions:
                    best_beam_per_state_stats = top_k_expansions[0][4]

            # Best beam is the first one (highest score at last iteration)
            if beams:
                chosen_cutpoints = beams[0][0]
                chosen_scores = [0.0]  # Beam search doesn't track per-iteration scores the same way
                alternative_sets = [b[0] for b in beams[1:] if b[0] != chosen_cutpoints]
            else:
                chosen_cutpoints = []
                chosen_scores = []
                alternative_sets = []

            cutpoints_str = ", ".join([f"{c:.4f}" for c in chosen_cutpoints])
            print(f"  [TID3]   Final cutpoints (beam search, K={K}): [{cutpoints_str}] → {len(chosen_cutpoints)+1} bins")
            if alternative_sets:
                print(f"  [TID3]   {len(alternative_sets)} alternative cutoff set(s) for MV refinement")

            if best_beam_per_state_stats is not None:
                self.per_state_stats[temporal_property_id] = best_beam_per_state_stats

            return chosen_cutpoints, chosen_scores, alternative_sets
'''


def _multivariate_refinement_exhaustive(self, data):
    """
    LEGACY: Refine cutoffs via joint enumeration of candidate cutoff sets across all variables.

    Enumerates every combination (best beam + alternatives) of cutoff sets over all
    variables and picks the single combination maximizing the cross-variable
    separability score. Per-pair cross_durations are cached so each unique
    (V, W, set_V, set_W) is computed only once, keeping pairwise work at
    C(N,2) * K^2 instead of K^N for the pair computation step, but the combo
    enumeration itself is still K^N which is what makes this expensive at real N.

    Depends on:
      - self._alternative_cutoff_sets (populated by the beam-search Phase 1)
      - self.boundaries (univariate-best cutoffs as the first candidate per variable)
      - self.raw_to_STIs, self._compute_cross_durations,
        self._merge_cross_durations, self._score_cross_variable
    """
    import itertools
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm
    try:
        from ..constants import TEMPORAL_PROPERTY_ID
    except ImportError:
        from ta_package.constants import TEMPORAL_PROPERTY_ID
    import logging
    logger = logging.getLogger(__name__)

    if not self.boundaries or len(self.boundaries) < 2:
        logger.info("Multivariate refinement skipped: need at least 2 variables")
        return

    # Step 1: collect candidate sets per variable
    var_list = []
    cand_sets = {}
    for v_tpid, best in self.boundaries.items():
        if not best:
            logger.info(f"  Variable {v_tpid}: empty boundaries, skipping")
            continue
        alternatives = self._alternative_cutoff_sets.get(v_tpid, [])
        var_list.append(v_tpid)
        cand_sets[v_tpid] = [best] + list(alternatives)

    if len(var_list) < 2:
        logger.info("Multivariate refinement skipped: <2 usable variables")
        return

    K = {v: len(cand_sets[v]) for v in var_list}
    total_combos = 1
    for v in var_list:
        total_combos *= K[v]
    total_pair_computations = sum(
        K[var_list[a]] * K[var_list[b]]
        for a in range(len(var_list))
        for b in range(a + 1, len(var_list))
    )

    logger.info(
        f"Multivariate refinement: {len(var_list)} variables, "
        f"K_v={[K[v] for v in var_list]}, "
        f"{total_combos} joint combos, {total_pair_computations} unique pair computations"
    )
    print(
        f"[MV_REFINE] Joint enumeration: {len(var_list)} vars, "
        f"K_v={[K[v] for v in var_list]}, "
        f"{total_combos} combos, {total_pair_computations} pair computations",
        flush=True,
    )

    # Step 2a: cache discretized intervals per (V, set_idx) — reused across pairs
    interval_cache = {}
    for v_tpid in var_list:
        v_data = data[data[TEMPORAL_PROPERTY_ID] == v_tpid]
        if len(v_data) == 0:
            for i in range(K[v_tpid]):
                interval_cache[(v_tpid, i)] = {}
            continue
        for i, cutoff_set in enumerate(cand_sets[v_tpid]):
            interval_cache[(v_tpid, i)] = self.raw_to_STIs(
                v_data, cutoff_set, v_tpid
            )

    # Step 2b: per-pair cross_durations cache
    pair_cache = {}
    pair_pbar = tqdm(
        total=total_pair_computations,
        desc="MV refinement: pair cache",
        unit="pair",
    )
    for a_idx, v_a in enumerate(var_list):
        for v_b in var_list[a_idx + 1:]:
            pair_cache[(v_a, v_b)] = {}
            for i in range(K[v_a]):
                a_intervals = interval_cache[(v_a, i)]
                for j in range(K[v_b]):
                    b_intervals = interval_cache[(v_b, j)]
                    if not a_intervals or not b_intervals:
                        pair_cache[(v_a, v_b)][(i, j)] = {}
                    else:
                        pair_cache[(v_a, v_b)][(i, j)] = self._compute_cross_durations(
                            a_intervals, b_intervals, self.entity_class
                        )
                    pair_pbar.update(1)
    pair_pbar.close()

    # Step 3: enumerate combos, merge cached pair dicts, score
    pair_indices = [
        (a, b)
        for a in range(len(var_list))
        for b in range(a + 1, len(var_list))
    ]
    ranges = [range(K[v]) for v in var_list]

    # 1. Define the worker function that handles a single combination
    def evaluate_combo(combo):
        dicts_to_merge = [
            pair_cache[(var_list[a], var_list[b])][(combo[a], combo[b])]
            for a, b in pair_indices
        ]
        merged = self._merge_cross_durations(dicts_to_merge)
        score = self._score_cross_variable(merged)
        return score, combo

    # 2. Materialize the generator into a list so joblib can chunk it
    all_combos = list(itertools.product(*ranges))

    logger.info(f"Starting parallel evaluation on {total_combos} combos...")

    # 3. Fire up the CPU cores
    # n_jobs=-1 automatically detects and uses all CPUs allocated by SLURM
    # batch_size='auto' chunks the workload efficiently to reduce overhead
    results = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(evaluate_combo)(combo)
        for combo in tqdm(all_combos, desc="MV refinement: evaluate", unit="combo")
    )

    # 4. Find the absolute winner from the parallel results
    best_score = -np.inf
    best_combo = tuple(0 for _ in var_list)

    for score, combo in results:
        if score > best_score:
            best_score = score
            best_combo = combo

    # Step 4: apply winning combo
    changes_made = 0
    for idx, v_tpid in enumerate(var_list):
        chosen_i = best_combo[idx]
        chosen_set = cand_sets[v_tpid][chosen_i]
        prev_set = self.boundaries[v_tpid]
        if chosen_set != prev_set:
            self.boundaries[v_tpid] = chosen_set
            changes_made += 1
            logger.info(
                f"  Variable {v_tpid}: cutoffs updated "
                f"(beam idx {chosen_i}, joint_score={best_score:.4f})"
            )
            print(
                f"[MV_REFINE] Variable {v_tpid}: UPDATED "
                f"before={prev_set} -> after={chosen_set} (beam={chosen_i})",
                flush=True,
            )
        else:
            print(
                f"[MV_REFINE] Variable {v_tpid}: unchanged "
                f"cutoffs={prev_set} (beam={chosen_i})",
                flush=True,
            )

    print(
        f"[MV_REFINE] Complete: {changes_made}/{len(var_list)} variables updated, "
        f"best joint_score={best_score:.4f}, combo={best_combo}",
        flush=True,
    )
    logger.info(
        f"Multivariate refinement complete: {changes_made}/{len(var_list)} variables updated, "
        f"best joint_score={best_score:.4f}"
    )
