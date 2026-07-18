import pandas as pd
import os
import argparse
import sys
import time
import config
from config import SKIP_SAME_VARIABLE, TIRP_VS_MIN, TIRP_VS_MAX

# --- Assume necessary imports succeed ---
from New_KarmaLego_Framework import RunKarmaLego


# --- KL <-> DataFrame helpers ---

def txt_2_csv(data_path):
    """ Parses a KarmaLego .txt (KL) file into a DataFrame of STIs. """
    rows = []
    entity_id = None
    with open(data_path, 'r') as file:
        lines = file.readlines()
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        if "," not in line and line.endswith(";"):
            entity_id = line.strip(";")
        elif entity_id is not None and line.endswith(";"):
            intervals = line.strip(";").split(";")
            for interval_str in intervals:
                if interval_str:
                    interval_data = interval_str.split(",")
                    if len(interval_data) == 4:
                        start_time, end_time, state_id, temporal_property_id = map(int, interval_data)
                        # Drop class/placeholder rows (state -1) early.
                        if state_id != -1:
                            rows.append([entity_id, start_time, end_time, state_id, temporal_property_id])
    return pd.DataFrame(rows, columns=["EntityID", "StartTime", "EndTime", "StateID", "TemporalPropertyID"])


def csv_2_txt(df, output_path, number_of_entities):
    """ Converts an STI DataFrame back to KarmaLego .txt (KL) format. """
    with open(output_path, 'w') as file:
        file.write("startToncepts\n")
        file.write(f"numberOfEntities,{int(number_of_entities)}\n")
        for entity_id, group in df.groupby("EntityID"):
            file.write(f"{entity_id};\n")
            intervals = []
            for _, row in group.iterrows():
                intervals.append(
                    f"{int(row['StartTime'])},{int(row['EndTime'])},"
                    f"{int(row['StateID'])},{int(row['TemporalPropertyID'])}"
                )
            file.write((";".join(intervals) + ";\n") if intervals else "\n")


def load_training_kl(abstraction_output_dir, kl_data_dir):
    """
    Loads the Stage 1 training STIs into a single combined KL file for mining.

    MARIO has no classes, so mining runs over one pool of entities read from the
    single 'Train/KL.txt' produced by Stage 1.

    Returns (combined_kl_path, total_entities).
    """
    train_dir = os.path.join(abstraction_output_dir, 'Train')
    single_kl = os.path.join(train_dir, 'KL.txt')

    if not os.path.exists(single_kl):
        print(f"ERROR: No training KL file found in {train_dir} (expected 'KL.txt').")
        sys.exit(1)
    df = txt_2_csv(single_kl)

    if df.empty:
        print("ERROR: Training KL data is empty after parsing.")
        sys.exit(1)

    total_entities = df['EntityID'].nunique()
    os.makedirs(kl_data_dir, exist_ok=True)
    combined_kl = os.path.join(kl_data_dir, 'KL.txt')
    csv_2_txt(df, combined_kl, total_entities)
    return combined_kl, total_entities


# --- Main Mining Function ---

def run_single_mining_task(abstraction_output_dir, mining_run_dir, tirp_objects_output_dir,
                           mvs, max_gap, relations, skip_followers, epsilon,
                           vs_min, vs_max):
    """
    MARIO Stage 2: mine ALL frequent TIRPs from the single training KL file and
    keep those whose vertical-support fraction is within [vs_min, vs_max].

    No event-of-interest filtering, no synthetic event insertion, no class split.
    Every retained TIRP is used in Stage 3 to build a model that forecasts the
    target variable's symbol at t + horizon.
    """
    print("--- Starting Stage 2: TIRP Mining (MARIO forecasting) ---")
    print(f"Params: mvs={mvs}, max_gap={max_gap}, relations={relations}, "
          f"skip_followers={skip_followers}, epsilon={epsilon}, "
          f"vs_range=[{vs_min}, {vs_max}]")
    print(f"Input: {abstraction_output_dir}")
    print(f"Output: {tirp_objects_output_dir}")
    start_total_time = time.time()

    os.makedirs(tirp_objects_output_dir, exist_ok=True)
    kl_data_dir = os.path.join(mining_run_dir, 'KL_data')

    # --- Step 1: Load training STIs into a single KL file ---
    print("Step 1: Loading training KL data...")
    combined_kl, total_entities = load_training_kl(abstraction_output_dir, kl_data_dir)
    print(f"  Mining over {total_entities} entities. KL file: {combined_kl}")

    # --- Step 2: Run KarmaLego ---
    print("Step 2: Running KarmaLego algorithm...")
    start_kl_time = time.time()
    lego_result, _ = RunKarmaLego.runKarmaLego(
        time_intervals_path=combined_kl,
        min_ver_support=mvs,
        num_relations=relations,
        max_gap=max_gap,
        epsilon=epsilon,
        skip_followers=skip_followers,
        max_tirp_length=8,
        label='KarmaLegoRun',
        output_path=mining_run_dir,
        print_instances=False,
        print_params=False,
        processes_num=4,
        skip_same_variable=SKIP_SAME_VARIABLE,
    )
    print(f"KarmaLego finished. Time: {time.time() - start_kl_time:.2f} seconds.")

    frequent_tirps = []
    if lego_result and hasattr(lego_result, 'frequent_tirps'):
        frequent_tirps = lego_result.frequent_tirps
    print(f"KarmaLego found {len(frequent_tirps)} frequent TIRPs.")

    # --- Step 3: Filter by vertical-support fraction and save ---
    print(f"Step 3: Filtering TIRPs by vertical-support fraction in [{vs_min}, {vs_max}]...")
    rows = []
    tirps_to_save = []
    for tirp in frequent_tirps:
        vs_count = tirp.get_vertical_support()
        vs_frac = (vs_count / total_entities) if total_entities else 0.0
        selected = (vs_min <= vs_frac <= vs_max)
        rows.append({
            'TIRP_Representation': tirp.to_string(),
            'Size': tirp.size,
            'Vertical_Support': vs_count,
            'Vertical_Support_Fraction': round(vs_frac, 4),
            'Selected': int(selected),
        })
        if selected:
            tirps_to_save.append(tirp)

    scores_df = pd.DataFrame(rows)
    scores_csv_path = os.path.join(mining_run_dir, "tirp_selection_scores.csv")
    try:
        scores_df.to_csv(scores_csv_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save scores CSV {scores_csv_path}: {e}")

    saved_count = 0
    for tirp in tirps_to_save:
        try:
            tirp.save_tirp_object(tirp_objects_output_dir)
            saved_count += 1
        except Exception as e:
            tirp_str = getattr(tirp, 'to_string', lambda: 'UnknownTIRP')()
            print(f"Warning: Failed to save TIRP object {tirp_str}: {e}")

    print(f"Selected {len(tirps_to_save)}/{len(frequent_tirps)} TIRPs; saved {saved_count} "
          f"objects to {tirp_objects_output_dir}")
    print(f"--- Finished Stage 2: TIRP Mining. Total Time: {time.time() - start_total_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 (MARIO): KarmaLego TIRP mining + vertical-support-range filtering.")

    parser.add_argument("--abstraction_output_dir", required=True)
    parser.add_argument("--mining_run_dir", required=True)
    parser.add_argument("--tirp_objects_output_dir", required=True)
    parser.add_argument("--mvs", required=True, type=float)
    parser.add_argument("--max_gap", required=True, type=int)
    parser.add_argument("--relations", required=True, type=int)
    parser.add_argument("--skip_followers", required=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--epsilon", required=True, type=int)
    parser.add_argument("--vs_min", type=float, default=TIRP_VS_MIN,
                        help="Lower bound (inclusive) on vertical-support fraction for keeping a TIRP.")
    parser.add_argument("--vs_max", type=float, default=TIRP_VS_MAX,
                        help="Upper bound (inclusive) on vertical-support fraction for keeping a TIRP.")
    # Accepted but ignored: MARIO has no event of interest. Kept for orchestrator compatibility.
    parser.add_argument("--event_symbol", required=False, default=None,
                        help="(Ignored in MARIO) retained for backward-compatible invocation.")

    args = parser.parse_args()

    run_single_mining_task(
        abstraction_output_dir=args.abstraction_output_dir,
        mining_run_dir=args.mining_run_dir,
        tirp_objects_output_dir=args.tirp_objects_output_dir,
        mvs=args.mvs,
        max_gap=args.max_gap,
        relations=args.relations,
        skip_followers=args.skip_followers,
        epsilon=args.epsilon,
        vs_min=args.vs_min,
        vs_max=args.vs_max,
    )

    sys.exit(0)
