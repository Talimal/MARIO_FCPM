"""
run_experiment.py — Multi-stage SLURM experiment orchestrator for the CPM Framework.

Pipeline overview (each stage is an independent SLURM job):
  Stage 0  — Split the dataset into cross-validation folds.
  Stage 1  — Temporal abstraction (discretization) of time-series data per fold.
  Stage 2  — TIRP (Time-Interval Relational Pattern) mining per abstraction run.
  Stage 3  — Build a predictive model for each discovered TIRP (batched).
  Stage 3.5— Validate each TIRP model on the validation split (batched).
  Stage 4  — Generate predictions for each test entity (batched).
  Stage 5  — Aggregate predictions and evaluate overall model performance.

Completion tracking uses "done files": each stage script writes an empty sentinel
file (*.done) upon successful exit. The main monitoring loop checks for these files
every cycle to decide which downstream jobs can be submitted.
"""
import os
import itertools
import subprocess
import time
import pickle
import argparse
import hashlib
import shutil
import getpass
import math # For ceiling function in batching
import pandas as pd
import glob
import csv
import logging
import json



# --- Import Configuration and Parameters ---
try:
    import config
    import experiment_params
    from experiment_params import Tirp_selection_methods
except ImportError as e: print(f"FATAL ERROR importing config/params: {e}"); exit(1)
except AttributeError as e: print(f"FATAL ERROR: Missing value in config.py: {e}"); exit(1)
except Exception as e: print(f"FATAL ERROR during import: {e}"); exit(1)



def update_tirp_selection_scores_inplace(tirp_selection_scores_path, feature_matrix_base_dir, top_k_list):
    """
    Updates the existing TIRP selection scores file in-place with validation metrics.
    Adds Score_ and Binary_ columns initialized to 0, then updates directly.
    """
    print(f"Updating TIRP selection scores in-place: {tirp_selection_scores_path}")
    
    # 1. Load existing scores
    if not os.path.exists(tirp_selection_scores_path):
        print(f"Error: Could not find {tirp_selection_scores_path}")
        return
        
    df_scores = pd.read_csv(tirp_selection_scores_path)
    
    # Assume the first column identifies the TIRP (e.g., 'TIRP_name' or 'TIRP_ID')
    tirp_identifier_col = "TIRP_name" if "TIRP_name" in df_scores.columns else df_scores.columns[0]
    
    # 2. Define the exact new methods requested from the experiment_params list
    stage3_valid_methods = ['val_AUC', 'val_AUPRC', 'val_Precision', 'val_F1']
    new_methods = [m for m in Tirp_selection_methods if m in stage3_valid_methods]
    
    if not new_methods:
        print("No validation metrics specified in Tirp_selection_methods. Skipping in-place update.")
        return
    
    # 3. Initialize Score columns to 0.0 for ALL TIRPs
    for method in new_methods:
        score_col = f"Score_{method}"
        df_scores[score_col] = 0.0
        
    # 4. Initialize Binary columns to 0 for ALL combinations
    for method in new_methods:
        for k in top_k_list:
            binary_col = f"Binary_{method}#{k}"
            df_scores[binary_col] = 0
            
    # 5. Collect validation results and update specific TIRPs
    tirp_dirs = glob.glob(os.path.join(feature_matrix_base_dir, "tirp_*"))
    valid_updates_count = 0
    
    for t_dir in tirp_dirs:
        summary_file = os.path.join(t_dir, "train_summary_metrics.csv")
        if os.path.exists(summary_file):
            try:
                summary_df = pd.read_csv(summary_file)
                if not summary_df.empty:
                    row = summary_df.iloc[0]
                    t_name = str(row['TIRP_name'])
                    
                    # Find the mask for this specific TIRP in the main dataframe
                    mask = df_scores[tirp_identifier_col].astype(str) == t_name
                    
                    if mask.any():
                        # Update the score columns ONLY for this TIRP
                        df_scores.loc[mask, "Score_val_AUC"] = float(row.get('val_AUC', 0.0))
                        df_scores.loc[mask, "Score_val_AUPRC"] = float(row.get('val_AUPRC', 0.0))
                        df_scores.loc[mask, "Score_val_Precision"] = float(row.get('val_Precision', 0.0))
                        df_scores.loc[mask, "Score_val_F1"] = float(row.get('val_F1', 0.0))
                        valid_updates_count += 1
                        
            except Exception as e:
                print(f"Warning: Failed to read/update metrics from {t_dir}: {e}")
                
    print(f"Successfully updated scores for {valid_updates_count} TIRPs.")

    # 6. Calculate Top-K and update Binary columns
    # We use Horizontal_Support as a tie-breaker if it exists, otherwise fallback to the identifier
    tie_breaker_col = "Horizontal_Support" if "Horizontal_Support" in df_scores.columns else tirp_identifier_col
    
    for method in new_methods:
        score_col = f"Score_{method}"
        
        # Sort descending by score, and descending by tie-breaker
        if tie_breaker_col in df_scores.columns:
            df_sorted = df_scores.sort_values(by=[score_col, tie_breaker_col], ascending=[False, False])
        else:
            df_sorted = df_scores.sort_values(by=[score_col], ascending=[False])
            
        for k in top_k_list:
            binary_col = f"Binary_{method}#{k}"
            
            # Extract the index of the top K rows
            top_k_indices = df_sorted.head(k).index
            
            # Set the binary column to 1 for these specific top K indices
            df_scores.loc[top_k_indices, binary_col] = 1

    # 7. Save back to the EXACT SAME file, overwriting it
    df_scores.to_csv(tirp_selection_scores_path, index=False)
    print("In-place update completed and saved.")


# --- Setup Logging for Real-time Monitoring ---
def setup_logging(base_dir):
    """Setup logging to both console and file for real-time monitoring"""
    log_dir = os.path.join(base_dir, "status")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create main experiment log file
    log_file = os.path.join(log_dir, "experiment_main.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()  # Console output
        ]
    )
    return logging.getLogger(__name__)

def update_status_file(base_dir, dataset_name, status_info):
    """Update status file for real-time monitoring"""
    status_dir = os.path.join(base_dir, "status")
    os.makedirs(status_dir, exist_ok=True)
    
    status_file = os.path.join(status_dir, "experiment_status.json")
    
    # Read existing status
    try:
        with open(status_file, 'r') as f:
            all_status = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_status = {}
    
    # Update status for current dataset
    all_status[dataset_name] = {
        **status_info,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Write updated status
    try:
        with open(status_file, 'w') as f:
            json.dump(all_status, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update status file: {e}")

def log_progress_summary(logger, dataset_name, fold_num, abs_combo, mine_combo, stage_status):
    """Log detailed progress for current combination"""
    abs_str = create_param_string(abs_combo, prefix="abs")
    mine_str = create_param_string(mine_combo, prefix="mine")
    
    logger.info(f"[{dataset_name}] Fold {fold_num} - {abs_str} - {mine_str}")
    for stage, status in stage_status.items():
        logger.info(f"  Stage {stage}: {status}")

# --- Construct Conda Activation Lines ---
conda_activation_lines = ""
if config.CONDA_ENV_NAME:
    if config.CONDA_LOAD_COMMAND: conda_activation_lines += f"{config.CONDA_LOAD_COMMAND}\n"
    conda_activation_lines += f"source activate {config.CONDA_ENV_NAME}\n"
    conda_activation_lines += f"""
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate Conda environment '{config.CONDA_ENV_NAME}'."
  exit 1
fi
"""
else: conda_activation_lines = "# Conda activation not configured."

# --- Helper Functions ---
def generate_parameter_combinations(params_dict, keys):
    """
    Generates parameter combinations. 
    If 'd_method' is in keys, it applies specific logic for Supervised Abstraction 
    (handling split_event_class and event_window).
    Otherwise, it performs a standard Cartesian product.
    """
    # Check if we need to apply Abstraction logic
    if "d_method" in keys:
        # --- Abstraction Logic ---
        base_keys = [k for k in keys if k != "split_event_class" and k != "event_window"] # Ensure we don't double count if user added them
        
        # Get base combinations (d_method, b, ig, etc.)
        # We recursively call this function but strictly for the base keys (which won't trigger this block if d_method is treated carefully, 
        # but to avoid recursion issues, we can just do the standard product here for the base)
        
        relevant_params_base = {k: params_dict[k] for k in base_keys if k in params_dict}
        if not relevant_params_base: return [{}]
        
        param_names = list(relevant_params_base.keys())
        param_values = list(relevant_params_base.values())
        base_combinations = []
        for values_tuple in itertools.product(*param_values): 
            base_combinations.append(dict(zip(param_names, values_tuple)))

        # Get dependent parameters
        split_options = params_dict.get("split_event_class", [False])
        window_options = params_dict.get("event_window", [None])
        
        final_combinations = []
        
        for base_combo in base_combinations:
            method = base_combo.get("d_method")
            if not method: 
                final_combinations.append(base_combo)
                continue
                
            # Check if method is supervised
            is_supervised = method in config.SUPERVISED_ABSTRACTION_METHODS
            
            if is_supervised:
                # Iterate over split_event_class options
                for split in split_options:
                    if split:
                        # If split is True, iterate over event_window
                        for window in window_options:
                            combo = base_combo.copy()
                            combo["split_event_class"] = True
                            combo["event_window"] = window
                            final_combinations.append(combo)
                    else:
                        # If split is False, window is ignored (None)
                        combo = base_combo.copy()
                        combo["split_event_class"] = False
                        combo["event_window"] = None
                        final_combinations.append(combo)
            else:
                # Unsupervised methods: neither split nor window applies
                combo = base_combo.copy()
                combo["split_event_class"] = False
                combo["event_window"] = None
                final_combinations.append(combo)
        
        return final_combinations

    else:
        # --- Standard Cartesian Product ---
        relevant_params = {k: params_dict[k] for k in keys if k in params_dict}
        if not relevant_params: return [{}]
        param_names = list(relevant_params.keys()); param_values = list(relevant_params.values())
        combinations = [];
        for values_tuple in itertools.product(*param_values): combinations.append(dict(zip(param_names, values_tuple)))
        return combinations

def create_param_string(param_combo, prefix=""):
    """
    Converts a parameter combination dict into a filesystem-safe, human-readable string.

    Keys whose values are None are omitted. String values are used verbatim; numeric
    values are formatted as 'key=value'. Special characters (dots, spaces, colons, etc.)
    are replaced so the result can be used safely as a directory or file name component.

    Example: {'b': 5, 'ig': 0.5} with prefix="abs" → "abs_b=5-ig=0_dot_5"
    """
    if not param_combo: return prefix if prefix else ""
    items_to_join = []
    for k, v in sorted(param_combo.items()):
        if v is None: continue # Skip None values in directory string
        if isinstance(v, str): item_str = str(v)
        else: item_str = f"{k}={v}"
        sanitized_item = str(item_str).replace('.', '_dot_').replace(' ', '_').replace(':', '-').replace('*', 'x')
        sanitized_item = "".join(c if c.isalnum() or c in ('-', '_', '=') else '_' for c in sanitized_item)
        if sanitized_item: items_to_join.append(sanitized_item)
    param_str = "-".join(items_to_join)
    if prefix: return f"{prefix}_{param_str}" if param_str else prefix
    else: return param_str

def get_sanitized_tirp_id(tirp_object_file_path):
    """
    Derives a filesystem-safe identifier string from a TIRP pickle file path.

    Strips the file extension and replaces any character that is not alphanumeric,
    a dash, or an underscore. If the result would be empty (e.g., the filename
    consisted entirely of special characters), a short SHA1 hash of the original
    path is used as a fallback to guarantee a unique, valid identifier.
    """
    try:
        tirp_filename = os.path.basename(tirp_object_file_path); tirp_id_base = os.path.splitext(tirp_filename)[0]
        sanitized_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in tirp_id_base)
        if not sanitized_id: hasher = hashlib.sha1(tirp_object_file_path.encode()); sanitized_id = f"tirp_hash_{hasher.hexdigest()[:8]}"
        return sanitized_id
    except Exception as e: print(f"Warning: Could not generate TIRP ID: {e}"); path_hash = hashlib.sha1(tirp_object_file_path.encode()).hexdigest()[:8]; return f"tirp_error_id_{path_hash}"

def is_sbatch_available(): return shutil.which("sbatch") is not None

def get_current_slurm_job_count(username):
    if not is_sbatch_available(): return 0
    try:
        command = ["squeue", "-u", username, "-h", "-o", "%t"]; result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0: print(f"Warning: 'squeue' failed code {result.returncode}."); return -1
        states = result.stdout.strip().split('\n'); count = 0
        for state in states:
            if state and (state == 'R' or state == 'PD'): count += 1
        return count
    except Exception as e: print(f"ERROR getting SLURM job count: {e}"); return -1

def get_current_slurm_memory_usage(username):
    """Get total memory usage (in GB) of running and pending SLURM jobs for the user."""
    if not is_sbatch_available(): return 0
    try:
        # Get memory allocation for each job in format: state,memory
        # Memory format from squeue can be in different units (K, M, G, T)
        command = ["squeue", "-u", username, "-h", "-o", "%t %m"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0: 
            print(f"Warning: 'squeue' failed code {result.returncode}.")
            return -1
        
        lines = result.stdout.strip().split('\n')
        total_memory_gb = 0
        
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            state = parts[0]
            memory_str = parts[1]
            
            # Only count running (R) and pending (PD) jobs
            if state not in ['R', 'PD']:
                continue
            
            # Parse memory string (can be in format like "4000M", "4G", "4000000K", etc.)
            try:
                # Remove any trailing unit and convert
                if memory_str.endswith('K'):
                    memory_gb = float(memory_str[:-1]) / (1024 * 1024)
                elif memory_str.endswith('M'):
                    memory_gb = float(memory_str[:-1]) / 1024
                elif memory_str.endswith('G'):
                    memory_gb = float(memory_str[:-1])
                elif memory_str.endswith('T'):
                    memory_gb = float(memory_str[:-1]) * 1024
                else:
                    # Assume MB if no unit specified
                    memory_gb = float(memory_str) / 1024
                
                total_memory_gb += memory_gb
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse memory string '{memory_str}': {e}")
                continue
        
        return total_memory_gb
    except Exception as e: 
        print(f"ERROR getting SLURM memory usage: {e}")
        return -1

def get_test_entity_ids(input_dir):
    """
    Extracts unique entity IDs from the first column ('EntityID') of
    'entity-class-relations.csv' located in the input_dir.
    Returns a sorted list of unique entity IDs.
    (Simplified - assumes file exists and is valid).
    """
    csv_filename = 'entity-class-relations.csv'
    csv_path = os.path.join(input_dir,'Test', csv_filename)

    # Read the 'EntityID' column directly - assumes file and column exist
    df = pd.read_csv(csv_path, usecols=['EntityID'])

    # Get unique values, convert to list, ensure strings for sorting, then sort
    unique_ids = df['EntityID'].unique().tolist()
    entity_ids = sorted([str(eid) for eid in unique_ids])

    return entity_ids

# --- Modified submit_sbatch_job ---
def submit_sbatch_job(job_name, stage_num, script_path, arguments, done_file_path, log_path, base_dir,
                      sbatch_template, conda_lines):
    """Prepares sbatch script file and submits it with --chdir in the command line."""
    if not is_sbatch_available(): print("ERROR: 'sbatch' command not found."); return False
    username = getpass.getuser()
    
    # Get memory requirement for this job
    try: 
        job_memory_gb = config.SBATCH_RESOURCES[stage_num]['mem']
    except KeyError: 
        print(f"Warning: Could not get memory requirement for stage {stage_num}. Using default 4GB.")
        job_memory_gb = 4
    
    # Rate-limit submissions based on total SLURM memory already allocated to this user.
    # Before adding a new job, we poll the cluster until the projected memory usage
    # (current + this job's requirement) would stay within config.MAX_MEMORY_GB.
    # This prevents overwhelming the cluster when many jobs are queued simultaneously.
    while True:
        current_memory_gb = get_current_slurm_memory_usage(username)
        if current_memory_gb == -1:
            print("Warning: Could not get memory usage. Proceeding.")
            break
        # Check if adding this job would exceed the memory limit
        if current_memory_gb + job_memory_gb <= config.MAX_MEMORY_GB:
            break
        else:
            print(f"Max memory ({config.MAX_MEMORY_GB}GB) would be exceeded (current: {current_memory_gb:.1f}GB, job needs: {job_memory_gb}GB). Waiting {config.JOB_SUBMISSION_WAIT_INTERVAL_SECONDS}s...")
            time.sleep(config.JOB_SUBMISSION_WAIT_INTERVAL_SECONDS)
    try: resources = config.SBATCH_RESOURCES[stage_num]; cpus, mem, time_limit = resources['cpus'], resources['mem'], resources['time_limit']
    except KeyError: print(f"ERROR: Resources for stage {stage_num} missing. Using fallback."); cpus, mem, time_limit = 1, 4, "01:00:00"
    safe_job_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in job_name); max_len = 100
    if len(safe_job_name) > max_len: hash_suffix = hashlib.sha1(job_name.encode()).hexdigest()[:8]; safe_job_name = safe_job_name[:max_len - len(hash_suffix) - 1] + "_" + hash_suffix
    full_script_path = os.path.abspath(script_path)
    if not os.path.exists(full_script_path): print(f"ERROR: Script not found: {full_script_path}."); return False
    code_directory = os.path.abspath(config.CODE_DIR)
    if not os.path.isdir(code_directory): print(f"ERROR: config.CODE_DIR not valid: {code_directory}"); return False
    sbatch_content = sbatch_template.format(job_name=safe_job_name, log_path=log_path, cpus=cpus, mem=mem, time_limit=time_limit, stage_num=stage_num, script_path=full_script_path, arguments=arguments, done_file_path=done_file_path, conda_activation_lines=conda_lines)
    sbatch_file_path = os.path.join(base_dir, "sbatch_scripts", f"{safe_job_name}.slurm")
    os.makedirs(os.path.dirname(sbatch_file_path), exist_ok=True)
    try: 
        with open(sbatch_file_path, "w") as f: f.write(sbatch_content)
    except IOError as e: print(f"ERROR writing sbatch script {sbatch_file_path}: {e}"); return False
    try:
        print(f"Submitting job: {safe_job_name} (Stage {stage_num}) -> Script: {os.path.basename(script_path)}")
        sbatch_command = ["sbatch", f"--chdir={code_directory}", sbatch_file_path]
        print(f"  Command: {' '.join(sbatch_command)}") # Optional debug
        result = subprocess.run(sbatch_command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]; print(f"  -> Submitted. Job ID: {job_id}"); return True
    except subprocess.CalledProcessError as e: print(f"ERROR submitting {safe_job_name}: {e.stderr.strip()}")
    except Exception as e: print(f"ERROR during submission {safe_job_name}: {e}")
    return False

# --- Stage-Specific Job Submission Functions ---

def submit_stage0_job(dataset_params, n_folds, dataset_base_dir, log_dir, base_dir):
    stage_num = 0; dataset_name = dataset_params['dataset_name']; stage0_done_file = os.path.join(dataset_base_dir, 'stage0_split.done')
    if not os.path.exists(stage0_done_file):
        job_name = f"stg0_{dataset_name}_stage0_split"
        try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
        except KeyError: print(f"ERROR: Wrapper path for Stage {stage_num} missing."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
        arguments = (f"--dataset_name \"{dataset_name}\" --data_path \"{dataset_params['data_path']}\" " f"--matching_path \"{dataset_params['matching_path']}\" --n_folds {n_folds} " f"--dataset_base_dir \"{dataset_base_dir}\"")
        return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage0_done_file, log_dir, base_dir, config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False

def submit_stage1_job(dataset_params, fold_num, abs_combo, fold_dir, log_dir, base_dir):
    stage_num = 1; dataset_name = dataset_params['dataset_name']; abs_param_string = create_param_string(abs_combo, prefix="abs")
    abstraction_run_dir = os.path.join(fold_dir, abs_param_string); stage1_done_file = os.path.join(abstraction_run_dir, 'stage1_abstraction.done')
    train_data_file = os.path.join(fold_dir, 'train.csv'); test_data_file = os.path.join(fold_dir, 'test.csv')
    if not os.path.exists(stage1_done_file):
        if not os.path.exists(train_data_file) or not os.path.exists(test_data_file): return False
        os.makedirs(abstraction_run_dir, exist_ok=True); job_name = f"stg1_{dataset_name}_f{fold_num}_{abs_param_string}"
        try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
        except KeyError: print(f"ERROR: Wrapper path for Stage {stage_num} missing."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
        
        # Prepare arguments including the new optional ones
        arguments = (f"--train_data_file \"{train_data_file}\" --test_data_file \"{test_data_file}\" " 
                     f"--d_method \"{abs_combo['d_method']}\" --num_of_bins {abs_combo['b']} " 
                     f"--interpolation_gap {abs_combo['ig']} --abstraction_output_dir \"{abstraction_run_dir}\"")
        
        # Add conditional arguments if they exist and are not None
        if abs_combo.get('split_event_class') is not None:
            arguments += f" --split_event_class {str(abs_combo['split_event_class'])}"
        if abs_combo.get('event_window') is not None:
             arguments += f" --event_window {abs_combo['event_window']}"

        return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage1_done_file, log_dir, base_dir, config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False

def submit_stage2_job(dataset_params, fold_num, abs_combo, mine_combo, abstraction_run_dir, log_dir, base_dir, event_symbol):
    stage_num = 2; dataset_name = dataset_params['dataset_name']; abs_param_string = create_param_string(abs_combo, prefix="abs"); mining_param_string = create_param_string(mine_combo, prefix="mine")
    mining_run_dir = os.path.join(abstraction_run_dir, mining_param_string); tirp_objects_output_dir = os.path.join(mining_run_dir, "tirps"); stage2_done_file = os.path.join(mining_run_dir, 'stage2_mining.done')
    if not os.path.exists(stage2_done_file):
        expected_stage1_output_train = os.path.join(abstraction_run_dir, 'Train', 'KL-class-0.0.txt');
        if not os.path.exists(expected_stage1_output_train): return False
        os.makedirs(tirp_objects_output_dir, exist_ok=True); job_name = f"stg2_{dataset_name}_f{fold_num}_{abs_param_string}_{mining_param_string}"
        try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
        except KeyError: print(f"ERROR: Wrapper path for Stage {stage_num} missing."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
        arguments = (f"--abstraction_output_dir \"{abstraction_run_dir}\" --mining_run_dir \"{mining_run_dir}\" " f"--tirp_objects_output_dir \"{tirp_objects_output_dir}\" --mvs {mine_combo['mvs']} " f"--max_gap {mine_combo['mg']} --relations {mine_combo['rel']} " f"--skip_followers {mine_combo['sf']} --epsilon {mine_combo['e']} " f"--event_symbol {event_symbol}")
        return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage2_done_file, log_dir, base_dir, config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False

# --- NEW Stage 3: Build Model Only ---
# --- NEW Stage 3: Build Model Only (Batched) ---
def submit_stage3_batch_job(dataset_params, fold_num, abs_combo, mine_combo, batch_num, tirp_list_file,
                           abstraction_run_dir, mining_run_dir, log_dir, base_dir, event_symbol):
    """Submits job for Stage 3: Build TIRP Model for a batch of TIRPs."""
    stage_num = 3; dataset_name = dataset_params['dataset_name']
    abs_param_string = create_param_string(abs_combo, prefix="abs")
    mining_param_string = create_param_string(mine_combo, prefix="mine")
    
    # Output directory for the built model components (base dir for all models in this mining run)
    built_model_base_dir = os.path.join(mining_run_dir, "feature_matrix")
    os.makedirs(built_model_base_dir, exist_ok=True) # Ensure base dir exists

    # Done file for the BATCH (to track submission/completion of the batch job itself)
    # We store batch done files in the mining run dir or a temp dir? 
    # Let's put them in a 'status' subdir within feature_matrix to keep it clean, or just in feature_matrix
    batch_status_dir = os.path.join(built_model_base_dir, "batch_status")
    os.makedirs(batch_status_dir, exist_ok=True)
    stage3_batch_done_file = os.path.join(batch_status_dir, f'stage3_batch_{batch_num:04d}.done')

    if not os.path.exists(stage3_batch_done_file):
         if not os.path.exists(tirp_list_file): return False
         
         # Inputs needed: Original KL files from Stage 1 might still be needed if feature matrix logic uses them
         expected_stage1_output_train_cls0 = os.path.join(abstraction_run_dir, 'Train', 'KL-class-0.0.txt')
         if not os.path.exists(expected_stage1_output_train_cls0): return False
         
         job_name = f"stg3_{dataset_name}_f{fold_num}_{abs_param_string}_{mining_param_string}_batch_{batch_num}"
         try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
         except KeyError: print(f"ERROR: Wrapper path for Stage {stage_num} missing."); return False
         if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
         
         # Arguments for run_stage3_build_model.py (Updated for batching):
         arguments = (f"--abstraction_output_dir \"{abstraction_run_dir}\" " # Needed for original KL files?
                      f"--tirp_model_run_dir \"{built_model_base_dir}\" "    # Base dir where 'tirp_X' folders will be created
                      f"--tirp_list_file \"{tirp_list_file}\" "              # Path to file containing list of TIRPs to process
                      f"--max_gap {mine_combo['mg']} "
                      f"--num_relations {mine_combo['rel']} "
                      f"--epsilon {mine_combo['e']} "
                      f"--event_symbol {event_symbol} ")
         
         if getattr(config, 'BUILD_CPML', False):
             arguments += "--build_cpml "
             
         return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage3_batch_done_file, log_dir, base_dir,
                                  config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False

# --- NEW Stage 3.5: Validation ---
def submit_stage3_5_batch_job(dataset_params, fold_num, abs_combo, mine_combo, batch_num, tirp_list_file,
                           abstraction_run_dir, mining_run_dir, log_dir, base_dir, event_symbol):
    """Submits job for Stage 3.5: Validation for a batch of TIRPs."""
    stage_num = 3.5 # Specific SLURM resources
    dataset_name = dataset_params['dataset_name']
    abs_param_string = create_param_string(abs_combo, prefix="abs")
    mining_param_string = create_param_string(mine_combo, prefix="mine")
    
    built_model_base_dir = os.path.join(mining_run_dir, "feature_matrix")
    
    batch_status_dir = os.path.join(built_model_base_dir, "batch_status_val")
    os.makedirs(batch_status_dir, exist_ok=True)
    stage3_5_batch_done_file = os.path.join(batch_status_dir, f'stage3_5_batch_{batch_num:04d}.done')

    if not os.path.exists(stage3_5_batch_done_file):
         if not os.path.exists(tirp_list_file): return False
         
         job_name = f"stg3_5_{dataset_name}_f{fold_num}_{abs_param_string}_{mining_param_string}_batch_{batch_num}"
         script_path = os.path.join(config.CODE_DIR, "run_stage3_5_validation.py")
         if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False
         
         # Retrieve parameters
         epsilon_val = config.EPSILON_FCPM
         tte_w_list = dataset_params.get('TTE_W', [10000])
         ew_w = tte_w_list[0] if len(tte_w_list) > 0 else 10000
         e_w_list = dataset_params.get('e_w', [0])
         ew_e = e_w_list[0] if len(e_w_list) > 0 else 0

         arguments = (f"--tirp_model_run_dir \"{built_model_base_dir}\" "
                      f"--tirp_list_file \"{tirp_list_file}\" "
                      f"--abstraction_output_dir \"{abstraction_run_dir}\" "
                      f"--epsilon {epsilon_val} "
                      f"--ew_window_size {ew_w} "
                      f"--ew_early_warning_value {ew_e} ")
         
         return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage3_5_batch_done_file, log_dir, base_dir,
                                  config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False

# --- NEW Stage 4: Predict per Entity Batch ---
def submit_stage4_job(dataset_params, fold_num, abs_combo, mine_combo, batch_num, total_batches, entity_list_file,
                      abstraction_run_dir, mining_run_dir, log_dir, base_dir):
    """Submits job for Stage 4: Predict for a batch of entities."""
    stage_num = 4; dataset_name = dataset_params['dataset_name']; abs_param_string = create_param_string(abs_combo, prefix="abs"); mining_param_string = create_param_string(mine_combo, prefix="mine")

    # Output directory for predictions for this batch
    prediction_base_dir = os.path.join(mining_run_dir, "predictions")
    # Done files stored separately from prediction data
    batch_status_dir = os.path.join(mining_run_dir, "batch_status_stage4")
    os.makedirs(batch_status_dir, exist_ok=True)
    stage4_done_file = os.path.join(batch_status_dir, f'stage4_predict_batch_{batch_num:04d}.done')

    if not os.path.exists(stage4_done_file):
        job_name = f"stg4_{dataset_name}_f{fold_num}_{abs_param_string}_{mining_param_string}_predict_b{batch_num}"
        try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
        except KeyError: print(f"ERROR: Wrapper path for Stage {stage_num} missing."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper script not found: {script_path}."); return False

        # Directory where Stage 3 saved the built models for all TIRPs of this mining run
        built_models_base_dir = os.path.join(mining_run_dir, "feature_matrix")

        # Arguments for run_stage4_predict_entities.py
        arguments = (f"--entity_list_file \"{entity_list_file}\" "          # File containing entities for this batch
                     f"--built_models_base_dir \"{built_models_base_dir}\" "  # Dir with all built TIRP models
                     f"--prediction_output_dir \"{prediction_base_dir}\" ") # Where to save this batch's predictions

        return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage4_done_file, log_dir, base_dir,
                                 config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
    return False




def submit_stage5_job(dataset_params, fold_num, abs_combo, mine_combo, model_combo, tirp_selection_method,
                     mining_run_dir, log_dir, base_dir, model_nickname="FCPM"): # Added model_nickname
    """Submits job for Stage 5: Aggregation and Evaluation."""
    stage_num = 5; dataset_name = dataset_params['dataset_name']
    # Generate param strings needed for job name, output file name, and potentially passing to wrapper
    abs_param_string = create_param_string(abs_combo)
    mining_param_string = create_param_string(mine_combo)
    model_param_string = create_param_string(model_combo)
    results_dir = os.path.join(base_dir,dataset_name, "results") # Renamed for clarity
    tte_w_list = dataset_params.get('TTE_W', [])
    e_w_list = dataset_params.get('e_w', [])
    tte_w_str = ",".join(map(str, tte_w_list))
    e_w_str = ",".join(map(str, e_w_list))
 
    # Final results for this aggregation method and TIRP selection method go here
    final_results_output_dir = os.path.join(mining_run_dir, "results") # Renamed for clarity
    final_result_file_base = os.path.join(final_results_output_dir, f"{model_nickname}_{model_param_string}_{tirp_selection_method}") # Base name for output
    # Done file indicates this specific aggregation/evaluation finished
    stage5_done_file = f"{final_result_file_base}{e_w_str}.done"

    # Directory containing the raw prediction outputs from all Stage 4 batches
    prediction_base_dir = os.path.join(mining_run_dir, f"predictions_{model_nickname}")

    if not os.path.exists(stage5_done_file):
        os.makedirs(final_results_output_dir, exist_ok=True)
        job_name = f"stg5_{dataset_name}_{model_nickname}_f{fold_num}_abs_{abs_param_string}_mine_{mining_param_string}_agg_{model_param_string}_{tirp_selection_method}" # Include all param strings
        try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_num]; script_path = os.path.join(config.CODE_DIR, script_basename)
        except KeyError: print(f"ERROR: Wrapper path Stage {stage_num} missing."); return False
        if not os.path.exists(script_path): print(f"ERROR: Wrapper not found: {script_path}."); return False

        # --- Prepare Arguments for Stage 5 Wrapper ---
        arguments_list = []
        # General Info
        arguments_list.append(f"--dataset_name \"{dataset_name}\"")
        arguments_list.append(f"--fold_num {fold_num}")
        # Abstraction Params (pass individually)
        for key, value in sorted(abs_combo.items()):
            arguments_list.append(f"--abs_{key} \"{value}\"") # Add prefix 'abs_'
        # Mining Params (pass individually)
        for key, value in sorted(mine_combo.items()):
            arguments_list.append(f"--mine_{key} \"{value}\"") # Add prefix 'mine_'
        # Aggregation/Eval Params (pass individually)
        for key, value in sorted(model_combo.items()):
            arguments_list.append(f"--agg_{key} \"{value}\"") # Add prefix 'agg_'
        
        arguments_list.append(f"--tirp_selection_method \"{tirp_selection_method}\"")
        arguments_list.append(f"--TTE_W_list \"{tte_w_str}\"")
        arguments_list.append(f"--e_w_list \"{e_w_str}\"")
        arguments_list.append(f"--model_type \"{model_nickname}\"") # Added model_type argument
        
        # Paths
        arguments_list.append(f"--prediction_base_dir \"{prediction_base_dir}\"") # Input: Where prediction files are
        arguments_list.append(f"--output_csv_path \"{final_results_output_dir}\"") # Output: Specific CSV for this run
        arguments_list.append(f"--results_dir \"{results_dir}\"") # Output: Specific CSV for this run


        # Join arguments
        arguments = " ".join(arguments_list)
        # --- End Argument Preparation ---

        return submit_sbatch_job(job_name, stage_num, script_path, arguments, stage5_done_file, log_dir, base_dir,
                                config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)
def submit_cleanup_job(dataset_name, fold_num, fold_dir_path, log_dir, base_dir):
    """Submits job for Cleanup of fold directory."""
    # This is not a formal numeric stage, but a named task
    stage_key = "cleanup" # Matches config.WRAPPER_SCRIPT_PATHS key
    job_name = f"cleanup_{dataset_name}_f{fold_num}"
    
    cleanup_done_file = os.path.join(log_dir, f"{job_name}.done")
    
    try: script_basename = config.WRAPPER_SCRIPT_PATHS[stage_key]; script_path = os.path.join(config.CODE_DIR, script_basename)
    except KeyError: print(f"ERROR: Wrapper path for '{stage_key}' missing."); return False
    if not os.path.exists(script_path): print(f"ERROR: Wrapper not found: {script_path}."); return False

    arguments = f"--fold_dir \"{fold_dir_path}\""

    # We submit it as a job, using the resources defined for 'cleanup'
    # Note: submit_sbatch_job expects stage_num (int) for config loopups often, 
    # but we adapted config to have string key resources?
    # Actually submit_sbatch_job uses 'stage_num' to look up SBATCH_RESOURCES.
    # So we need to make sure SBATCH_RESOURCES has 'cleanup' key (done).
    
    return submit_sbatch_job(job_name, stage_key, script_path, arguments, cleanup_done_file, log_dir, base_dir,
                            config.SBATCH_SCRIPT_TEMPLATE, conda_activation_lines)

import os
import glob
import csv
import uuid
import shutil

def merge_results_files(input_dir_path: str) -> bool:
    """
    Merges all applicable CSV files from input_dir_path into 'summery_result_all.csv'.
    Uses an Atomic Replace pattern to prevent data loss or file corruption during crashes.
    """
    print('Starting safe merge of results...')
    output_filename = "summery_result_all.csv"
    output_file_full_path = os.path.join(input_dir_path, output_filename)

    if not os.path.isdir(input_dir_path):
        return False

    # Find all CSV files in the input directory
    all_csv_in_dir = glob.glob(os.path.join(input_dir_path, '*.csv'))
    
    source_csv_files_to_process = []
    try:
        # Normalize paths for robust comparison
        norm_output_path = os.path.normpath(output_file_full_path)
        for f_path in all_csv_in_dir:
            if os.path.normpath(f_path) != norm_output_path:
                source_csv_files_to_process.append(f_path)
    except Exception: 
        return False

    if not source_csv_files_to_process:
        # If no source files, check if the summary file already exists.
        if os.path.exists(output_file_full_path):
            print(f"  Summary file '{output_filename}' already exists. No new files to merge.")
            return True
        return False

    # Create a temporary working directory for atomic operations
    # Placing it inside input_dir_path ensures it is on the same filesystem (required for os.replace)
    transaction_id = uuid.uuid4().hex
    tmp_dir = os.path.join(input_dir_path, f".merge_txn_{transaction_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    tmp_output_file = os.path.join(tmp_dir, output_filename)
    header_written = False

    try:
        with open(tmp_output_file, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)

            # 1. Preserve existing data: If summary exists, copy it to the temp file first
            if os.path.exists(output_file_full_path):
                with open(output_file_full_path, 'r', newline='', encoding='utf-8') as infile:
                    csv_reader = csv.reader(infile)
                    for row in csv_reader:
                        csv_writer.writerow(row)
                        header_written = True

            # 2. Append all new source files to the temp file
            for csv_file_path in source_csv_files_to_process:
                try:
                    with open(csv_file_path, 'r', newline='', encoding='utf-8') as infile:
                        csv_reader = csv.reader(infile)
                        try:
                            header = next(csv_reader) 
                            if not header_written:
                                csv_writer.writerow(header)
                                header_written = True
                            
                            for row in csv_reader: 
                                csv_writer.writerow(row)
                        except StopIteration: 
                            pass
                except Exception:
                    continue # Skip files that cannot be read

        # 3. Atomic Commit: Replace the old summary with the newly constructed one.
        # This guarantees that a crash here will not corrupt the existing file.
        os.replace(tmp_output_file, output_file_full_path)

    except Exception as e:
        # Rollback on critical failure (e.g., full disk, permission issues)
        print(f"Error during safe merge: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # 4. Cleanup: Only delete source CSVs after a successful atomic commit
    for file_to_delete in source_csv_files_to_process:
        try:
            os.remove(file_to_delete)
        except OSError:
            pass
            
    # Remove the temporary transaction directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
        
    return True


def generate_aggregated_results(input_dir_path: str) -> bool:
    """
    Reads summery_result_all.csv and writes summery_result_aggregated.csv with
    mean AUC and mean AUPRC (plus std and fold count) across folds for each combination.
    """
    import pandas as pd

    summary_path = os.path.join(input_dir_path, "summery_result_all.csv")
    output_path = os.path.join(input_dir_path, "summery_result_aggregated.csv")

    if not os.path.exists(summary_path):
        print(f"  Cannot aggregate: {summary_path} not found.")
        return False

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"  Error reading summary CSV: {e}")
        return False

    if df.empty:
        print("  Summary CSV is empty, skipping aggregation.")
        return False

    group_cols = [c for c in df.columns if c not in ("fold_num", "AUC", "AUPRC")]

    try:
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_folds=("AUC", "count"),
                mean_AUC=("AUC", "mean"),
                std_AUC=("AUC", "std"),
                mean_AUPRC=("AUPRC", "mean"),
                std_AUPRC=("AUPRC", "std"),
            )
            .reset_index()
        )
    except Exception as e:
        print(f"  Error during aggregation: {e}")
        return False

    transaction_id = uuid.uuid4().hex
    tmp_dir = os.path.join(input_dir_path, f".agg_txn_{transaction_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_output = os.path.join(tmp_dir, "summery_result_aggregated.csv")

    try:
        agg.to_csv(tmp_output, index=False)
        os.replace(tmp_output, output_path)
    except Exception as e:
        print(f"  Error writing aggregated results: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Aggregated results written to {output_path} ({len(agg)} rows).")
    return True


# --- Main Orchestration Function (Updated for 5 Stages) ---
def run_full_experiment(datasets_list, n_folds, base_dir):
    if not is_sbatch_available(): print("CRITICAL ERROR: sbatch command not found."); return
    
    # Setup logging and monitoring
    logger = setup_logging(base_dir)
    
    start_time = time.time()
    logger.info(f"=== EXPERIMENT STARTED ===")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: Output={os.path.abspath(base_dir)}, Folds={n_folds}, Code={os.path.abspath(config.CODE_DIR)}, MaxJobs={config.MAX_CONCURRENT_JOBS}")
    logger.info(f"Total datasets to process: {len(datasets_list)}")
    
    log_dir = os.path.join(base_dir, "logs"); sbatch_scripts_dir = os.path.join(base_dir, "sbatch_scripts"); temp_dir = os.path.join(base_dir, "temp_files")
    os.makedirs(log_dir, exist_ok=True); os.makedirs(sbatch_scripts_dir, exist_ok=True); os.makedirs(temp_dir, exist_ok=True)

    abstraction_params_keys = ["d_method", "b", "ig"]; mining_params_keys = ["mvs", "mg", "rel", "sf", "e"]; model_params_keys = ["aggregation_method", "num_tirps_for_selection"]
    total_datasets = len(datasets_list)

    # --- Setup logger ---
    logger = setup_logging(base_dir)

    for i, dataset_params in enumerate(datasets_list):
        dataset_name = dataset_params.get('dataset_name', f'dataset_{i+1}')
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING DATASET: {dataset_name} ({i+1}/{total_datasets})")
        logger.info(f"{'='*60}")
        
        dataset_base_dir = os.path.join(base_dir, dataset_name); os.makedirs(dataset_base_dir, exist_ok=True)
        event_symbol = dataset_params.get('event_symbol', config.DEFAULT_EVENT_SYMBOL)
        logger.info(f"Dataset directory: {dataset_base_dir}")
        logger.info(f"Event symbol: {event_symbol}")
        
        # Update status file
        update_status_file(base_dir, dataset_name, {
            'status': 'starting',
            'stage': 'initialization',
            'progress': f"{i+1}/{total_datasets}",
            'dataset_dir': dataset_base_dir
        })

        # --- Stage 0 ---
        logger.info(f"Checking Stage 0 (fold splitting)...")
        stage0_done_file = os.path.join(dataset_base_dir, 'stage0_split.done')
        
        update_status_file(base_dir, dataset_name, {
            'status': 'running',
            'stage': 'stage0_check',
            'stage0_done': os.path.exists(stage0_done_file)
        })
        
        if not os.path.exists(stage0_done_file):
             logger.info(f"Stage 0 not complete. Submitting Stage 0 job...")
             submitted_stage0 = submit_stage0_job(dataset_params, n_folds, dataset_base_dir, log_dir, base_dir)
             if not submitted_stage0 and not os.path.exists(stage0_done_file): 
                 logger.error(f"Stage 0 submission failed. Skipping dataset {dataset_name}.")
                 update_status_file(base_dir, dataset_name, {'status': 'failed', 'stage': 'stage0', 'error': 'submission_failed'})
                 continue
        
        if not os.path.exists(stage0_done_file):
             logger.info(f"Waiting for Stage 0 completion...")
             wait_start_time = time.time(); max_wait_stage0 = config.MAX_WAIT_STAGE0_SECONDS
             try:
                 while not os.path.exists(stage0_done_file):
                     elapsed = time.time() - wait_start_time
                     if elapsed > max_wait_stage0: raise TimeoutError
                     if int(elapsed) % 60 == 0:  # Log every minute
                         logger.info(f"Still waiting for Stage 0... ({elapsed:.0f}s elapsed)")
                     time.sleep(30)
             except TimeoutError: 
                 logger.error(f"Timeout waiting for Stage 0. Skipping dataset {dataset_name}.")
                 update_status_file(base_dir, dataset_name, {'status': 'failed', 'stage': 'stage0', 'error': 'timeout'})
                 continue
             except KeyboardInterrupt: 
                 logger.info("Interrupted by user.")
                 continue
             logger.info(f"Stage 0 completed successfully.")
        else: 
            logger.info(f"Stage 0 already completed.")

        # --- Generate Combinations ---
        logger.info(f"Generating parameter combinations...")
        try:

            abstraction_combinations = generate_parameter_combinations(dataset_params, abstraction_params_keys)
            mining_combinations = generate_parameter_combinations(dataset_params, mining_params_keys)
            model_combinations = generate_parameter_combinations(dataset_params, model_params_keys)
            if not abstraction_combinations: 
                logger.warning(f"No abstraction combinations found. Skipping dataset {dataset_name}.")
                update_status_file(base_dir, dataset_name, {'status': 'skipped', 'stage': 'combination_generation', 'error': 'no_combinations'})
                continue
            
            total_abs = len(abstraction_combinations)
            total_mine = len(mining_combinations)
            total_model = len(model_combinations) if model_combinations != [{}] else 0
            total_tirp_methods = len(Tirp_selection_methods)
            
            logger.info(f"Generated combinations:")
            logger.info(f"  - Abstraction: {total_abs}")
            logger.info(f"  - Mining: {total_mine}")
            logger.info(f"  - Model: {total_model}")
            logger.info(f"  - TIRP selection methods: {total_tirp_methods}")
            
            update_status_file(base_dir, dataset_name, {
                'status': 'running',
                'stage': 'combinations_generated',
                'combinations': {
                    'abstraction': total_abs,
                    'mining': total_mine,
                    'model': total_model,
                    'tirp_methods': total_tirp_methods
                }
            })
            
        except Exception as e: 
            logger.error(f"Error generating combinations: {e}. Skipping dataset {dataset_name}.")
            update_status_file(base_dir, dataset_name, {'status': 'failed', 'stage': 'combination_generation', 'error': str(e)})
            continue

        # --- Determine Final Stage and Expected Done Files ---
        final_stage_to_check = -1
        # Check in reverse order of dependency
        if model_combinations and model_combinations != [{}]: final_stage_to_check = 5
        elif mining_combinations and mining_combinations != [{}]: final_stage_to_check = 4 # If no aggregation, check if predictions finished
        elif abstraction_combinations: final_stage_to_check = 1 # If only abstraction
        else: 
            logger.warning(f"No combinations to monitor for dataset {dataset_name}.")
            update_status_file(base_dir, dataset_name, {'status': 'skipped', 'stage': 'monitoring', 'error': 'no_combinations'})
            continue

        logger.info(f"Will monitor progress up to Stage {final_stage_to_check} completion.")
        
        update_status_file(base_dir, dataset_name, {
            'status': 'running',
            'stage': 'monitoring_setup',
            'final_stage': final_stage_to_check
        })

        # --- Monitoring Loop for Stages 1-5 ---
        logger.info(f"Entering monitoring loop...")
        
        # --- Monitoring state tracking ---
        # submitted_jobs_stageN: Sets of unique job IDs (tuples of fold+params) that have
        #   already been sent to SLURM in this session. Prevents double-submission if the
        #   orchestrator restarts or a cycle re-scans the same combination. The authoritative
        #   completion signal is always the *.done file on disk, not membership in these sets.
        # stage4_trigger_status: Keyed by job_id_stage2 (fold+abs+mine tuple). Stores whether
        #   Stage 4 entity-batch jobs have been dispatched for a given mining configuration,
        #   plus the expected number of batches — used to check Stage 4 completion later.
        # counted_combinations: Tracks which (fold, abs, mine) combos have already been
        #   counted as "complete" to avoid incrementing the progress counter every cycle.
        active_monitoring = True
        submitted_jobs_stage1 = set(); submitted_jobs_stage2 = set()
        submitted_jobs_stage3 = set(); submitted_jobs_stage3_5 = set()
        submitted_jobs_stage4 = {}   # key=(fold,abs,mine) → set of submitted batch IDs
        submitted_jobs_stage5 = set()
        stage4_trigger_status = {}   # key=(fold,abs,mine) → {'triggered', 'num_batches', 'entity_ids'}
        counted_combinations = set()
        deleted_feature_matrices = set()  # Track feature_matrix dirs already cleared
        monitoring_cycles, consecutive_idle_cycles = 0, 0
        max_monitoring_cycles, max_idle_cycles = config.MAX_MONITORING_CYCLES, config.MAX_IDLE_CYCLES

        while active_monitoring:
            monitoring_cycles += 1;
            cycle_start_time = time.time()
            
            if monitoring_cycles > max_monitoring_cycles: 
                logger.error(f"Maximum monitoring cycles ({max_monitoring_cycles}) reached. Stopping monitoring.")
                update_status_file(base_dir, dataset_name, {'status': 'failed', 'stage': 'monitoring', 'error': 'max_cycles_reached'})
                active_monitoring = False; break
                
            new_jobs_submitted_this_cycle = False
            logger.info(f"\n[CYCLE {monitoring_cycles}] {time.strftime('%Y-%m-%d %H:%M:%S')} - Scanning all combinations...")
            
            # Pessimistic scan: start each cycle assuming everything is done.
            # Any combination that still has an unmet dependency will flip this flag to
            # False, keeping the monitoring loop alive for another cycle.
            all_expected_final_jobs_done = True
            
            # Track progress for this cycle
            cycle_progress = {
                'total_combinations': 0,
                'completed_combinations': 0,
                'stage_stats': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }

            for fold_num in range(1, n_folds + 1):
                fold_dir = os.path.join(dataset_base_dir, f'fold_{fold_num}');
                if not os.path.isdir(fold_dir) or not abstraction_combinations: continue

                for abs_combo in abstraction_combinations:
                    abs_param_string = create_param_string(abs_combo, prefix="abs"); abstraction_run_dir = os.path.join(fold_dir, abs_param_string)
                    stage1_done_file = os.path.join(abstraction_run_dir, 'stage1_abstraction.done')
                    job_id_stage1 = (fold_num, tuple(sorted(abs_combo.items()))) # Unique ID for stage 1 run
                    
                    cycle_progress['total_combinations'] += 1
                    combination_complete = True

                    # Track current combination status
                    combo_status = {
                        'stage1': 'checking',
                        'stage2': 'pending',
                        'stage3': 'pending', 
                        'stage4': 'pending',
                        'stage5': 'pending'
                    }

                    # --- Submit Stage 1 ---
                    if job_id_stage1 not in submitted_jobs_stage1 and not os.path.exists(stage1_done_file):
                        if submit_stage1_job(dataset_params, fold_num, abs_combo, fold_dir, log_dir, base_dir): 
                            submitted_jobs_stage1.add(job_id_stage1); new_jobs_submitted_this_cycle = True
                            logger.info(f"  Submitted Stage 1: Fold {fold_num}, {abs_param_string}")

                    if not os.path.exists(stage1_done_file): 
                        all_expected_final_jobs_done = False; combination_complete = False
                        combo_status['stage1'] = 'waiting'
                        continue # Wait for stage 1
                    else:
                        combo_status['stage1'] = 'done'
                        cycle_progress['stage_stats'][1] += 1
                        
                    if not mining_combinations or mining_combinations == [{}]: 
                        combo_status['stage2'] = 'skipped'
                        continue # Skip mining if no combos

                    for mine_combo in mining_combinations:
                        mining_param_string = create_param_string(mine_combo, prefix="mine"); mining_run_dir = os.path.join(abstraction_run_dir, mining_param_string)
                        stage2_done_file = os.path.join(mining_run_dir, 'stage2_mining.done')
                        job_id_stage2 = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items()))) # Unique ID for stage 2 run

                        # --- Submit Stage 2 ---
                        if job_id_stage2 not in submitted_jobs_stage2 and not os.path.exists(stage2_done_file):
                            if submit_stage2_job(dataset_params, fold_num, abs_combo, mine_combo, abstraction_run_dir, log_dir, base_dir, event_symbol): 
                                submitted_jobs_stage2.add(job_id_stage2); new_jobs_submitted_this_cycle = True
                                logger.info(f"  Submitted Stage 2: Fold {fold_num}, {abs_param_string}, {mining_param_string}")

                        if not os.path.exists(stage2_done_file): 
                            all_expected_final_jobs_done = False; combination_complete = False
                            combo_status['stage2'] = 'waiting'
                            continue # Wait for stage 2
                        else:
                            combo_status['stage2'] = 'done'
                            cycle_progress['stage_stats'][2] += 1

                        # --- Stage 2 Done: Submit Stage 3 (Build Model) for each TIRP (BATCHED) ---
                        tirp_objects_dir = os.path.join(mining_run_dir, "tirps"); built_models_base_dir = os.path.join(mining_run_dir, "feature_matrix")
                        try: tirp_object_files = [f for f in os.listdir(tirp_objects_dir) if f.endswith(".pkl")] if os.path.isdir(tirp_objects_dir) else []
                        except Exception as e: print(f" WARN listing TIRPs: {e}"); tirp_object_files = []

                        all_stage3_done_for_this_run = True # Assume true until a missing one is found
                        expected_stage3_jobs_count = len(tirp_object_files)
                        
                        # Sentinel survives feature_matrix/ deletion (it lives in mining_run_dir),
                        # so prefer it when present to avoid re-scanning a wiped feature_matrix/.
                        stage3_sentinel_path = os.path.join(mining_run_dir, "stage3_all_built.done")

                        if os.path.exists(stage3_sentinel_path):
                            all_stage3_done_for_this_run = True # Already fully built in a prior cycle
                        elif expected_stage3_jobs_count == 0:
                             all_stage3_done_for_this_run = True # No TIRPs means Stage 3 is "done" (skipped)
                        else:
                            # 1. GROUP TIRPS INTO BATCHES
                            batch_size = config.TIRP_BATCH_SIZE if hasattr(config, 'TIRP_BATCH_SIZE') else 1
                            if batch_size < 1: batch_size = 1 # Safety check
                            
                            # Sort for deterministic batching
                            # Stage 3
                            tirp_object_files.sort()
                            
                            batch_size = getattr(config, 'TIRP_BATCH_SIZE_STAGE3', getattr(config, 'TIRP_BATCH_SIZE', 1))
                            num_batches = math.ceil(expected_stage3_jobs_count / batch_size) if batch_size > 0 else 1
                            # Create a unique key for this mining configuration to track batches
                            mining_config_key = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())))
                            if mining_config_key not in submitted_jobs_stage3: submitted_jobs_stage3.add(mining_config_key) # Just marking the config as active/seen? Actually checking batches below.

                            # We need to track which BATCHES have been submitted.
                            # Re-using submitted_jobs_stage3_batches set if defined, else define it outside or use a key
                            # For now, let's just check the done file of the batch to decide submission.
                            
                            temp_batch_dir = os.path.join(temp_dir, dataset_name, f'fold_{fold_num}', abs_param_string, mining_param_string, 'stage3_batches')
                            os.makedirs(temp_batch_dir, exist_ok=True)

                            for i in range(num_batches):
                                start_idx = i * batch_size
                                end_idx = min(start_idx + batch_size, expected_stage3_jobs_count)
                                batch_files = tirp_object_files[start_idx:end_idx]
                                batch_num = i + 1
                                
                                # Check if ALL TIRPs in this batch are already done (optimization)
                                # If all individual done files exist, we don't need to submit this batch?
                                # Ideally yes, but let's trust the batch done file or re-submit if missing to be safe and simple.
                                # Actually, checking individual files is safer if a batch crashed halfway.
                                # But `run_stage3_build_model` with `submit_stage3_batch_job` uses a `stage3_batch_X.done` file.
                                # If we rely ONLY on batch done file, we might miss partial failures.
                                # However, for submission logic, let's use the batch done file.
                                # For "all_stage3_done_for_this_run", we scan individual TIRPs below.
                                
                                # Prepare list file
                                batch_list_filename = f"batch_{batch_num:04d}_tirps.txt"
                                batch_list_path = os.path.join(temp_batch_dir, batch_list_filename)
                                
                                # Check submission status (using batch done file availability in submit function)
                                # Uniquely identify this batch job for the `submitted_jobs` set
                                job_id_stage3_batch = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())), f"batch_{batch_num}")
                                
                                if job_id_stage3_batch not in submitted_jobs_stage3:
                                    # Write batch file if not already submitted in this session (or if we need to check done file)
                                    # We write it every time before submission check to be safe
                                    with open(batch_list_path, 'w') as f_list:
                                        for t_file in batch_files:
                                            full_path = os.path.join(tirp_objects_dir, t_file)
                                            f_list.write(f"{full_path}\n")

                                    if submit_stage3_batch_job(dataset_params, fold_num, abs_combo, mine_combo, batch_num, batch_list_path,
                                                              abstraction_run_dir, mining_run_dir, log_dir, base_dir, event_symbol):
                                        submitted_jobs_stage3.add(job_id_stage3_batch)
                                        new_jobs_submitted_this_cycle = True
                            
                            # 2. CHECK OVERALL COMPLETION (Scan individual TIRPs)
                            # Two-level done-file tracking: batch done files (stage3_batch_XXXX.done)
                            # are used only for submission de-duplication above. To verify actual
                            # completion we scan per-TIRP done files (stage3_build_<id>.done).
                            # This catches partial failures where a batch job exited successfully
                            # but individual TIRP scripts inside it failed silently.
                            
                            for tirp_filename in tirp_object_files:
                                full_tirp_file_path = os.path.join(tirp_objects_dir, tirp_filename); sanitized_tirp_id = get_sanitized_tirp_id(full_tirp_file_path)
                                if not sanitized_tirp_id: continue
                                
                                built_model_dir = os.path.join(built_models_base_dir, f'tirp_{sanitized_tirp_id}')
                                stage3_done_path = os.path.join(built_model_dir, f'stage3_build_{sanitized_tirp_id}.done')
                                
                                if not os.path.exists(stage3_done_path):
                                     all_stage3_done_for_this_run = False
                                     break # Found a missing one, so stage 3 is not fully done

                        if not all_stage3_done_for_this_run: 
                            all_expected_final_jobs_done = False; combination_complete = False
                            combo_status['stage3'] = 'waiting'
                            # If we are waiting, and no jobs were submitted this cycle, it means either:
                            # 1. Jobs are running.
                            # 2. Jobs failed but batch done file exists (stuck).
                            # For now, assume standard flow.
                            continue # Wait for stage 3 models to build
                        else:
                            combo_status['stage3'] = 'done'
                            cycle_progress['stage_stats'][3] += 1

                        # --- Stage 3 Done: Submit Stage 3.5 (Validation) for each TIRP (BATCHED) ---
                        all_stage3_5_done_for_this_run = True # Assume true until a missing one is found
                        
                        if getattr(config, 'RUN_STAGE3_5_VALIDATION', True):
                            if os.path.exists(stage3_sentinel_path):
                                all_stage3_5_done_for_this_run = True # Already validated in a prior cycle (feature_matrix may be wiped)
                            elif expected_stage3_jobs_count == 0:
                                 all_stage3_5_done_for_this_run = True # No TIRPs means Stage 3.5 is skipped
                            else:
                                # Submission Logic
                                batch_size_3_5 = getattr(config, 'TIRP_BATCH_SIZE_STAGE3_5', 10)
                                num_batches_3_5 = math.ceil(expected_stage3_jobs_count / batch_size_3_5) if batch_size_3_5 > 0 else 1
                                temp_batch_dir = os.path.join(temp_dir, dataset_name, f'fold_{fold_num}', abs_param_string, mining_param_string, 'stage3_5_batches')
                                os.makedirs(temp_batch_dir, exist_ok=True)

                                for i in range(num_batches_3_5):
                                    start_idx = i * batch_size_3_5
                                    end_idx = min(start_idx + batch_size_3_5, expected_stage3_jobs_count)
                                    batch_files = tirp_object_files[start_idx:end_idx]
                                    
                                    batch_num = i + 1
                                    batch_list_filename = f"batch_{batch_num:04d}_tirps.txt"
                                    batch_list_path = os.path.join(temp_batch_dir, batch_list_filename)
                                    job_id_stage3_5_batch = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())), f"stage3_5_batch_{batch_num}")
                                    
                                    if job_id_stage3_5_batch not in submitted_jobs_stage3_5:
                                        with open(batch_list_path, 'w') as f_list:
                                            for t_file in batch_files:
                                                full_path = os.path.join(tirp_objects_dir, t_file)
                                                f_list.write(f"{full_path}\n")

                                        if submit_stage3_5_batch_job(dataset_params, fold_num, abs_combo, mine_combo, batch_num, batch_list_path,
                                                                  abstraction_run_dir, mining_run_dir, log_dir, base_dir, event_symbol):
                                            submitted_jobs_stage3_5.add(job_id_stage3_5_batch)
                                            new_jobs_submitted_this_cycle = True

                                # Check completion (Scan individual TIRPs)
                                for tirp_filename in tirp_object_files:
                                    full_tirp_file_path = os.path.join(tirp_objects_dir, tirp_filename)
                                    sanitized_tirp_id = get_sanitized_tirp_id(full_tirp_file_path)
                                    if not sanitized_tirp_id: continue
                                    
                                    built_model_dir = os.path.join(built_models_base_dir, f'tirp_{sanitized_tirp_id}')
                                    stage3_5_done_path = os.path.join(built_model_dir, f'stage3_val_{sanitized_tirp_id}.done')
                                    
                                    if not os.path.exists(stage3_5_done_path):
                                         all_stage3_5_done_for_this_run = False
                                         break # Found a missing one, so stage 3.5 is not fully done

                        if not all_stage3_5_done_for_this_run: 
                            all_expected_final_jobs_done = False; combination_complete = False
                            continue # Wait for stage 3.5 validation to finish

                        # Stage 3 and Stage 3.5 are both fully built/validated here. Persist a
                        # sentinel so later cycles don't re-scan feature_matrix/ (which may be
                        # wiped by DELETE_FEATURE_MATRIX_ON_COMPLETION).
                        if not os.path.exists(stage3_sentinel_path):
                            try:
                                os.makedirs(mining_run_dir, exist_ok=True)
                                open(stage3_sentinel_path, 'a').close()
                            except Exception as e:
                                logger.warning(f"Could not write stage3 sentinel at {stage3_sentinel_path}: {e}")

                        # --- Stage 3.5 Done: Trigger Stage 4 (Predict Batches) if not already triggered ---
                        # stage4_key ties Stage 4 state to the exact (fold, abstraction, mining) config.
                        # Keying by job_id_stage2 ensures we dispatch entity-batch jobs exactly once
                        # per mining run, even if the monitoring loop revisits this combination in
                        # subsequent cycles.
                        stage4_key = job_id_stage2
                        if stage4_key not in stage4_trigger_status:
                            print(f"  All Stage 3.5 validations finished for {mining_run_dir}. Triggering Stage 4 predictions...")

                            print("Stage 3.5 completed for all TIRPs. Updating TIRP selection scores with validation metrics...")

                            current_tirp_selection_file = os.path.join(mining_run_dir, 'tirp_selection_scores.csv')
                            feature_matrix_dir = os.path.join(mining_run_dir, "feature_matrix")

                            try:
                                update_tirp_selection_scores_inplace(
                                    tirp_selection_scores_path=current_tirp_selection_file,
                                    feature_matrix_base_dir=feature_matrix_dir,
                                    top_k_list=config.MAX_TIRPS_FOR_SELECTION
                                )
                            except Exception as update_error:
                                print(f"CRITICAL ERROR updating TIRP selection scores: {update_error}")
                                # Depending on your logic, you might want to sys.exit(1) here if this is mandatory

                            print("Proceeding to Stage 4...")

                            entity_ids = get_test_entity_ids(abstraction_run_dir)
                            num_entities = len(entity_ids)
                            batch_size = config.ENTITY_BATCH_SIZE_FOR_PREDICTION
                            num_batches = math.ceil(num_entities / batch_size) if batch_size > 0 else 1
                            print(f"    Found {num_entities} test entities, creating {num_batches} prediction batches (size={batch_size}).")

                            stage4_trigger_status[stage4_key] = {'triggered': True, 'num_batches': num_batches, 'entity_ids': entity_ids}
                            submitted_jobs_stage4[stage4_key] = set() # Initialize set for submitted batch IDs

                            prediction_base_dir = os.path.join(mining_run_dir, "predictions")
                            temp_batch_dir = os.path.join(temp_dir, dataset_name, f'fold_{fold_num}', abs_param_string, mining_param_string)
                            os.makedirs(temp_batch_dir, exist_ok=True)

                            if num_entities > 0 and batch_size > 0:
                                for i in range(num_batches):
                                    start_idx = i * batch_size
                                    end_idx = start_idx + batch_size
                                    entity_batch = entity_ids[start_idx:end_idx]
                                    batch_id = i + 1

                                    # Create temp file for entity list
                                    entity_list_file = os.path.join(temp_batch_dir, f"batch_{batch_id:04d}_entities.txt")
                                    with open(entity_list_file, 'w') as f_ent:
                                        for ent_id in entity_batch: f_ent.write(f"{ent_id}\n")

                                    # Submit job for this batch
                                    if submit_stage4_job(dataset_params, fold_num, abs_combo, mine_combo, batch_id, num_batches, entity_list_file,
                                                        abstraction_run_dir, mining_run_dir, log_dir, base_dir):
                                        submitted_jobs_stage4[stage4_key].add(batch_id)
                                        new_jobs_submitted_this_cycle = True
                            elif num_entities == 0:
                                 print("    Warning: No test entities found. Skipping prediction stage.")
                                 # Mark as 'done' since there's nothing to predict
                                 stage4_trigger_status[stage4_key]['num_batches'] = 0


                        # --- Check Stage 4 Completion ---
                        all_stage4_done_for_this_run = True
                        status_info = stage4_trigger_status.get(stage4_key)
                        if not status_info or not status_info.get('triggered'):
                             all_stage4_done_for_this_run = False # Not triggered yet
                        else:
                            num_expected_batches = status_info['num_batches']
                            if num_expected_batches == 0:
                                 all_stage4_done_for_this_run = True # No batches needed, so considered done
                            else:
                                batch_status_dir = os.path.join(mining_run_dir, "batch_status_stage4")
                                for batch_num in range(1, num_expected_batches + 1):
                                    stage4_done_file = os.path.join(batch_status_dir, f'stage4_predict_batch_{batch_num:04d}.done')
                                    if not os.path.exists(stage4_done_file):
                                        all_stage4_done_for_this_run = False
                                        break # One missing is enough

                        if not all_stage4_done_for_this_run: 
                            all_expected_final_jobs_done = False; combination_complete = False
                            combo_status['stage4'] = 'waiting'
                            continue # Wait for stage 4 predictions
                        else:
                            combo_status['stage4'] = 'done'
                            cycle_progress['stage_stats'][4] += 1

                        # --- Stage 4 Done: Submit Stage 5 (Aggregation) ---
                        if model_combinations and model_combinations != [{}]:
                            all_stage5_done_for_this_run = True  # Track Stage 5 completion for this combination
                            
                            models_to_eval = ["FCPM", "CPML"] if getattr(config, 'BUILD_CPML', False) else ["FCPM"]
                            e_w_str = ",".join(map(str, dataset_params.get('e_w', [])))

                            # Handle "all" method specially - only run once per aggregation_method
                            if "all" in Tirp_selection_methods:
                                aggregation_methods = set()
                                for model_combo in model_combinations:
                                    if "aggregation_method" in model_combo:
                                        aggregation_methods.add(model_combo["aggregation_method"])
                                
                                for agg_method in aggregation_methods:
                                    model_combo_for_all = next((mc for mc in model_combinations if mc.get("aggregation_method") == agg_method), None)
                                    
                                    if model_combo_for_all:
                                        for m_nick in models_to_eval:
                                            job_id_stage5 = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())), tuple(sorted(model_combo_for_all.items())), "all", m_nick)
                                            model_param_str = create_param_string(model_combo_for_all)
                                            stg5_done_file = os.path.join(mining_run_dir, "results", f"{m_nick}_{model_param_str}_all{e_w_str}.done")
                                        
                                            if job_id_stage5 not in submitted_jobs_stage5 and not os.path.exists(stg5_done_file):
                                                if submit_stage5_job(dataset_params, fold_num, abs_combo, mine_combo, model_combo_for_all, "all", mining_run_dir, log_dir, base_dir, model_nickname=m_nick):
                                                    submitted_jobs_stage5.add(job_id_stage5); new_jobs_submitted_this_cycle = True
                                                all_stage5_done_for_this_run = False
                                            elif not os.path.exists(stg5_done_file):
                                                all_stage5_done_for_this_run = False
                            
                            # Handle other TIRP selection methods normally (run for each num_tirps_for_selection value)
                            for model_combo in model_combinations:
                                for tirp_selection_method in Tirp_selection_methods:
                                    if tirp_selection_method == "all": continue
                                    
                                    for m_nick in models_to_eval:
                                        job_id_stage5 = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())), tuple(sorted(model_combo.items())), tirp_selection_method, m_nick)
                                        model_param_str = create_param_string(model_combo)
                                        stg5_done_file = os.path.join(mining_run_dir, "results", f"{m_nick}_{model_param_str}_{tirp_selection_method}{e_w_str}.done")

                                        if job_id_stage5 not in submitted_jobs_stage5 and not os.path.exists(stg5_done_file):
                                            if submit_stage5_job(dataset_params, fold_num, abs_combo, mine_combo, model_combo, tirp_selection_method, mining_run_dir, log_dir, base_dir, model_nickname=m_nick):
                                                submitted_jobs_stage5.add(job_id_stage5); new_jobs_submitted_this_cycle = True
                                            all_stage5_done_for_this_run = False
                                        elif not os.path.exists(stg5_done_file):
                                            all_stage5_done_for_this_run = False

                            # Check if all Stage 5 jobs for this run are complete
                            if not all_stage5_done_for_this_run:
                                all_expected_final_jobs_done = False; combination_complete = False
                                combo_status['stage5'] = 'waiting'
                            else:
                                combo_status['stage5'] = 'done'
                                cycle_progress['stage_stats'][5] += 1

                        # If we reached here and stage 5 is the final stage, check its completion above
                        # If stage 4 was the final stage (no model_combinations), it's done
                        
                        # Log detailed progress for this combination (every 10 cycles or if new jobs submitted)
                        if monitoring_cycles % 10 == 0 or new_jobs_submitted_this_cycle:
                            log_progress_summary(logger, dataset_name, fold_num, abs_combo, mine_combo, combo_status)
                        
                        # --- Delete feature_matrix contents once combination is fully complete ---
                        if combination_complete and getattr(config, 'DELETE_FEATURE_MATRIX_ON_COMPLETION', False):
                            fm_key = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())))
                            if fm_key not in deleted_feature_matrices:
                                feature_matrix_dir = os.path.join(mining_run_dir, "feature_matrix")
                                if os.path.isdir(feature_matrix_dir):
                                    try:
                                        for item in os.listdir(feature_matrix_dir):
                                            item_path = os.path.join(feature_matrix_dir, item)
                                            if os.path.isdir(item_path):
                                                shutil.rmtree(item_path)
                                            else:
                                                os.remove(item_path)
                                        logger.info(f"Deleted feature_matrix contents: {feature_matrix_dir}")
                                    except Exception as e:
                                        logger.warning(f"Could not delete feature_matrix contents at {feature_matrix_dir}: {e}")
                                deleted_feature_matrices.add(fm_key)

                        # Mark combination as complete if all required stages are done
                        # Use a unique key to track if we've already counted this combination
                        combination_key = (fold_num, tuple(sorted(abs_combo.items())), tuple(sorted(mine_combo.items())))
                        if combination_complete and combination_key not in counted_combinations:
                            cycle_progress['completed_combinations'] += 1
                            counted_combinations.add(combination_key)

            # Update status file with current cycle progress
            update_status_file(base_dir, dataset_name, {
                'status': 'running',
                'stage': f'monitoring_cycle_{monitoring_cycles}',
                'cycle_progress': cycle_progress,
                'jobs_submitted_this_cycle': new_jobs_submitted_this_cycle,
                'consecutive_idle_cycles': consecutive_idle_cycles
            })
            
            # Log cycle summary
            cycle_duration = time.time() - cycle_start_time
            logger.info(f"[CYCLE {monitoring_cycles}] Completed in {cycle_duration:.1f}s")
            logger.info(f"  Progress: {cycle_progress['completed_combinations']}/{cycle_progress['total_combinations']} combinations complete")
            logger.info(f"  Stage completion counts: {cycle_progress['stage_stats']}")
            logger.info(f"  New jobs submitted: {new_jobs_submitted_this_cycle}")

            # --- Check overall completion for this dataset ---
            # We now check dynamically based on the `all_expected_final_jobs_done` flag
            if not all_expected_final_jobs_done:
                 # Three-tier adaptive sleep to balance responsiveness and cluster polling overhead:
                 #   Active  — new jobs were submitted this cycle; check back soon (ACTIVE_WAIT).
                 #   Normal  — jobs are running but nothing new to submit; use DEFAULT_WAIT.
                 #   Idle    — many consecutive cycles with no new submissions; slow down (IDLE_WAIT)
                 #             to avoid hammering the filesystem and SLURM scheduler unnecessarily.
                 wait_time = config.DEFAULT_WAIT_TIME_SECONDS; status_msg = f"Waiting..."
                 if new_jobs_submitted_this_cycle:
                     wait_time = config.ACTIVE_WAIT_TIME_SECONDS
                     status_msg += f" New jobs submitted. Next check in {wait_time}s."
                     consecutive_idle_cycles = 0
                 else:
                     status_msg += " No new jobs submitted."
                     consecutive_idle_cycles += 1

                 if consecutive_idle_cycles > 5: wait_time = config.IDLE_WAIT_TIME_SECONDS
                 if consecutive_idle_cycles >= max_idle_cycles:
                     logger.warning(f"Reached idle limit ({max_idle_cycles} cycles). Still waiting...")
                     
                 status_msg += f" Next check in {wait_time}s."
                 logger.info(status_msg)
                 
                 try: time.sleep(wait_time)
                 except KeyboardInterrupt: 
                     logger.info("Interrupted by user.")
                     active_monitoring = False; break # Break inner loop

            else: # All expected final jobs for *all combinations checked this cycle* seem done
                  logger.info(f"{'='*60}")
                  logger.info(f"ALL EXPECTED JOBS COMPLETE for dataset {dataset_name}")
                  logger.info(f"Completed monitoring after {monitoring_cycles} cycles")
                  logger.info(f"Final stage reached: {final_stage_to_check}")
                  logger.info(f"{'='*60}")
                  
                  update_status_file(base_dir, dataset_name, {
                      'status': 'completed_monitoring',
                      'stage': f'all_jobs_complete',
                      'total_cycles': monitoring_cycles
                  })
                  
                  active_monitoring = False # Exit monitor loop for this dataset

        # End of while active_monitoring loop
        logger.info(f"Starting results merging for dataset {dataset_name}...")
        results_dir = os.path.join(dataset_base_dir, 'results')
        merge_success = merge_results_files(results_dir)
        
        if merge_success:
            logger.info(f"Results merging completed successfully for dataset {dataset_name}")
        else:
            logger.warning(f"Results merging had issues for dataset {dataset_name}")

        logger.info(f"Generating aggregated results (mean AUC/AUPRC across folds) for dataset {dataset_name}...")
        agg_success = generate_aggregated_results(results_dir)
        if agg_success:
            logger.info(f"Aggregated results generated successfully for dataset {dataset_name}")
        else:
            logger.warning(f"Aggregated results generation had issues for dataset {dataset_name}")
            
        update_status_file(base_dir, dataset_name, {
            'status': 'completed',
            'stage': 'results_merged',
            'merge_success': merge_success
        })
        
        update_status_file(base_dir, dataset_name, {
            'status': 'completed',
            'stage': 'results_merged',
            'merge_success': merge_success
        })

        # --- Cleanup Fold Directories (Post-Experiment) ---
        # Trigger ONLY if:
        # 1. Config enabled
        # 2. Merging was successful
        # 3. monitoring ended because ALL jobs completed (not idle timeout or interruption)
        # We can check cycle_progress['completed_combinations'] == cycle_progress['total_combinations']
        
        # all_combinations_done = (cycle_progress['completed_combinations'] == cycle_progress['total_combinations'])
        
        if config.DELETE_FOLDS_ON_SUCCESS:
            if merge_success:
                logger.info(f"Experiment SUCCESS (All combinations done + Merge Success).")
                logger.info(f"Initiating cleanup of fold directories for dataset {dataset_name}...")
                
                for fold_num in range(1, n_folds + 1):
                    fold_dir_name = f"fold_{fold_num}"
                    fold_dir_path = os.path.join(dataset_base_dir, fold_dir_name)
                    
                    if os.path.exists(fold_dir_path):
                         submit_cleanup_job(dataset_name, fold_num, fold_dir_path, log_dir, base_dir)
                    else:
                        logger.warning(f"Cleanup: Fold dir not found: {fold_dir_path}")
            else:
                logger.warning(f"Skipping cleanup for {dataset_name}. Conditions not met:")
                logger.warning(f"  - Config Match: {config.DELETE_FOLDS_ON_SUCCESS}")
                logger.warning(f"  - Merge Success: {merge_success}")

        logger.info(f"{'='*60}")
        logger.info(f"FINISHED PROCESSING DATASET: {dataset_name}")
        logger.info(f"{'='*60}")

    # End of loop over all datasets
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL DATASETS PROCESSING COMPLETED")
    logger.info(f"{'='*80}")
    
    # Clean up temporary files for the datasets processed in THIS run only.
    # Removing the shared temp_files/ wholesale would delete batch list files
    # (tirp/entity lists read at job runtime) belonging to other orchestrator
    # processes running concurrently against the same base_dir.
    for dataset_params in datasets_list:
        ds_name = dataset_params.get('dataset_name')
        if not ds_name:
            continue
        ds_temp_dir = os.path.join(temp_dir, ds_name)
        if os.path.exists(ds_temp_dir):
            try:
                shutil.rmtree(ds_temp_dir)
                logger.info(f"Cleaned up temporary directory: {ds_temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory {ds_temp_dir}: {e}")

    end_time = time.time(); total_duration = end_time - start_time
    logger.info(f"=== EXPERIMENT COMPLETED ===")
    logger.info(f"Total execution time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final status update
    update_status_file(base_dir, "EXPERIMENT", {
        'status': 'completed',
        'total_duration_seconds': total_duration,
        'total_duration_hours': total_duration/3600,
        'datasets_processed': len(datasets_list)
    })




# --- Main Execution Block ---
if __name__ == "__main__":
    # Import configuration and parameters (essential step)
    try:
        import config
        import experiment_params
    except ImportError as e:
        print(f"FATAL ERROR: Could not import config.py or experiment_params.py: {e}")
        exit(1)
    except Exception as e:
         print(f"FATAL ERROR during import: {e}")
         exit(1)

    # --- Select which dataset(s) to run ---
    # By default every dataset in experiment_params.datasets_to_run is processed.
    # Pass --datasets "name1,name2" to restrict the run to specific dataset(s) so
    # each one can be launched as its own independent SLURM job (e.g. MIMIC-III and
    # MIMIC-IV as two separate sbatch submissions running in parallel).
    cli_parser = argparse.ArgumentParser(description="CPM Framework experiment orchestrator")
    cli_parser.add_argument("--datasets", type=str, default=None,
                            help="Comma-separated dataset_name(s) to run. Default: all in experiment_params.datasets_to_run")
    cli_args = cli_parser.parse_args()

    all_datasets = experiment_params.datasets_to_run
    if cli_args.datasets:
        requested_names = [n.strip() for n in cli_args.datasets.split(",") if n.strip()]
        name_to_dataset = {d.get("dataset_name"): d for d in all_datasets}
        missing = [n for n in requested_names if n not in name_to_dataset]
        if missing:
            print(f"FATAL ERROR: dataset(s) not found in experiment_params.datasets_to_run: {missing}")
            print(f"Available dataset names: {list(name_to_dataset.keys())}")
            exit(1)
        selected_datasets = [name_to_dataset[n] for n in requested_names]
    else:
        selected_datasets = all_datasets

    # Resolve base output directory to absolute path (maintains consistency)
    # Assume config.BASE_OUTPUT_DIR exists and is valid in config.py
    try:
        resolved_base_output_dir = os.path.abspath(config.BASE_OUTPUT_DIR)
        # Minimal directory creation (needed for logs/sbatch scripts) - assumes permissions are okay
        os.makedirs(resolved_base_output_dir, exist_ok=True)
    except AttributeError:
         print("FATAL ERROR: BASE_OUTPUT_DIR not defined in config.py.")
         exit(1)
    except OSError as e:
         print(f"FATAL ERROR creating base directory {resolved_base_output_dir}: {e}")
         exit(1)
    except Exception as e:
         print(f"FATAL ERROR resolving or creating base directory: {e}")
         exit(1)


    print(f"Starting experiment orchestration...")
    print(f" -> Output Directory: {resolved_base_output_dir}")
    print(f" -> Code Directory (from config): {config.CODE_DIR}")
    print(f" -> Number of Folds: {config.N_FOLDS}")
    print(f" -> Datasets selected for this run: {[d.get('dataset_name') for d in selected_datasets]}")
    # No pre-checks for sbatch, wrappers, or input files are performed here.

    try:
        # Run the main experiment function
        run_full_experiment(
            datasets_list=selected_datasets,
            n_folds=config.N_FOLDS,
            base_dir=resolved_base_output_dir # Pass the resolved absolute path
        )
    except KeyboardInterrupt:
        # Catch user interruption
        print("\n\nExperiment orchestration interrupted by user (Ctrl+C).")
    except Exception as e:
        # Catch any other unexpected error during the orchestration
        print(f"\nFATAL ERROR during experiment execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        # This will always run, even after errors or interruptions
        print("\nExperiment orchestration script finished.")