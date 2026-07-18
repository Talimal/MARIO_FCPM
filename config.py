import os

# --- Directory Settings ---
# ** USER ACTION REQUIRED: Update this path to the directory containing wrapper scripts **
CODE_DIR = "/mnt/new_groups/robertmo_group/Tali/CPM_Framework_2/"# Example: '/home/user/my_experiment/code' or '.' if in same directory
# Base directory for all experiment output
# BASE_OUTPUT_DIR =  "/mnt/new_groups/robertmo_group/Niv/Falls_PAA3_val_CPML16042026/"
BASE_OUTPUT_DIR =  "/mnt/new_groups/robertmo_group/Tali/FCPM_expirment"

# --- Core Experiment Settings ---
N_FOLDS = 1  # Number of cross-validation folds
SEED = 2026  # Random seed for reproducibility

# --- Forecasting (MARIO) split settings (Stage 0) ---
# Fraction of entities held out entirely from training (new-entity test regime).
# Set to 0.0 for pure within-entity forecasting.
HOLDOUT_ENTITY_FRACTION = 0.2
# Per-entity chronological cut point, as a fraction of the entity's time span.
CHRONO_SPLIT_RATIO = 0.8
# Default forecast horizon / embargo size, used when a dataset does not set
# its own 'horizon'. HORIZON is conceptually per-dataset.
DEFAULT_HORIZON = 1

# --- MARIO TIRP selection (Stage 2) ---
# All mined TIRPs are kept, then filtered to those whose vertical support
# (as a fraction of training entities) falls in [TIRP_VS_MIN, TIRP_VS_MAX].
# Defaults are a no-op (keep everything above the mining mvs floor); tighten
# the range to drop rare (low VS) or ubiquitous/uninformative (high VS) TIRPs.
TIRP_VS_MIN = 0.0
TIRP_VS_MAX = 1.0

# --- MARIO cross-TIRP aggregation (Stage 5) ---
# How the per-TIRP forecast distributions are combined into one forecast per
# (entity, t): currently only unweighted 'average'.
STAGE5_AGGREGATION_METHOD = "average"
# Grace window C (TimeStamp units): a TIRP counts as active at t if its most
# recent forecast falls in [t - C, t]. 0 = exact (the prefix must span t).
STAGE5_CONTEXT_WINDOW = 0
# Per-entity warm-up before new_entity (holdout) rows are scored.
STAGE5_WARMUP = 0
DEFAULT_EVENT_SYMBOL = 999  # Default event symbol if not specified per dataset
BUILD_CPML = True  # Flag to build and evaluate the CPML model
RUN_STAGE3_5_VALIDATION = False # Make Stage 3.5 optional

# Flag to control Train Validation behavior (Stage 3.5):
# True  = Conditional Validation: Evaluate only on entities where the TIRP actually appears (focuses on precision and conditional predictive power).
# False = Global Validation: Evaluate on all entities in the training set (penalizes rare TIRPs).
SUPPORTING_ENTITIES_ONLY = True

# --- Stage-Specific Parameters ---
# Max number of TIRPs to consider in Stage 2 Tirp_selection (e.g., after sorting by some criteria)
# Set to -1 or None for no limit by default
MAX_TIRPS_FOR_SELECTION = [20] #
SKIP_SAME_VARIABLE = False

# --- Stage 1: Knowledge-Based and Gradient Configuration ---
# These are static configuration values for KB and gradient methods
# They are NOT hyperparameters to tune - they are fixed based on clinical/domain knowledge
GRADIENT_WINDOW_SIZE = 5  # - clinically determined optimal window
KB_STATES_PATH = ""
GRADIENT_CUTOFFS_PATH = ""

# Ratio used in Stage 3 to calculate window_size for split_data based on max_gap from mining stage
WINDOW_SIZE_RATIO = 1.2 # 

# Number of entities per prediction job batch in Stage 4
ENTITY_BATCH_SIZE_FOR_PREDICTION = 30 # Adjust as needed
# Number of TIRPs per model building job batch in Stage 3
TIRP_BATCH_SIZE_STAGE3 = 30 # Adjust as needed
# Number of TIRPs per validation job batch in Stage 3.5
TIRP_BATCH_SIZE_STAGE3_5 = 10 
EPSILON_FCPM = 1


# --- Supervised Abstraction Methods ---
# Defines which methods support split_event_class and event_window
SUPERVISED_ABSTRACTION_METHODS = [
    'tid3',
    'tid3_c0longer',
    'tid3_c1longer',
    'tid3_logrank',
    'tid3_logrank_c0longer',
    'tid3_logrank_c1longer',
    'td4c_cosine',
    'td4c',
    'tid3_selftrans',
    'tid3_selftrans_c1longer',
    'tid3_selftrans_c0longer',
    'tid3_mv',
    'tid3_mv_c1longer',
    'tid3_mv_c0longer',
    'td4c_kullback-leibler',
    'td4c_cosine',
    'td4c_diffsum',
    'td4c_diffmax'
]


# --- Job Submission Control ---
MAX_CONCURRENT_JOBS = 600  # <<-- DEPRECATED: Use MAX_MEMORY_GB instead
MAX_MEMORY_GB = 4000  # <<-- NEW: Max total memory (GB) allowed for running+pending jobs
JOB_SUBMISSION_WAIT_INTERVAL_SECONDS = 30 # <<-- NEW: How long to wait (seconds) when memory limit is reached
USE_NORMALIZATION_RATIO = True # Use normalization ratio in Stage 3


DELETE_FOLDS_ON_SUCCESS = False # Delete fold directories after successful completion
DELETE_FEATURE_MATRIX_ON_COMPLETION = False # Delete feature_matrix folder contents after a combination (fold+abs+mine) finishes all stages


# --- Wrapper Script Basenames ---
WRAPPER_SCRIPT_PATHS = {
    0: "run_stage0_split_folds.py",
    1: "run_stage1_abstraction.py",
    2: "run_stage2_mining.py",
    3: "run_stage3_build_model.py",        # Builds model only
    3.5: "run_stage3_5_validation.py",     # Validates FCPM model
    4: "run_stage4_predict_entities.py",   # MARIO Stage 4: forecast on TEST set, batched by TIRP
    5: "run_stage5_aggregation_eval.py",    # Original stage 4 (aggregation) renamed
    "cleanup": "run_cleanup_fold.py"        # Cleanup script
}

# --- SLURM Resource Allocation per Stage ---
SBATCH_RESOURCES = {
    0: {'cpus': 1, 'mem': 2, 'time_limit': "00:30:00"}, # Split Folds
    1: {'cpus': 1, 'mem': 4, 'time_limit': "6-23:00:00"}, # Abstraction
    2: {'cpus': 4, 'mem': 10, 'time_limit': "6-12:00:00"}, # Mining
    3: {'cpus': 1, 'mem': 5, 'time_limit': "6-23:00:00"}, # Build Model
    3.5: {'cpus': 1, 'mem': 10, 'time_limit': "6-23:00:00"}, # Validation
    4: {'cpus': 1, 'mem': 5, 'time_limit': "6-23:30:00"}, # Predict Batch (adjust time/mem as needed)
    5: {'cpus': 1, 'mem': 5, 'time_limit': "6-23:45:00"}, # Aggregation/Eval
    "cleanup": {'cpus': 4, 'mem': 8, 'time_limit': "6-23:45:00"}    # Cleanup
}

# --- Monitoring Settings ---
MAX_WAIT_STAGE0_SECONDS = 30 * 60  # Max wait time for Stage 0 completion (in seconds)
MAX_MONITORING_CYCLES = 48 * 60    # Max monitoring loops (~48 hours with 1 min default sleep)
MAX_IDLE_CYCLES = 30               # Max consecutive cycles with no new jobs submitted before warning
DEFAULT_WAIT_TIME_SECONDS = 60     # Sleep time between monitoring checks when jobs are running
ACTIVE_WAIT_TIME_SECONDS = 20      # Sleep time after new jobs were submitted
IDLE_WAIT_TIME_SECONDS = 120       # Increased sleep time if idle for > 5 cycles

# --- Optional Conda Environment Activation ---
CONDA_ENV_NAME = "FCPM_env"
CONDA_LOAD_COMMAND = "module load anaconda"

# --- SLURM Script Template ---
SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}/{job_name}.%j.out
#SBATCH --error={log_path}/{job_name}.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}
# TEMP: cs-cpu-10 kills jobs at startup with RaisedSignal:53 (2026-06-17). Remove once node is repaired.
#SBATCH --exclude=cs-cpu-10

echo "Starting job {job_name} for stage {stage_num}"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"            # This should now show the --chdir path
echo "Python executable: $(which python)"
echo "Script path: {script_path}"
echo "Arguments: {arguments}"
echo "Done file path: {done_file_path}"

# --- Environment Activation ---
{conda_activation_lines}
# --- End Environment Activation ---

# Run the stage-specific python script
echo "Executing: python \\"{script_path}\\" {arguments}"
python "{script_path}" {arguments}

# Check command success and create done file
if [ $? -eq 0 ]; then
  echo "Job {job_name} completed successfully, creating done file: {done_file_path}"
  mkdir -p "$(dirname "{done_file_path}")"
  touch "{done_file_path}"
else
  echo "ERROR: Job {job_name} failed! Check error log: {log_path}/{job_name}.%j.err"
  exit 1
fi

echo "Job {job_name} finished."
"""

