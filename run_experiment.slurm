#!/bin/bash
#SBATCH --job-name=mario_orchestrator
#SBATCH --output=/mnt/new_groups/robertmo_group/Tali/CPM_Framework_2/orchestrator_logs/orchestrator_%j.out
#SBATCH --error=/mnt/new_groups/robertmo_group/Tali/CPM_Framework_2/orchestrator_logs/orchestrator_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6-23:59:00
#SBATCH --partition=main
# TEMP: cs-cpu-10 kills jobs at startup (see config.SBATCH_SCRIPT_TEMPLATE). Remove once repaired.
#SBATCH --exclude=cs-cpu-10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=talimal@post.bgu.ac.il

# =============================================================================
# MARIO forecasting pipeline launcher.
#
# This job runs the long-lived orchestrator (run_experiment.py), which itself
# submits and monitors the per-stage SLURM jobs (Stage 0 -> Stage 5) exactly as
# described in FCPM_MARIO_README.md and the explore_stageX.ipynb notebooks.
#
# All outputs (folds, abstraction KL, mined TIRPs, per-TIRP models, forecasts,
# metrics, PNG visualizations and every intermediate dataframe) are written under
# config.BASE_OUTPUT_DIR, which now points INSIDE this project:
#     <project>/experiment_output/
#
# Usage:
#     sbatch run_experiment.slurm                    # every experiment in experiment_params
#     sbatch run_experiment.slurm --datasets diabetes_h5   # one horizon experiment
#     sbatch run_experiment.slurm --datasets diabetes      # all horizons of a dataset
#
# Multi-horizon: a list-valued 'horizon' in experiment_params.py expands to one
# experiment per horizon named '<dataset>_h<horizon>' (e.g. diabetes_h1, diabetes_h5).
# Launch them in parallel as separate orchestrator jobs, one per horizon:
#     for H in 1 5 10; do sbatch run_experiment.slurm --datasets diabetes_h$H; done
#
# The trailing "$@" forwards any extra args (e.g. --datasets) to run_experiment.py.
# =============================================================================

PROJECT_DIR="/mnt/new_groups/robertmo_group/Tali/CPM_Framework_2"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

echo "=== MARIO orchestrator starting ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Host:              $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time:        $(date)"
echo "Extra args:        $*"

# --- Environment activation (matches config.CONDA_* so sub-jobs use the same env) ---
module load anaconda
source activate FCPM_env
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate Conda environment 'FCPM_env'."
  exit 1
fi
echo "Python executable: $(which python)"

# --- Run the orchestrator (submits + monitors all per-stage jobs) ---
python "$PROJECT_DIR/run_experiment.py" "$@"
status=$?

echo "=== MARIO orchestrator finished with exit code $status at $(date) ==="
exit $status
