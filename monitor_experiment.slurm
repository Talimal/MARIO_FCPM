#!/bin/bash
#SBATCH --job-name=exp_monitor
#SBATCH --output=monitor_logs/monitor_%j.out
#SBATCH --error=monitor_logs/monitor_%j.err
#SBATCH --time 6-23:59:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --partition=main
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eldarzos@post.bgu.ac.il

# Configuration - UPDATE THESE PATHS
# Must match config.BASE_OUTPUT_DIR in this project.
BASE_DIR="/mnt/new_groups/robertmo_group/Eldar/FCPM_CMPL_TID3_ttest_logrank_3"
LOGS_DIR="/mnt/new_groups/robertmo_group/Eldar/FCPM_CMPL_TID3_ttest_logrank_3/logs/"
EMAIL="eldarzos@post.bgu.ac.il"
INTERVAL=2  # Email every 6 hours
FAILURE_LOOKBACK=168  # Look back 168 hours (7 days) for failures

# Activate conda environment (if needed)
module load anaconda
source activate FCPM_env

echo "Starting experiment monitor..."
echo "Base directory: $BASE_DIR"
echo "Logs directory: $LOGS_DIR"
echo "Email: $EMAIL"
echo "Interval: $INTERVAL hours"
echo "Started at: $(date)"

# Run the monitoring script (this project's copy, not an external one)
python /mnt/new_groups/robertmo_group/Eldar/projects/CPM_Framework_2/experiment_monitor.py \
    --base_dir "$BASE_DIR" \
    --logs_dir "$LOGS_DIR" \
    --email "$EMAIL" \
    --interval "$INTERVAL" \
    --failure_lookback "$FAILURE_LOOKBACK" \
    --smtp_server "smtp.bgu.ac.il" \
    --smtp_port 25 \
    --sender_email "experiment_monitor@ise-pheno-12"

echo "Monitor finished at: $(date)" 