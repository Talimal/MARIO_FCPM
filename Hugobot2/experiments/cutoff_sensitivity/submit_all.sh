#!/bin/bash
# Submit one job per dataset
cd /sise/robertmo-group/Eldar/projects/CPM_Framework/Hugobot2/experiments/cutoff_sensitivity

for ds in falls_small diabetes icu ahe_small; do
    echo "Submitting $ds ..."
    sbatch ex.example "$ds"
done
