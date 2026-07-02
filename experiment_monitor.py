#!/usr/bin/env python3
"""
Experiment Progress Monitor with Email Notifications
Monitors experiment progress and sends status emails with failed job analysis
"""

import os
import glob
import time
import smtplib
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import concurrent.futures
from pathlib import Path
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import sys
import json
import subprocess
import hashlib
import shutil

# Try to import config - make it optional for standalone operation
try:
    import config
    BASE_DIR = getattr(config, 'BASE_OUTPUT_DIR', None)
    CODE_DIR = getattr(config, 'CODE_DIR', None)
except ImportError:
    BASE_DIR = None
    CODE_DIR = None

class OOMResubmissionManager:
    """Manages OOM failure detection and resubmission with increased memory"""
    
    def __init__(self, logs_dir, base_dir):
        self.logs_dir = Path(logs_dir)
        self.base_dir = Path(base_dir)
        self.resubmit_dir = self.logs_dir.parent / "OOM_resubmit_scripts"
        self.resubmit_dir.mkdir(exist_ok=True)
        
        # Track resubmissions to avoid duplicates
        self.resubmission_tracker_file = self.resubmit_dir / "resubmission_tracker.json"
        self.resubmission_tracker = self.load_resubmission_tracker()
        
        # File-based persistence for resubmitted job IDs
        self.resubmitted_job_ids_file = self.resubmit_dir / "resubmitted_job_ids.json"
        self.resubmitted_job_ids = set()
        
        # Memory increase per resubmission
        self.memory_increase_gb = 20
        
    def load_resubmission_tracker(self):
        """Load resubmission tracking data"""
        if self.resubmission_tracker_file.exists():
            try:
                with open(self.resubmission_tracker_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def load_resubmitted_job_ids(self):
        """Load previously resubmitted job IDs from persistent file"""
        if self.resubmitted_job_ids_file.exists():
            try:
                with open(self.resubmitted_job_ids_file, 'r') as f:
                    job_ids_list = json.load(f)
                    self.resubmitted_job_ids = set(job_ids_list)
                    if self.resubmitted_job_ids:
                        print(f"Loaded {len(self.resubmitted_job_ids)} previously resubmitted job IDs from file")
                    return
            except Exception as e:
                print(f"Warning: Could not load resubmitted job IDs from file: {e}")
        
        # Fallback: Load from tracker for backward compatibility
        for tracker_info in self.resubmission_tracker.values():
            original_job_id = tracker_info.get('original_slurm_job_id')
            if original_job_id:
                self.resubmitted_job_ids.add(original_job_id)
                
        if self.resubmitted_job_ids:
            print(f"Loaded {len(self.resubmitted_job_ids)} previously resubmitted job IDs from tracker (fallback)")
            # Save to file for future use
            self.save_resubmitted_job_ids()
    
    def save_resubmitted_job_ids(self):
        """Save resubmitted job IDs to persistent file"""
        try:
            with open(self.resubmitted_job_ids_file, 'w') as f:
                json.dump(list(self.resubmitted_job_ids), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save resubmitted job IDs to file: {e}")
    
    def load_resubmitted_job_ids_for_monitoring_cycle(self):
        """Load resubmitted job IDs at the start of each monitoring cycle"""
        self.load_resubmitted_job_ids()
    
    def save_resubmission_tracker(self):
        """Save resubmission tracking data"""
        try:
            with open(self.resubmission_tracker_file, 'w') as f:
                json.dump(self.resubmission_tracker, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save resubmission tracker: {e}")
    
    def parse_out_file(self, out_file_path):
        """Parse .out file to extract job parameters including SLURM Job ID"""
        try:
            with open(out_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            job_info = {}
            
            # Extract job name
            job_name_match = re.search(r'Starting job (\S+) for stage (\d+)', content)
            if job_name_match:
                job_info['job_name'] = job_name_match.group(1)
                job_info['stage'] = int(job_name_match.group(2))
            
            # Extract SLURM Job ID (this is the key improvement!)
            job_id_match = re.search(r'Job ID: (\d+)', content)
            if job_id_match:
                job_info['slurm_job_id'] = job_id_match.group(1)
            
            # Extract script path
            script_match = re.search(r'Script path: (.+)', content)
            if script_match:
                job_info['script_path'] = script_match.group(1).strip()
            
            # Extract arguments
            args_match = re.search(r'Arguments: (.+)', content)
            if args_match:
                job_info['arguments'] = args_match.group(1).strip()
            
            # Extract done file path
            done_file_match = re.search(r'Done file path: (.+)', content)
            if done_file_match:
                job_info['done_file_path'] = done_file_match.group(1).strip()
            
            # Extract working directory
            work_dir_match = re.search(r'Working directory: (.+)', content)
            if work_dir_match:
                job_info['work_dir'] = work_dir_match.group(1).strip()
            
            return job_info
            
        except Exception as e:
            print(f"Error parsing .out file {out_file_path}: {e}")
            return None
    
    def get_original_memory(self, stage):
        """Get original memory allocation for a stage"""
        try:
            # Import config to get SBATCH_RESOURCES
            if 'config' in sys.modules:
                config_module = sys.modules['config']
                if hasattr(config_module, 'SBATCH_RESOURCES'):
                    return config_module.SBATCH_RESOURCES.get(stage, {}).get('mem', 4)
            
            # Default memory allocations if config not available
            default_memory = {
                0: 10, 1: 10, 2: 120, 3: 71, 4: 50, 5: 70
            }
            return default_memory.get(stage, 4)
        except Exception:
            return 4
    
    def create_resubmission_script(self, job_info, original_err_file):
        """Create a new SBATCH script with increased memory"""
        if not job_info or 'stage' not in job_info:
            return None
        
        stage = job_info['stage']
        job_name = job_info.get('job_name', 'unknown_job')
        
        # Calculate new memory allocation
        original_memory = self.get_original_memory(stage)
        
        # Check how many times this job has been resubmitted
        job_key = f"{job_name}_{stage}"
        resubmission_count = self.resubmission_tracker.get(job_key, {}).get('count', 0)
        new_memory = original_memory + (self.memory_increase_gb * (resubmission_count + 1))
        
        # Generate new job name with resubmission suffix
        new_job_name = f"{job_name}_oom_resubmit_{resubmission_count + 1}"
        
        # Create SBATCH script content
        script_content = self.generate_sbatch_script(
            job_info, new_job_name, new_memory, stage
        )
        
        if not script_content:
            return None
        
        # Save script to resubmit directory
        script_filename = f"{new_job_name}.slurm"
        script_path = self.resubmit_dir / script_filename
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Update resubmission tracker
            self.resubmission_tracker[job_key] = {
                'count': resubmission_count + 1,
                'original_memory': original_memory,
                'new_memory': new_memory,
                'script_path': str(script_path),
                'last_resubmit': datetime.now().isoformat(),
                'original_err_file': str(original_err_file),
                'original_slurm_job_id': job_info.get('slurm_job_id'),  # Store the original SLURM Job ID
                'submitted': False,  # Will be updated when actually submitted
                'new_job_id': None,  # Will be updated when submitted
                'submission_time': None  # Will be updated when submitted
            }
            self.save_resubmission_tracker()
            
            return script_path
            
        except Exception as e:
            print(f"Error creating resubmission script: {e}")
            return None
    
    def generate_sbatch_script(self, job_info, new_job_name, new_memory, stage):
        """Generate SBATCH script content with increased memory"""
        try:
            # Get original resource settings
            original_resources = self.get_original_resources(stage)
            
            # Get conda activation lines
            conda_lines = self.get_conda_activation_lines()
            
            # Use the template from config.py directly
            script_template = self.get_config_template()
            
            # Modify the template to add OOM-specific information
            script_template = script_template.replace(
                'echo "Starting job {job_name} for stage {stage_num}"',
                'echo "Starting job {job_name} for stage {stage_num} (OOM RESUBMISSION)"\n'
                'echo "Original memory: {original_memory}GB, New memory: {mem}GB"'
            )
            
            script_content = script_template.format(
                job_name=new_job_name,
                log_path=self.logs_dir,
                cpus=original_resources['cpus'],
                mem=new_memory,
                time_limit=original_resources['time_limit'],
                stage_num=stage,
                script_path=job_info.get('script_path', ''),
                arguments=job_info.get('arguments', ''),
                done_file_path=job_info.get('done_file_path', ''),
                original_memory=self.get_original_memory(stage),
                conda_activation_lines=conda_lines
            )
            
            return script_content
            
        except Exception as e:
            print(f"Error generating SBATCH script: {e}")
            return None
    
    def get_config_template(self):
        """Get the SBATCH script template from config.py"""
        try:
            if 'config' in sys.modules:
                config_module = sys.modules['config']
                if hasattr(config_module, 'SBATCH_SCRIPT_TEMPLATE'):
                    return config_module.SBATCH_SCRIPT_TEMPLATE
            
            # Fallback template if config not available (should match config.py exactly)
            return """#!/bin/bash
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
        except Exception:
            # Ultimate fallback
            return """#!/bin/bash
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
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Script path: {script_path}"
echo "Arguments: {arguments}"
echo "Done file path: {done_file_path}"

{conda_activation_lines}

python "{script_path}" {arguments}

if [ $? -eq 0 ]; then
  mkdir -p "$(dirname "{done_file_path}")"
  touch "{done_file_path}"
else
  exit 1
fi
"""
    
    def get_original_resources(self, stage):
        """Get original resource settings for a stage"""
        config_module = sys.modules['config']
        if hasattr(config_module, 'SBATCH_RESOURCES'):
            return config_module.SBATCH_RESOURCES.get(stage, {
                'cpus': 1, 'mem': 4, 'time_limit': '01:00:00'
            })

    
    def get_conda_activation_lines(self):
        """Get conda activation lines"""
        try:
            if 'config' in sys.modules:
                config_module = sys.modules['config']
                conda_env = getattr(config_module, 'CONDA_ENV_NAME', None)
                conda_load = getattr(config_module, 'CONDA_LOAD_COMMAND', None)
                
                if conda_env:
                    lines = ""
                    if conda_load:
                        lines += f"{conda_load}\n"
                    lines += f"source activate {conda_env}\n"
                    lines += f"""
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate Conda environment '{conda_env}'."
  exit 1
fi
"""
                    return lines
            
            return "# Conda activation not configured."
        except Exception:
            return "# Conda activation not configured."
    
    def should_resubmit(self, job_name, stage, slurm_job_id):
        """Check if job should be resubmitted based on SLURM Job ID tracking"""
        # Check if this specific SLURM Job ID was already resubmitted
        if slurm_job_id in self.resubmitted_job_ids:
            return False  # This exact job ID was already resubmitted
        
        # Allow resubmission if this specific job ID hasn't been processed
        return True
    
    def submit_resubmission_script(self, script_path):
        """Submit a resubmission script using sbatch"""
        try:
            result = subprocess.run(['sbatch', str(script_path)], 
                                  capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            return True, job_id
        except subprocess.CalledProcessError as e:
            return False, f"sbatch failed: {e.stderr.strip()}"
        except Exception as e:
            return False, f"submission error: {e}"
    
    def process_oom_failures(self, oom_failures):
        """Process OOM failures, create resubmission scripts, and submit them automatically"""
        # Load the latest resubmitted job IDs at the start of each monitoring cycle
        self.load_resubmitted_job_ids_for_monitoring_cycle()
        
        resubmitted_jobs = []
        
        for failure_info in oom_failures:
            err_file = failure_info['err_file']
            
            # Find corresponding .out file
            out_file = err_file.replace('.err', '.out')
            if not os.path.exists(out_file):
                continue
            
            # Parse .out file to get job parameters
            job_info = self.parse_out_file(out_file)
            if not job_info:
                continue
            
            job_name = job_info.get('job_name', 'unknown')
            stage = job_info.get('stage', 0)
            slurm_job_id = job_info.get('slurm_job_id')
            
            # Skip if no SLURM Job ID found
            if not slurm_job_id:
                print(f"Warning: No SLURM Job ID found in {out_file}, skipping resubmission")
                continue
            
            # Check if we should resubmit based on SLURM Job ID
            if not self.should_resubmit(job_name, stage, slurm_job_id):
                print(f"Skipping resubmission for Job ID {slurm_job_id} (already processed)")
                continue
            
            # Create resubmission script
            script_path = self.create_resubmission_script(job_info, err_file)
            if script_path:
                job_key = f"{job_name}_{stage}"
                new_memory = self.resubmission_tracker[job_key]['new_memory']
                
                # Automatically submit the resubmission script
                success, result = self.submit_resubmission_script(script_path)
                
                job_info_dict = {
                    'job_name': job_name,
                    'stage': stage,
                    'script_path': script_path,
                    'new_memory': new_memory,
                    'submitted': success
                }
                
                if success:
                    job_info_dict['new_job_id'] = result
                    print(f"✓ Automatically submitted OOM resubmission: {job_name} (Original Job ID: {slurm_job_id}) -> New Job ID: {result}")
                    
                    # Update tracker with submission info
                    self.resubmission_tracker[job_key]['submitted'] = True
                    self.resubmission_tracker[job_key]['new_job_id'] = result
                    self.resubmission_tracker[job_key]['submission_time'] = datetime.now().isoformat()
                    self.save_resubmission_tracker()
                    
                    # Add the original SLURM Job ID to the resubmitted set to prevent future duplicates
                    self.add_resubmitted_job_id(slurm_job_id)
                else:
                    job_info_dict['error'] = result
                    print(f"✗ Failed to submit OOM resubmission: {job_name} (Original Job ID: {slurm_job_id}) -> {result}")
                
                # Add SLURM Job ID info to the job info for email reporting
                job_info_dict['original_slurm_job_id'] = slurm_job_id
                resubmitted_jobs.append(job_info_dict)
        
        return resubmitted_jobs

    def add_resubmitted_job_id(self, job_id):
        """Add a job ID to the resubmitted set with logging and immediate persistence"""
        self.resubmitted_job_ids.add(job_id)
        self.save_resubmitted_job_ids()  # Save immediately to file
        print(f"Added Job ID {job_id} to resubmitted tracking set (saved to file)")


class FastExperimentMonitor:
    """Extracted and adapted from EXP_PROGRESS.ipynb"""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
       
    def scan_directory_fast(self, directory_path):
        """Fast scan using pathlib and os.scandir for maximum performance"""
        directory_path = Path(directory_path)
        if not directory_path.exists():
            return {}
           
        results = {
            'done_files': [],
            'subdirs': [],
            'pkl_files': [],
            'csv_files': []
        }
       
        try:
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        name = entry.name
                        if name.endswith('.done'):
                            results['done_files'].append(name)
                        elif name.endswith('.pkl'):
                            results['pkl_files'].append(name)
                        elif name.endswith('.csv'):
                            results['csv_files'].append(name)
                    elif entry.is_dir():
                        results['subdirs'].append(entry.name)
        except (OSError, PermissionError):
            pass
           
        return results
   
    def count_files_in_directory(self, directory_path, pattern):
        """Ultra-fast file counting using glob"""
        try:
            return len(list(Path(directory_path).glob(pattern)))
        except (OSError, PermissionError):
            return 0
   
    def get_expected_batch_count(self, mine_dir):
        """Calculate expected number of batches based on test entities and batch size"""
        mine_dir = Path(mine_dir)

        # Navigate to abstraction directory to find Test folder
        abs_dir = mine_dir.parent
        test_dir = abs_dir / "Test"
        entity_class_file = test_dir / "entity-class-relations.csv"

        if not entity_class_file.exists():
            return 0

        try:
            # Count UNIQUE test entities (matches run_experiment.get_test_entity_ids,
            # which uses df['EntityID'].unique()). Counting rows would overcount if the
            # CSV has multiple rows per entity.
            df = pd.read_csv(entity_class_file, usecols=['EntityID'])
            num_entities = df['EntityID'].nunique()

            # Use the same batch size as run_experiment.py / config.py (no hardcoding).
            batch_size = 100  # Fallback if config is unavailable
            if 'config' in sys.modules:
                batch_size = getattr(sys.modules['config'], 'ENTITY_BATCH_SIZE_FOR_PREDICTION', batch_size)

            if batch_size > 0:
                import math
                expected_batches = math.ceil(num_entities / batch_size)
            else:
                expected_batches = 1

            return expected_batches
        except Exception:
            return 0

    def scan_mining_combination_fast(self, mine_dir):
        """Optimized scanning of mining combination directory"""
        mine_dir = Path(mine_dir)
       
        progress = {
            'stage2_done': (mine_dir / 'stage2_mining.done').exists(),
            'tirps_count': 0,
            'stage3_models_built': 0,
            'stage4_batches_done': 0,
            'stage4_expected_batches': 0,
            'stage4_complete': False,
            'stage5_results': 0
        }
       
        # Count TIRPs efficiently
        tirps_dir = mine_dir / "tirps"
        if tirps_dir.exists():
            progress['tirps_count'] = self.count_files_in_directory(tirps_dir, "*.pkl")

            # Count Stage 3 models built.
            # run_experiment writes a persistent sentinel (stage3_all_built.done) in the
            # mining dir once Stage 3 (and 3.5) finish, then may wipe feature_matrix/ when
            # DELETE_FEATURE_MATRIX_ON_COMPLETION is set. If the sentinel exists, Stage 3
            # is complete regardless of whether the per-TIRP done files still exist.
            if (mine_dir / "stage3_all_built.done").exists():
                progress['stage3_models_built'] = progress['tirps_count']
            else:
                feature_matrix_dir = mine_dir / "feature_matrix"
                if feature_matrix_dir.exists():
                    progress['stage3_models_built'] = self.count_files_in_directory(feature_matrix_dir, "*/stage3_build_*.done")

        # Calculate expected batches and check Stage 4 completion
        progress['stage4_expected_batches'] = self.get_expected_batch_count(mine_dir)

        # Stage 4 done files live in batch_status_stage4/ (the predictions/ dir holds the
        # prediction *data*, not the .done sentinels). See run_experiment.submit_stage4_job.
        batch_status_dir = mine_dir / "batch_status_stage4"
        if batch_status_dir.exists():
            progress['stage4_batches_done'] = self.count_files_in_directory(batch_status_dir, "stage4_predict_batch_*.done")

            # Check if ALL expected batches are complete
            expected = progress['stage4_expected_batches']
            if expected == 0:
                progress['stage4_complete'] = True  # No batches needed
            else:
                # Check if all batch files from 1 to expected exist
                all_batches_exist = True
                for batch_num in range(1, expected + 1):
                    batch_file = batch_status_dir / f"stage4_predict_batch_{batch_num:04d}.done"
                    if not batch_file.exists():
                        all_batches_exist = False
                        break
                progress['stage4_complete'] = all_batches_exist
        else:
            progress['stage4_complete'] = False
       
        # Count Stage 5 results
        results_dir = mine_dir / "results"
        if results_dir.exists():
            progress['stage5_results'] = self.count_files_in_directory(results_dir, "*.done")
       
        return progress
   
    def scan_dataset_parallel(self, dataset_dir):
        """Scan dataset using parallel processing for maximum speed"""
        dataset_dir = Path(dataset_dir)
        dataset_name = dataset_dir.name
       
        # Quick scan for stage0 and fold directories
        scan_result = self.scan_directory_fast(dataset_dir)
        stage0_done = 'stage0_split.done' in scan_result['done_files']
       
        fold_dirs = [d for d in scan_result['subdirs'] if d.startswith('fold_')]
       
        if not fold_dirs:
            return {
                'dataset_name': dataset_name,
                'stage0_done': stage0_done,
                'total_folds': 0,
                'total_combinations': 0,
                'stage_counts': defaultdict(int),
                'tirps_stats': {'total': 0, 'models_built': 0},
                'batch_stats': {'total': 0, 'expected': 0},
                'results_stats': {'total': 0}
            }
       
        # Collect all mining directories for parallel processing
        mining_dirs = []
        total_combinations = 0
       
        for fold_name in fold_dirs:
            fold_dir = dataset_dir / fold_name
            fold_scan = self.scan_directory_fast(fold_dir)
           
            for abs_dir_name in fold_scan['subdirs']:
                if abs_dir_name.startswith('abs_'):
                    abs_dir = fold_dir / abs_dir_name
                    abs_scan = self.scan_directory_fast(abs_dir)
                   
                    stage1_done = 'stage1_abstraction.done' in abs_scan['done_files']
                   
                    for mine_dir_name in abs_scan['subdirs']:
                        if mine_dir_name.startswith('mine_'):
                            total_combinations += 1
                            mine_dir = abs_dir / mine_dir_name
                            mining_dirs.append((mine_dir, fold_name, abs_dir_name, mine_dir_name, stage1_done))
       
        # Process mining directories in parallel
        stage_counts = defaultdict(int)
        tirps_stats = {'total': 0, 'models_built': 0}
        batch_stats = {'total': 0, 'expected': 0}
        results_stats = {'total': 0}
       
        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(mining_dirs))) as executor:
            future_to_mining = {
                executor.submit(self.scan_mining_combination_fast, mine_dir): (mine_dir, fold_name, abs_dir_name, mine_dir_name, stage1_done)
                for mine_dir, fold_name, abs_dir_name, mine_dir_name, stage1_done in mining_dirs
            }
           
            for future in concurrent.futures.as_completed(future_to_mining):
                mine_dir, fold_name, abs_dir_name, mine_dir_name, stage1_done = future_to_mining[future]
               
                try:
                    mine_progress = future.result()
                   
                    # Count stages
                    if stage1_done:
                        stage_counts['stage1'] += 1
                    if mine_progress['stage2_done']:
                        stage_counts['stage2'] += 1
                    if mine_progress['tirps_count'] > 0 and mine_progress['stage3_models_built'] >= mine_progress['tirps_count']:
                        stage_counts['stage3'] += 1
                    if mine_progress['stage4_complete']:
                        stage_counts['stage4'] += 1
                    if mine_progress['stage5_results'] > 0:
                        stage_counts['stage5'] += 1
                   
                    # Aggregate stats
                    tirps_stats['total'] += mine_progress['tirps_count']
                    tirps_stats['models_built'] += mine_progress['stage3_models_built']
                    batch_stats['total'] += mine_progress['stage4_batches_done']
                    batch_stats['expected'] += mine_progress['stage4_expected_batches']
                    results_stats['total'] += mine_progress['stage5_results']
                   
                except Exception as e:
                    print(f"Error processing {mine_dir}: {e}")
       
        return {
            'dataset_name': dataset_name,
            'stage0_done': stage0_done,
            'total_folds': len(fold_dirs),
            'total_combinations': total_combinations,
            'stage_counts': dict(stage_counts),
            'tirps_stats': tirps_stats,
            'batch_stats': batch_stats,
            'results_stats': results_stats
        }
   
    def generate_report_data(self):
        """Generate report data without printing"""
        start_time = time.time()
       
        # Find dataset directories quickly
        try:
            dataset_dirs = [d for d in self.base_dir.iterdir()
                          if d.is_dir() and d.name not in ('logs_old', 'logs', 'sbatch_scripts', 'temp_files', 'OOM_resubmit_scripts','cross_dataset_plots')]
        except OSError:
            return None, f"ERROR: Cannot access base directory {self.base_dir}"
       
        if not dataset_dirs:
            return None, "WARNING: No dataset directories found"
       
        # Process datasets in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(dataset_dirs))) as executor:
            future_to_dataset = {
                executor.submit(self.scan_dataset_parallel, dataset_dir): dataset_dir
                for dataset_dir in dataset_dirs
            }
           
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_dir = future_to_dataset[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error scanning {dataset_dir.name}: {e}")
       
        # Generate summary statistics
        summary = self.calculate_summary_stats(all_results)
        
        return {
            'results': all_results,
            'summary': summary,
            'scan_duration': time.time() - start_time
        }, None
   
    def calculate_summary_stats(self, all_results):
        """Calculate summary statistics efficiently"""
        summary = {
            'total_datasets': len(all_results),
            'datasets_stage0_done': sum(1 for r in all_results if r['stage0_done']),
            'total_folds': sum(r['total_folds'] for r in all_results),
            'total_combinations': sum(r['total_combinations'] for r in all_results),
            'stage_totals': defaultdict(int),
            'total_tirps': sum(r['tirps_stats']['total'] for r in all_results),
            'total_models_built': sum(r['tirps_stats']['models_built'] for r in all_results),
            'total_batches': sum(r['batch_stats']['total'] for r in all_results),
            'total_expected_batches': sum(r['batch_stats'].get('expected', 0) for r in all_results),
            'total_results': sum(r['results_stats']['total'] for r in all_results)
        }
       
        # Aggregate stage counts
        for result in all_results:
            for stage, count in result['stage_counts'].items():
                summary['stage_totals'][stage] += count
       
        return summary


class LogAnalyzer:
    """Analyzes SLURM error log files (.err) to detect failed jobs and categorize failure reasons"""
    
    def __init__(self, logs_dir):
        self.logs_dir = Path(logs_dir)
        
        # Patterns to detect different failure types
        self.oom_patterns = [
            # SLURM-specific OOM patterns
            r'slurmstepd.*Detected.*oom_kill event',
            r'Some of the step tasks have been OOM Killed',
            r'OOM Killed',
            r'slurm_script.*Killed.*python',  # Catches "slurm_script: line X: PID Killed python"
            
            # Generic OOM patterns
            r'slurmstepd.*Killed process.*due to cgroup out-of-memory',
            r'slurmstepd.*Exceeded memory limit',
            r'slurmstepd.*process.*killed by signal 9',
            r'Out of memory',
            r'oom-kill',
            r'Memory limit exceeded',
            r'Killed.*memory',
            r'CANCELLED.*DUE TO NODE FAILURE',
            r'slurmstepd.*Exceeded.*memory'
        ]
        
        self.other_failure_patterns = [
            r'slurmstepd.*error',
            r'FAILED',
            r'TIMEOUT',
            r'CANCELLED',
            r'Exception',
            r'Error',
            r'Traceback',
            r'exit code [1-9]',
            r'Command.*failed'
        ]
    
    def extract_stage_from_filename(self, log_file_path):
        """Extract stage number from log file name"""
        filename = os.path.basename(log_file_path)
        
        # Look for stage patterns in filename (stg0_, stg1_, etc.)
        stage_match = re.search(r'stg(\d+)_', filename)
        if stage_match:
            return int(stage_match.group(1))
        
        # Look for monitor logs (not experiment stages)
        if 'monitor_' in filename:
            return 'monitor'
            
        # Default to unknown if no stage found
        return 'unknown'

    def analyze_log_file(self, log_file_path):
        """Analyze a single log file for failure patterns and extract stage info"""
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract stage information
            stage = self.extract_stage_from_filename(log_file_path)
            
            # Check for OOM first (higher priority)
            for pattern in self.oom_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return 'OOM', stage
            
            # Check for other failures
            for pattern in self.other_failure_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return 'OTHER', stage
            
            # If no failure patterns found, assume success
            return 'SUCCESS', stage
            
        except Exception as e:
            print(f"Error analyzing log file {log_file_path}: {e}")
            return 'UNKNOWN', 'unknown'
    
    def analyze_logs(self, hours_lookback=24):
        """Analyze all ERROR log files from the last X hours"""
        if not self.logs_dir.exists():
            return {'oom_failures': 0, 'other_failures': 0, 'total_analyzed': 0, 'error': 'Logs directory not found'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        
        # Find only ERROR log files (SLURM .err files)
        error_patterns = ['*.err', 'slurm-*.err']
        log_files = []
        
        for pattern in error_patterns:
            log_files.extend(self.logs_dir.glob(pattern))
        
        # Filter by modification time
        recent_logs = []
        for log_file in log_files:
            try:
                if datetime.fromtimestamp(log_file.stat().st_mtime) > cutoff_time:
                    recent_logs.append(log_file)
            except Exception:
                continue
        
        # Analyze logs in parallel
        oom_count = 0
        other_failures = 0
        stage_failures = {}  # Track failures by stage
        oom_failure_details = []  # Collect detailed OOM failure info
        other_failure_details = []  # Collect detailed other failure info
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_log = {
                executor.submit(self.analyze_log_file, log_file): log_file
                for log_file in recent_logs
            }
            
            for future in concurrent.futures.as_completed(future_to_log):
                log_file = future_to_log[future]
                try:
                    result, stage = future.result()
                    
                    # Initialize stage in tracking dict if not present
                    if stage not in stage_failures:
                        stage_failures[stage] = {'oom': 0, 'other': 0, 'total': 0}
                    
                    # Count overall failures
                    if result == 'OOM':
                        oom_count += 1
                        stage_failures[stage]['oom'] += 1
                        stage_failures[stage]['total'] += 1
                        
                        # Collect detailed OOM failure info
                        oom_failure_details.append({
                            'err_file': str(log_file),
                            'stage': stage,
                            'timestamp': datetime.fromtimestamp(log_file.stat().st_mtime)
                        })
                        
                    elif result == 'OTHER':
                        other_failures += 1
                        stage_failures[stage]['other'] += 1
                        stage_failures[stage]['total'] += 1
                        
                        # Collect detailed other failure info
                        other_failure_details.append({
                            'err_file': str(log_file),
                            'stage': stage,
                            'timestamp': datetime.fromtimestamp(log_file.stat().st_mtime)
                        })
                        
                except Exception as e:
                    print(f"Error processing {log_file}: {e}")
        
        return {
            'oom_failures': oom_count,
            'other_failures': other_failures,
            'total_analyzed': len(recent_logs),
            'hours_lookback': hours_lookback,
            'stage_failures': stage_failures,
            'oom_failure_details': oom_failure_details,  # New field with detailed OOM info
            'other_failure_details': other_failure_details  # New field with detailed other info
        }


class EmailNotifier:
    """Handles sending email notifications"""
    
    def __init__(self, smtp_server='localhost', smtp_port=25, sender_email=None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or 'experiment_monitor@localhost'
    
    def send_status_email(self, recipient, subject, report_data, failure_data, oom_resubmit_data=None):
        """Send status email with experiment progress and failure analysis"""
        
        # Generate email body
        body = self.generate_email_body(report_data, failure_data, oom_resubmit_data)
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.send_message(msg)
            return True, "Email sent successfully"
        except Exception as e:
            return False, f"Failed to send email: {e}"
    
    def generate_email_body(self, report_data, failure_data, oom_resubmit_data=None):
        """Generate email body with experiment status and failure analysis"""
        
        if not report_data:
            return "Error: Could not generate experiment report"
        
        results = report_data['results']
        summary = report_data['summary']
        scan_duration = report_data['scan_duration']
        
        body = f"""
EXPERIMENT PROGRESS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Scan Duration: {scan_duration:.2f} seconds

================================================================================
OVERALL SUMMARY
================================================================================
Total datasets: {summary['total_datasets']}
Total folds: {summary['total_folds']}
Total parameter combinations: {summary['total_combinations']:,}
Total TIRPs found: {summary['total_tirps']:,}
Total models built: {summary['total_models_built']:,}
Total prediction batches: {summary['total_batches']:,}/{summary['total_expected_batches']:,}
"""
        
        if summary['total_expected_batches'] > 0:
            batch_percentage = (summary['total_batches'] / summary['total_expected_batches']) * 100
            body += f"Batch completion rate: {batch_percentage:.1f}%\n"
        
        body += f"Total result files: {summary['total_results']:,}\n\n"
        
        # Progress by stage
        body += "Progress by stage:\n"
        body += f"   Stage 0 (data splitting): {summary['datasets_stage0_done']}/{summary['total_datasets']} datasets\n"
        
        total_combos = summary['total_combinations']
        if total_combos > 0:
            stage_names = {
                1: "abstraction",
                2: "mining", 
                3: "model building",
                4: "prediction",
                5: "aggregation"
            }
            for stage_num in range(1, 6):
                stage_key = f'stage{stage_num}'
                completed = summary['stage_totals'].get(stage_key, 0)
                percentage = (completed / total_combos) * 100
                body += f"   Stage {stage_num} ({stage_names[stage_num]}): {completed:,}/{total_combos:,} ({percentage:.1f}%)\n"
        
        # Dataset details
        body += "\n" + "="*80 + "\n"
        body += "DATASET DETAILS\n"
        body += "="*80 + "\n"
        
        for result in sorted(results, key=lambda x: x['dataset_name']):
            name = result['dataset_name']
            stage0_status = "DONE" if result['stage0_done'] else "PENDING"
            dataset_combos = result['total_combinations']
            
            body += f"\n{name}\n"
            body += f"   Stage 0: {stage0_status}\n"
            body += f"   Folds: {result['total_folds']}\n"
            body += f"   Combinations: {dataset_combos:,}\n"
            
            if dataset_combos > 0:
                stage_counts = result['stage_counts']
                for stage_num in range(1, 6):
                    stage_key = f'stage{stage_num}'
                    completed = stage_counts.get(stage_key, 0)
                    if completed > 0:
                        percentage = (completed / dataset_combos) * 100
                        body += f"   Stage {stage_num}: {completed:,}/{dataset_combos:,} ({percentage:.1f}%)\n"
                
                # Additional stats
                tirps = result['tirps_stats']
                if tirps['total'] > 0:
                    body += f"   TIRPs: {tirps['total']:,} (models: {tirps['models_built']:,})\n"
                
                batch_stats = result['batch_stats']
                expected_batches = batch_stats.get('expected', 0)
                if expected_batches > 0:
                    batch_completion = (batch_stats['total'] / expected_batches) * 100
                    body += f"   Batches: {batch_stats['total']:,}/{expected_batches:,} ({batch_completion:.1f}%)\n"
                
                if result['results_stats']['total'] > 0:
                    body += f"   Results: {result['results_stats']['total']:,}\n"
        
        # Failure analysis
        body += "\n" + "="*80 + "\n"
        body += "FAILURE ANALYSIS\n"
        body += "="*80 + "\n"
        
        if failure_data.get('error'):
            body += f"Error analyzing failures: {failure_data['error']}\n"
        else:
            body += f"Analysis period: Last {failure_data['hours_lookback']} hours\n"
            body += f"Total error log files analyzed: {failure_data['total_analyzed']}\n"
            body += f"OOM (Out of Memory) failures: {failure_data['oom_failures']}\n"
            body += f"Other failures: {failure_data['other_failures']}\n"
            body += f"Total failures: {failure_data['oom_failures'] + failure_data['other_failures']}\n"
            body += f"Note: Only .err log files are analyzed for failures\n\n"
            
            # Add stage-specific failure breakdown
            stage_failures = failure_data.get('stage_failures', {})
            if stage_failures:
                body += "Failures by Stage:\n"
                # Sort stages (numeric stages first, then special cases)
                sorted_stages = sorted(stage_failures.keys(), key=lambda x: (isinstance(x, str), x))
                
                for stage in sorted_stages:
                    failures = stage_failures[stage]
                    if failures['total'] > 0:
                        if isinstance(stage, int):
                            stage_name = f"Stage {stage}"
                            stage_descriptions = {
                                0: "data splitting",
                                1: "abstraction", 
                                2: "mining",
                                3: "model building",
                                4: "prediction",
                                5: "aggregation"
                            }
                            stage_desc = stage_descriptions.get(stage, "unknown")
                            body += f"   {stage_name} ({stage_desc}): {failures['total']} failures "
                        else:
                            body += f"   {stage}: {failures['total']} failures "
                        
                        body += f"(OOM: {failures['oom']}, Other: {failures['other']})\n"
                
                if not any(failures['total'] > 0 for failures in stage_failures.values()):
                    body += "   No stage-specific failures detected\n"
            else:
                body += "Stage-specific failure data not available\n"
        
        # OOM Resubmission section
        if oom_resubmit_data:
            body += "\n" + "="*80 + "\n"
            body += "OOM RESUBMISSION ACTIONS\n"
            body += "="*80 + "\n"
            
            resubmitted_jobs = oom_resubmit_data.get('resubmitted_jobs', [])
            if resubmitted_jobs:
                submitted_count = sum(1 for job in resubmitted_jobs if job.get('submitted', False))
                failed_count = len(resubmitted_jobs) - submitted_count
                
                body += f"OOM resubmission actions taken: {len(resubmitted_jobs)} jobs processed\n"
                body += f"   Successfully submitted: {submitted_count}\n"
                body += f"   Submission failed: {failed_count}\n\n"
                
                for job in resubmitted_jobs:
                    body += f"   Job: {job['job_name']}\n"
                    body += f"   Stage: {job['stage']}\n"
                    body += f"   Original Job ID: {job.get('original_slurm_job_id', 'N/A')}\n"
                    body += f"   New memory: {job['new_memory']}GB\n"
                    body += f"   Script: {os.path.basename(job['script_path'])}\n"
                    
                    if job.get('submitted', False):
                        body += f"   ✓ Submitted automatically -> New Job ID: {job.get('new_job_id', 'N/A')}\n"
                    else:
                        body += f"   ✗ Submission failed: {job.get('error', 'Unknown error')}\n"
                        body += f"   → Manual submission needed: sbatch {job['script_path']}\n"
                    body += "\n"
                
                body += f"Resubmission scripts directory: {oom_resubmit_data.get('resubmit_dir', 'N/A')}\n"
            else:
                body += "No OOM jobs were resubmitted this cycle.\n"
                if failure_data.get('oom_failures', 0) > 0:
                    body += "Either they were recently resubmitted or corresponding .out files were not found.\n"
        
        body += "\n" + "="*80 + "\n"
        body += "END OF REPORT\n"
        
        return body

def process_other_failures(other_failures, logs_dir):
    """Copy logs for non-OOM failures to a separate directory for easier debugging"""
    if not other_failures:
        return 0
        
    logs_path = Path(logs_dir)
    other_errors_dir = logs_path.parent / "Other_error_logs"
    other_errors_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    for failure in other_failures:
        err_file = Path(failure['err_file'])
        
        # Determine destination paths
        dest_err = other_errors_dir / err_file.name
        
        # Check and copy err file if it doesn't exist
        if not dest_err.exists():
            try:
                shutil.copy2(err_file, dest_err)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {err_file}: {e}")
        
        # Also copy the corresponding out file if it exists
        out_file_path = str(err_file).replace('.err', '.out')
        out_file = Path(out_file_path)
        
        if out_file.exists():
            dest_out = other_errors_dir / out_file.name
            if not dest_out.exists():
                try:
                    shutil.copy2(out_file, dest_out)
                except Exception as e:
                    print(f"Failed to copy {out_file}: {e}")
                    
    return copied_count


def main():
    """Main monitoring loop"""
    parser = argparse.ArgumentParser(description='Experiment Progress Monitor with Email Notifications')
    parser.add_argument('--base_dir', type=str, required=True, help='Base experiment directory')
    parser.add_argument('--logs_dir', type=str, required=True, help='Logs directory path')
    parser.add_argument('--email', type=str, default='eldarzos@post.bgu.ac.il', help='Email recipient')
    parser.add_argument('--interval', type=int, default=6, help='Email interval in hours')
    parser.add_argument('--failure_lookback', type=int, default=24, help='Hours to look back for failures')
    parser.add_argument('--smtp_server', type=str, default='smtp.bgu.ac.il', help='SMTP server')
    parser.add_argument('--smtp_port', type=int, default=25, help='SMTP port')
    parser.add_argument('--sender_email', type=str, default='experiment_monitor@ise-pheno-12', help='Sender email')
    parser.add_argument('--run_once', action='store_true', help='Run once and exit (for testing)')
    
    args = parser.parse_args()
    
    # Initialize components
    monitor = FastExperimentMonitor(args.base_dir)
    log_analyzer = LogAnalyzer(args.logs_dir)
    email_notifier = EmailNotifier(args.smtp_server, args.smtp_port, args.sender_email)
    oom_manager = OOMResubmissionManager(args.logs_dir, args.base_dir)
    
    print(f"Starting experiment monitor...")
    print(f"Base directory: {args.base_dir}")
    print(f"Logs directory: {args.logs_dir}")
    print(f"Email recipient: {args.email}")
    print(f"Email interval: {args.interval} hours")
    print(f"Failure lookback: {args.failure_lookback} hours")
    
    if args.run_once:
        print("Running once for testing...")
        
        # Generate progress report
        print("Generating progress report...")
        report_data, error = monitor.generate_report_data()
        if error:
            print(f"Error generating report: {error}")
            return 1
        
        # Analyze failures
        print("Analyzing failures...")
        failure_data = log_analyzer.analyze_logs(args.failure_lookback)
        
        # Process OOM failures and create resubmission scripts
        print("Processing OOM failures...")
        oom_resubmit_data = None
        if failure_data.get('oom_failure_details'):
            resubmitted_jobs = oom_manager.process_oom_failures(failure_data['oom_failure_details'])
            oom_resubmit_data = {
                'resubmitted_jobs': resubmitted_jobs,
                'resubmit_dir': str(oom_manager.resubmit_dir)
            }
            if resubmitted_jobs:
                print(f"Created {len(resubmitted_jobs)} OOM resubmission scripts")
        
        # Process other failures
        print("Processing other failures...")
        if failure_data.get('other_failure_details'):
            copied = process_other_failures(failure_data['other_failure_details'], args.logs_dir)
            if copied > 0:
                print(f"Copied {copied} new 'other' failure log files to Other_error_logs directory")
        
        # Send email
        subject = f"Experiment Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        success, message = email_notifier.send_status_email(args.email, subject, report_data, failure_data, oom_resubmit_data)
        
        if success:
            print("Email sent successfully!")
        else:
            print(f"Failed to send email: {message}")
            return 1
        
        return 0
    
    # Main monitoring loop
    last_email_time = datetime.now() - timedelta(hours=args.interval)  # Send immediately on start
    first_run = True
    
    while True:
        try:
            current_time = datetime.now()
            
            # Check if it's time to send email (immediately on first run, then every interval)
            if current_time - last_email_time >= timedelta(hours=args.interval):
                is_initial_report = first_run
                if first_run:
                    print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Generating initial status report...")
                    first_run = False
                else:
                    print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Generating periodic status report...")
                
                # Generate progress report
                report_data, error = monitor.generate_report_data()
                if error:
                    print(f"Error generating report: {error}")
                    time.sleep(300)  # Wait 5 minutes before retry
                    continue
                
                # Analyze failures
                failure_data = log_analyzer.analyze_logs(args.failure_lookback)
                
                # Process OOM failures and create resubmission scripts
                oom_resubmit_data = None
                if failure_data.get('oom_failure_details'):
                    resubmitted_jobs = oom_manager.process_oom_failures(failure_data['oom_failure_details'])
                    oom_resubmit_data = {
                        'resubmitted_jobs': resubmitted_jobs,
                        'resubmit_dir': str(oom_manager.resubmit_dir)
                    }
                    if resubmitted_jobs:
                        print(f"Created {len(resubmitted_jobs)} OOM resubmission scripts")
                
                # Process other failures
                if failure_data.get('other_failure_details'):
                    copied = process_other_failures(failure_data['other_failure_details'], args.logs_dir)
                    if copied > 0:
                        print(f"Copied {copied} new 'other' failure log files to Other_error_logs directory")
                
                # Send email
                subject = f"Experiment Status Report {'(Initial)' if is_initial_report else ''} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                success, message = email_notifier.send_status_email(args.email, subject, report_data, failure_data, oom_resubmit_data)
                
                if success:
                    print(f"Status email sent successfully to {args.email}")
                    last_email_time = current_time
                else:
                    print(f"Failed to send email: {message}")
                
                # Print brief status to console
                if report_data:
                    summary = report_data['summary']
                    print(f"Current status: {summary['total_combinations']:,} combinations, "
                          f"{summary['total_batches']:,}/{summary['total_expected_batches']:,} batches, "
                          f"{failure_data['oom_failures']} OOM failures, "
                          f"{failure_data['other_failures']} other failures")
            
            # Wait before next check (check every 30 minutes)
            time.sleep(1800)
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user.")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(300)  # Wait 5 minutes before retry
    
    print("Monitoring stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 