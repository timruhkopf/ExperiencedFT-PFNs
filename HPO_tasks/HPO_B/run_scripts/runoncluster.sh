#!/bin/bash
#SBATCH --partition=2080-galvani
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=3-00:00             # Runtime in D-HH:MM
#SBATCH --mem=32G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:0

search_space_id=$1
policy=$2

# Set the output and error file paths for SLURM
#SBATCH --output=${output_path}
#SBATCH --error=${error_path}

cd ..
conda run -n project_dmpfn python3 -u main.py  --search_space_id $search_space_id --method $policy