#!/bin/bash
#SBATCH --job-name=ifbo_array
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --array=0-25


#for split in range(3):
#    for algo in ['ifbo', 'ifbo-pfnargmin', 'ifbo-pfnsoftmax', 'ifbo-distill']:
#        for bench in ['lcbench', 'taskset', 'pd1']:
#
#            print( f"{bench} {algo} {split}")

# Arguments
if [[ $HOME == /mnt/home* ]]; then
  # if we are on kisski
    BIGWORK=$HOME
fi


# File with parameter lines
PARAM_FILE="$BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/job_params.txt"

# Count the number of lines in the parameter file
NUM_LINES=$(wc -l < "$PARAM_FILE")

# Extract the array range from SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT
# SLURM_ARRAY_TASK_COUNT is available in recent SLURM versions and gives the total number of array tasks
if [[ -z "$SLURM_ARRAY_TASK_COUNT" ]]; then
  # Fallback: try to infer from the array range (e.g., 0-26 means 27 tasks)
  ARRAY_COUNT=$(( $(echo "$SLURM_ARRAY_TASK_MAX" - "$SLURM_ARRAY_TASK_MIN" + 1 | bc) ))
else
  ARRAY_COUNT="$SLURM_ARRAY_TASK_COUNT"
fi

if [[ "$NUM_LINES" -ne "$ARRAY_COUNT" ]]; then
  echo "ERROR: Number of lines in \$PARAM_FILE ($NUM_LINES) does not match number of array jobs ($ARRAY_COUNT)."
  echo "Aborting."
  exit 1
fi




# File with parameter lines
PARAM_FILE="$BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/job_params.txt"

# Get the line corresponding to this array task
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_FILE")

# Parse out the parameters
BENCH=$(echo $LINE | awk '{print $1}')
MODEL=$(echo $LINE | awk '{print $2}')
SPLIT_SEED=$(echo $LINE | awk '{print $3}')

JOB_NAME="${MODEL}_${BENCH}"

# Call your actual job script

# TODO at least once:
chmod +x $BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/start_hydra_ifbo.sh

$BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/start_hydra_ifbo.sh \
    device=cuda \
    benchmark=$BENCH \
    +algorithm=$MODEL \
    split_seed=$SPLIT_SEED \
    $@




#for i in {0..8}; do
#  echo "===== Last 50 lines of ifbo_array_380079_${i}.err ====="
#  tail -n 50 "ifbo_array_380079_${i}.err"
#  echo
#done

#sbatch --array=0-26 jobs/ifbo/run_ifbo.sh ifbo-execution
#
#experiment_name=ifbo-pfnsoftmax benchmark=pd1 +algorithm=ifbo-pfnsoftmax split_seed=0