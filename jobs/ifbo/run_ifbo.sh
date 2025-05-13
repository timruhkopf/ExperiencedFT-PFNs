#!/bin/bash
#SBATCH --job-name=ifbo_array
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=ai,tnt
#SBATCH --gres=gpu:1
#SBATCH --array=0-35


#for split in range(3):
#    for algo in ['ifbo', 'ifbo-pfnargmin', 'ifbo-pfn_softmax', 'ifbo-distill']:
#        for bench in ['lcbench', 'taskset', 'pd1']:
#
#            print( f"{bench} {algo} {split}")


# Arguments
EXPERIMENT_NAME=$1

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
#chmod +x /bigwork/nhwpruht/ExperiencedFT-PFNs/jobs/ifbo/start_hydra_ifbo.sh
$BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/start_hydra_ifbo.sh \
    experiment_name=$EXPERIMENT_NAME \
    device=cuda \
    benchmark=$BENCH \
    +algorithm=$MODEL \
    split_seed=$SPLIT_SEED
