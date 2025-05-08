#!/bin/bash
#SBATCH --job-name=surrogate_array
#SBATCH --array=0-7        # 8 tasks, adjust if you add more lines to job_params.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=surrogate_array%j.out
#SBATCH --error=surrogate_array%j.err
#SBATCH --partition=ai,taurus,amo
#SBATCH --exclude=ai-n[001-004],ai-n009

DEVICE=cpu
PARAM_FILE=job_params.txt

# Read the relevant line for this task
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $PARAM_FILE)
MODEL=$(echo $LINE | awk '{print $1}')
BENCHMARK=$(echo $LINE | awk '{print $2}')
SPLIT_SEED=$(echo $LINE | awk '{print $3}')

bash jobs/start_hydra.sh \
  experiment_name=surrogate \
  device=$DEVICE \
  model=$MODEL \
  benchmark=$BENCHMARK \
  split_seed=$SPLIT_SEED \
  allocation_seeds=[0,200] \
  ~fold \
  ~target_idx

wait


##!/bin/bash
#
#DEVICE=cpu
#MODELS=(pfn_mixture distill)
#SPLIT_SEEDS=(1 ) # 2 3 4 5)
## PFN mixture jobs
#
#for split_seed in "${SPLIT_SEEDS[@]}"
#do
#  for model in "${MODELS[@]}"
#  do
#
#    sbatch jobs/start_hydra.sh \
#      experiment_name=surrogate \
#      device=$DEVICE \
#      model=$model \
#      benchmark=sanity_same \
#      split_seed=$split_seed \
#      allocation_seeds=[0,200] \
#    ~fold \
#    ~target_idx
#
#    sbatch jobs/start_hydra.sh \
#      experiment_name=surrogate \
#      device=$DEVICE \
#      model=$model \
#      benchmark=lcbench \
#      split_seed=$split_seed \
#      allocation_seeds=[0,200] \
#    ~fold \
#    ~target_idx
#
#    sbatch jobs/start_hydra.sh \
#      experiment_name=surrogate \
#      device=$DEVICE \
#      model=$model \
#      benchmark=pd1 \
#      split_seed=$split_seed \
#      allocation_seeds=[0,200] \
#      ~fold \
#      ~target_idx
#
#    sbatch jobs/start_hydra.sh \
#      experiment_name=surrogate \
#      device=$DEVICE \
#      model=$model \
#      benchmark=taskset \
#      split_seed=$split_seed \
#      allocation_seeds=[0,200] \
#      ~fold \
#      ~target_idx
#
#  done
#done
#
#
## DISTILL jobs
