#!/bin/bash

# Usage: array_job.sh <experiment_name> <device> "<model> <benchmark> <split_seed>"

EXPERIMENT_NAME=$1
DEVICE=$2
LINE="$3"

# Parse parameters from the line
MODEL=$(echo $LINE | awk '{print $1}')
BENCHMARK=$(echo $LINE | awk '{print $2}')
SPLIT_SEED=$(echo $LINE | awk '{print $3}')

INTERVAL_SIZE=25
INTERVAL_START=0
INTERVAL_END=200

NUM_INTERVALS=$(( (INTERVAL_END - INTERVAL_START) / INTERVAL_SIZE ))

# SLURM_ARRAY_TASK_ID is 0-based, corresponds to interval index
INTERVAL_IDX=$SLURM_ARRAY_TASK_ID

START=$(( INTERVAL_START + INTERVAL_IDX * INTERVAL_SIZE ))
END=$(( START + INTERVAL_SIZE ))

echo "MODEL: $MODEL"
echo "BENCHMARK: $BENCHMARK"
echo "SPLIT_SEED: $SPLIT_SEED"
echo "ALLOCATION_SEEDS: [$START,$END]"
echo "DEVICE: $DEVICE"

bash jobs/start_hydra.sh \
  experiment_name=$EXPERIMENT_NAME \
  device=$DEVICE \
  model=$MODEL \
  benchmark=$BENCHMARK \
  split_seed=$SPLIT_SEED \
  allocation_seeds="[$START,$END]" \
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
