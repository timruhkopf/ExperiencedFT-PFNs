#!/bin/bash
#SBATCH --job-name=pfns_hpo
#SBATCH --output=pfns_hpo_%A_%a.out
#SBATCH --error=pfns_hpo_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

module load Miniforge3

if [[ $HOME == /mnt/home* ]]; then
    conda activate $HOME/.conda/envs/ft-pfn
    BIGWORK=~
else
    conda activate $BIGWORK/envs/ft-pfn-experimental
fi

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH

cd $BIGWORK/$REPONAME/ifBO_icml2024/src/pfns_hpo

benchmarks=($BENCHMARK_LIST)
BENCHMARK=${benchmarks[$SLURM_ARRAY_TASK_ID]}

echo "Running algorithm=$ALGO on benchmark=$BENCHMARK (task ID: $SLURM_ARRAY_TASK_ID)"

HYDRA_FULL_ERROR=1 python pfns_hpo/run.py algorithm="$ALGO" benchmark="$BENCHMARK" experiment_group='ifbo_reproduction'
