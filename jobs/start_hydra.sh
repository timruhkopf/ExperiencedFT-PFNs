#!/bin/bash
#SBATCH --job-name=start_hydra
#SBATCH --output=start_hydra%j.out
#SBATCH --error=start_hydra%j.err

#SBATCH --nodes=1
#SBATCH --partition=ai,taurus,amo
#SBATCH --exclude=ai-n[001-004],ai-n009

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

#SBATCH --mem=8GB

commit_hash=$(git log -1 --pretty=format:"%h")

module load Miniforge3
echo 'activating conda'

REPONAME=Experienced-FT-PFNs

conda activate $BIGWORK/envs/eft-pfn2
export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH


export CUBLAS_WORKSPACE_CONFIG=:4096:8


# Initialize an empty array to store Hydra overrides
HYDRA_OVERRIDES=()
MULTIRUN_FLAG=""
CFG_FLAG=""
RESOLVE_FLAG=""

# Parse all command-line arguments as Hydra overrides or check for --multirun flag
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--multirun" ]]; then
        MULTIRUN_FLAG="--multirun"
    elif [[ "$1" == "--cfg=all" ]]; then
        CFG_FLAG="--cfg=all"
    elif [[ "$1" == "--resolve" ]]; then
        RESOLVE_FLAG="--resolve"
    else
        HYDRA_OVERRIDES+=("$1")
    fi
    shift
done

# Construct the Hydra command
HYDRA_CMD="python main.py"

# Add all Hydra overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    HYDRA_CMD+=" $override"
done

# Prepare final command with commit hash and multirun flag if specified
FINAL_CMD="$HYDRA_CMD commit_hash=$commit_hash slurm_id=$SLURM_JOB_ID"

if [[ -n "$MULTIRUN_FLAG" ]]; then
    FINAL_CMD+=" $MULTIRUN_FLAG"
fi

if [[ -n "$CFG_FLAG" ]]; then
    FINAL_CMD+=" $CFG_FLAG"
fi

if [[ -n "$RESOLVE_FLAG" ]]; then
    FINAL_CMD+=" $RESOLVE_FLAG"
fi

export HYDRA_FULL_ERROR=1
echo "Executing: $FINAL_CMD"
HYDRA_DIR=$(eval "$FINAL_CMD"  | grep "Sweep dir:" | awk -F': ' '{print $2}')

wait

echo "Hydra output directory: $HYDRA_DIR"

echo "Running read_data:"
DIR=$BIGWORK/$REPONAME/$HYDRA_DIR
python $BIGWORK/$REPONAME/src/read_data.py \
  --root_dir $DIR \
  --file_pattern "results.csv" \
  --keys "[\"experiment_name\",\"model.meta.name\",\"dataset.meta.name\",\"budget\",\"fidelity_seed\"]" \
  - to_csv $DIR/joint_results.csv

wait


#salloc --partition=ai  --nodes=1  --time=02:00:00  --cpus-per-task=8  --gres=gpu:1  --mem=8GB
#HYDRA_FULL_ERROR=1 python main.py smactuner.epochs=1 smactuner='al' scheduler='sh' +budgets=[0.0001,0.0002,0.0003] n_init_cfgs=2 dataset='cifar10' smactuner.batch_size=512 smactuner.track_scores=False seed=1 al_method='DCOM' dataset.path='/bigwork/nhwpruht/AdaptiveMFSimple/data' +pretrain_epochs=1
