#!/bin/bash
#SBATCH --job-name=start_hydra
#SBATCH --output=start_hydra%j.out
#SBATCH --error=start_hydra%j.err

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

#SBATCH --mem=8GB

# Set up the environment variables
commit_hash=$(git log -1 --pretty=format:"%h")

REPONAME=ExperiencedFT-PFNs

# on kisski:
module load Miniforge3

if [[ $HOME == /mnt/home* ]]; then
    conda activate $HOME/.conda/envs/ft-pfn
    # if we are on kisski
    BIGWORK=~
else

  conda activate $BIGWORK/envs/ft-pfn-experimental
fi

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH

#HYDRA_FULL_ERROR=1; python $REPONAME/main_ifbo.py benchmark=lcbench device=cuda +algorithm=ifbo ~fold
#python
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
#conda activate $BIGWORK/envs/eft-pfn2
#export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
#export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH


export CUBLAS_WORKSPACE_CONFIG=:4096:8


# Parsing the command line arguments --------------------
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
HYDRA_CMD="python main_ifbo.py"

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

# Parsing the Directory from the overrides --------------------
# Defaults (optional)
EXPERIMENT_NAME=""
EXPERIMENT_GROUP=""

echo "Debugging output"
echo ${HYDRA_OVERRIDES[@]}

# Extract specific values from overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    if [[ "$override" =~ ^experiment_name= ]]; then
        EXPERIMENT_NAME="${override#experiment_name=}"
    elif [[ "$override" =~ ^experiment_group= ]]; then
        EXPERIMENT_GROUP="${override#experiment_group=}"
    fi
done


# LOGGING the execution: ----------------------------------
# Timestamp in readable format
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Escape any quotes in the final command
FINAL_CMD_ESCAPED=$(echo "$FINAL_CMD" | sed 's/"/""/g')

# Log file path
LOG_DIR="$BIGWORK/$REPONAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_log.csv"

# Ensure header is written only once
if [[ ! -f "$LOG_FILE" ]]; then
    echo "timestamp,commit_hash,slurm_job_id,experiment_group,experiment_name,sweep_dir,final_cmd" >> "$LOG_FILE"
fi

# Append CSV line
echo "\"$TIMESTAMP\",\"$commit_hash\",\"$SLURM_JOB_ID\",\"$EXPERIMENT_GROUP\",\"$EXPERIMENT_NAME\",\"$SWEEP_DIR_CLEAN\",\"$FINAL_CMD_ESCAPED\"" >> "$LOG_FILE"



# Execute the command and capture output -------------------
export HYDRA_FULL_ERROR=1

# Temporary file to capture the output
#HYDRA_LOG_OUTPUT_FILE=$(mktemp)

# Run the command, streaming live output and writing to file
eval "$FINAL_CMD" # 2>&1 | tee "$HYDRA_LOG_OUTPUT_FILE"
#EXIT_CODE=${PIPESTATUS[0]}  # Get the exit status of `eval`, not `tee`

## Read the full output into a variable
#HYDRA_LOG_OUTPUT=$(cat "$HYDRA_LOG_OUTPUT_FILE")

## Clean up temp file
#rm "$HYDRA_LOG_OUTPUT_FILE"
#
#
#wait
#
#if [[ $EXIT_CODE -ne 0 ]]; then
#    echo "Error: Command failed with exit code $EXIT_CODE"
#    echo "Output was:"
#    echo "$HYDRA_LOG_OUTPUT"
#    exit $EXIT_CODE
#fi

#echo "Hydra output directory parsed from the main logging: $HYDRA_DIR"

# Collect the data from the direcotry --------------------

#
#
##echo "Running read_data:"
#DIR=$BIGWORK/$REPONAME/results/$EXPERIMENT_GROUP/
#echo "Results directory: $DIR"
#

#wait



#/bigwork/nhwpruht/ExperiencedFT-PFNs/results/test/test-scaled
#wait

# scp -r nhwpruht@transfer.cluster.uni-hannover.de:/bigwork/nhwpruht/ExperiencedFT-PFNs/output/surrogate_new_seeds/joint_results.csv /home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/surrogate_joint_results.csv

#sbatch jobs/ifbo/run_ifbo.sh +fold=0 +flip=False +target_idx=0 experiment_group=unflipped nepsnevals=200

#watch -n 1 nvidia-smi

# salloc --nodes=1 --time=02:00:00 --cpus-per-task=8 --gres=gpu:1 --mem=8GB srun --pty bash #  forces GPU visible allocation on salloc

#srun --pty bash # on kisski connect to the job

#HYDRA_FULL_ERROR=1 python main.py smactuner.epochs=1 smactuner='al' scheduler='sh' +budgets=[0.0001,0.0002,0.0003] n_init_cfgs=2 dataset='cifar10' smactuner.batch_size=512 smactuner.track_scores=False seed=1 al_method='DCOM' dataset.path='/bigwork/nhwpruht/AdaptiveMFSimple/data' +pretrain_epochs=1


#scp -r truhkopf@kisski01.cluster.uni-hannover.de:/mnt/home/truhkopf/ExperiencedFT-PFNs/results/test/joint_results_bdd5b01.csv .
#
#$BIGWORK/ExperiencedFT-PFNs/jobs/ifbo/start_hydra_ifbo.sh   device=cuda benchmark=taskset  +algorithm=ifbo-pfnimpute   split_seed=0    experiment_name=test     +target_idx=0
#
#
#salloc --partition=gpu.test --time=02:00:00 --gres=gpu:1
#
#benchmark=taskset +algorithm=ifbo-pfnsoftmax-cv split_seed=0 experiment_name=pfnsoftmax-cv split_seed=0 +target_idx=0
#
#sbatch --array=0-11 --gres=gpu:1 --partition=ai jobs/ifbo/run_ifbo.sh device=cuda split_seed=0    experiment_name=pfnimpute-1sttrial +target_idx=0
#
#sbatch --array=0-11 --gres=gpu:1  jobs/ifbo/run_ifbo.sh device=cuda split_seed=0    experiment_name=pfnimpute-1sttrial +target_idx=0
#
#bash jobs/ifbo/start_hydra_ifbo.sh benchmark=taskset +algorithm=ifbo-pfnimpute split_seed=0 experiment_name=reliabiltiy_test split_seed=0 +target_idx=0  device=cuda
#
#sbatch --array=0-11 --gres=gpu:1 --partition=ai jobs/ifbo/run_ifbo.sh device=cuda split_seed=0    experiment_name=pfnimpute-larget-train +target_idx=0 fold_size=-1

