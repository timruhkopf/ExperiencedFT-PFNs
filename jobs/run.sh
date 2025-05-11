#!/bin/bash
# jobs/submit_array.sh

PARAM_FILE="$BIGWORK/ExperiencedFT-PFNs/jobs/job_params.txt"
NUM_PARAMS=$(wc -l < "$PARAM_FILE")

INTERVAL_SIZE=25
INTERVAL_START=0
INTERVAL_END=200
NUM_INTERVALS=$(( (INTERVAL_END - INTERVAL_START) / INTERVAL_SIZE ))
EXPERIMENT_NAME=$1

TOTAL_TASKS=$((NUM_PARAMS * NUM_INTERVALS))

for ((i=0; i<$NUM_PARAMS; i++)); do
    LINE=$(sed -n "$((i + 1))p" "$PARAM_FILE")
    MODEL=$(echo $LINE | awk '{print $1}')
    BENCH=$(echo $LINE | awk '{print $2}')
    SPLIT_SEED=$(echo $LINE | awk '{print $3}')

    JOB_NAME="${MODEL}_${BENCH}_${SPLIT_SEED}"
    if [[ "$MODEL" == "distill" ]]; then
        PARTITION="ai,tnt"
        GRES="--gres=gpu:1"
        EXCLUDE=""
        TIME="--time=48:00:00"
        DEVICE="cuda"
        CPUS=8
    else
        PARTITION="ai,taurus,amo"
        GRES=""
        EXCLUDE="--exclude=ai-n[001-004],ai-n009"
        TIME="--time=20:00:00"
        DEVICE="cpu"
        CPUS=32
    fi

    sbatch --job-name=$JOB_NAME \
           --array=0-$((NUM_INTERVALS-1)) \
           --cpus-per-task=$CPUS \
           --mem=8G \
           $TIME \
           --output=${JOB_NAME}_%a.out \
           --error=${JOB_NAME}_%a.err \
           --partition=$PARTITION \
           $GRES \
           $EXCLUDE \
           $BIGWORK/ExperiencedFT-PFNs/jobs/array_job.sh "$EXPERIMENT_NAME" "$DEVICE" "$LINE"

done


# watch -n 1 "squeue --me -o '%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R'"

# filter for running jobs (PD instead of R for pending)
#watch -n 1 "squeue --me -t R -o '%.18i %.9P %.25j %.8u %.2t %.10M %.6D %R'"

