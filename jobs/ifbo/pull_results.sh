#!/bin/bash



REPONAME=ExperiencedFT-PFNs
BIGWORK=/home/ruhkopf/PycharmProjects

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH
python main_ifbo.py \
  experiment_group=minimizefail \
  experiment_name=hardcodemax \
  benchmark=synthetic  \
  +benchmark.cls.data_path=/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/data/synthetic_tabular/batch.pt \
  +algorithm=ifbo-pfnimpute-cv \
  split_seed=0 \
  +fold=0 \
  +target_idx=0 \
  device=cpu  \
  +nepsnevals=150 \
  ++algorithm.surrogate_model.cls.min_context_size=3

python main_ifbo.py \
  experiment_group=minimizefail \
  experiment_name=hardcodemax \
  benchmark=synthetic  \
  +benchmark.cls.data_path=/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/data/synthetic_tabular/batch.pt \
  +algorithm=ifbo \
  split_seed=0 \
  +fold=0 \
  +target_idx=0 \
  device=cpu  \
  +nepsnevals=150 \


DIR=$BIGWORK/$REPONAME/results/minimizefail/hardcodemax/

python src/utils/read_neps.py \
--root_dir $DIR \
--csv $DIR/joint_results.csv \
--keys "[\"experiment_name\",\"algorithm.surrogate_model.meta.name\",\"benchmark.meta.name\",\"split_seed\"]"

python src/plots/plot_acq_runs.py \
--file $DIR/joint_results.csv \
--minimize False

#commit_hash=$(git log -1 --pretty=format:"%h")
##echo "Running read_data:"
##DIR=$BIGWORK/$REPONAME/$HYDRA_DIR
#python $BIGWORK/$REPONAME/src/utils/read_neps.py \
#  --root_dir $DIR \
#  --file_pattern "all_losses_and_configs.txt" \
#  --keys "[\"experiment_name\",\"algorithm.surrogate_model.meta.name\",\"benchmark.meta.name\",\"split_seed\"]" \
#  --csv $DIR/joint_results_$commit_hash.csv
#
#DIR=/bigwork/nhwpruht/ExperiencedFT-PFNs/results/test/max-meta-train2
#python $BIGWORK/$REPONAME/src/utils/read_data.py \
#  --root_dir $DIR \
#  --file_pattern "results.csv" \
#  --keys "[\"experiment_name\",\"algorithm.surrogate_model.meta.name\",\"benchmark.meta.name\",\"split_seed\"]" \
#  - to_csv $DIR/joint_results_reliability_${commit_hash}_${$SLURM_JOB_ID}.csv