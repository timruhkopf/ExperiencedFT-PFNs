#!/bin/bash




#EXPERIMENT_NAME=EQW-softmax-flip
#EXPERIMENT_GROUP=debugging_synthetic
#
#export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
#export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
#export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH
#python main_ifbo.py \
#  experiment_group=$EXPERIMENT_GROUP \
#  experiment_name=$EXPERIMENT_NAME \
#  benchmark=synthetic  \
#  +benchmark.cls.data_path=$HOME/PycharmProjects/ExperiencedFT-PFNs/data/synthetic_tabular \
#  +algorithm=ifbo-pfnsoftmax \
#  split_seed=0 \
#  +fold=0 \
#  +target_idx=0 \
#  device=cpu  \
#  +nepsnevals=150 \
#  ++algorithm.surrogate_model.cls.min_context_size=10 \
#  ++algorithm.surrogate_model.cls.flippable_related=True \
#  ++algorithm.surrogate_model.cls.mixture_fn._target_="src.model.mixing.EqualWeights"
##  ++algorithm.surrogate_model.cls.mixture_strategy.projection=False \
##  ++algorithm.surrogate_model.cls.incumbent_calculation="imputation only" \


REPONAME=ExperiencedFT-PFNs
BIGWORK=$HOME/PycharmProjects

EXPERIMENT_NAME=EQW-pfnimpute-cv-flip2
EXPERIMENT_GROUP=debugging_synthetic

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH
python main_ifbo.py \
  experiment_group=$EXPERIMENT_GROUP \
  experiment_name=$EXPERIMENT_NAME \
  benchmark=synthetic  \
  benchmark.cls.seed=0,1,2,3,4,6,7,8,9 \
  +algorithm=ifbo-pfnimpute-cv \
  split_seed=0 \
  +fold=0 \
  +target_idx=0 \
  device=cpu  \
  +nepsnevals=150 \
  ++algorithm.surrogate_model.cls.min_context_size=10 \
  ++algorithm.surrogate_model.cls.flippable_related=True \
  ++algorithm.surrogate_model.cls.mixture_strategy._target_="src.model.mixing.EqualWeights" \
  --multirun
#  ++algorithm.surrogate_model.cls.mixture_strategy.projection=False \
#  ++algorithm.surrogate_model.cls.incumbent_calculation="imputation only"
#  +benchmark.cls.data_path=$HOME/PycharmProjects/ExperiencedFT-PFNs/data/synthetic_tabular \

REPONAME=ExperiencedFT-PFNs
BIGWORK=$HOME/PycharmProjects

EXPERIMENT_NAME=EQW-pfnimpute-cv-flip2
EXPERIMENT_GROUP=debugging_synthetic

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH
python main_ifbo.py \
  experiment_group=$EXPERIMENT_GROUP \
  experiment_name=$EXPERIMENT_NAME \
  benchmark=synthetic  \
  benchmark.cls.seed=0,1,2,3,4,6,7,8,9 \
  +algorithm=ifbo-pfnimpute-cv \
  split_seed=0 \
  +fold=0 \
  +target_idx=0 \
  device=cpu  \
  +nepsnevals=150 \
  ++algorithm.surrogate_model.cls.min_context_size=10 \
  ++algorithm.surrogate_model.cls.flippable_related=True \
  ++algorithm.surrogate_model.cls.mixture_strategy.projection=False \
  ++algorithm.surrogate_model.cls.incumbent_calculation="imputation only" \
  --multirun


python main_ifbo.py \
  experiment_group=$EXPERIMENT_GROUP \
  experiment_name=$EXPERIMENT_NAME \
  benchmark=synthetic  \
  benchmark.cls.seed=0,1,2,3,4,6,7,8,9 \
  +algorithm=ifbo \
  split_seed=0 \
  +fold=0 \
  +target_idx=0 \
  device=cpu  \
  +nepsnevals=150
  --multirun


DIR=$BIGWORK/$REPONAME/results/$EXPERIMENT_GROUP/$EXPERIMENT_NAME/

python $BIGWORK/$REPONAME/src/utils/read_neps.py \
--root_dir $DIR \
--csv $DIR/joint_results.csv \
--keys "[\"experiment_name\",\"algorithm.surrogate_model.meta.name\",\"benchmark.meta.name\",\"split_seed\"]" \
 - empty # just to shut "fire" up about that the return is a pdataframe



python src/plots/plot_acq_runs.py \
--file $DIR/joint_results.csv \
--minimize True \
--title $EXPERIMENT_NAME \
--save=True


#commit_hash=$(git log -1 --pretty=format:"%h")
##echo "Running read_data:"
#
#DIR=/bigwork/nhwpruht/ExperiencedFT-PFNs/results/test/max-meta-train2
#python $BIGWORK/$REPONAME/src/utils/read_data.py \
#  --root_dir $DIR \
#  --file_pattern "results.csv" \
#  --keys "[\"experiment_name\",\"algorithm.surrogate_model.meta.name\",\"benchmark.meta.name\",\"split_seed\"]" \
#  - to_csv $DIR/joint_results_reliability_${commit_hash}_${$SLURM_JOB_ID}.csv