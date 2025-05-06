# Experienced FT-PFNs

Refine the prior of a PFN using prior experiences

The idea of this paper is a conceptual follow-up of [MASIF](), in which a meta distribution over HPO 
tasks exists and that we can draw experience from to refine our prior and basically warm start the Freeze-Thaw PFN
from [ifBO]() without any modifications to the pretrained PFN model.

Core idea here is, that we can use past runs to 
1. formulate good default configurations we'd like to try, to disambiguate 
how reliable the previous experiences are. 
2. Once we have at least some data, we verify the reliability by checking the nll of the current collected context when predicted
through the PFN endowed with the previous experience i.e. $$(\lambda_j, t_j, y_j) \in D^{current}, (\lambda', t', y') \in D^{related}$$
and try to predict  $$p\left(y | \{(\lambda_j, t_j, .)\}_j ,  \{(\lambda_i', t_i', y_i')_i\}_i \in D^{related}\right)$$
3. using this measure of reliability, we can endow the PFN with additional context either 
   1. sampled from the previous experience, according to reliability
   2. based on the distilled experiences similar to [In-Context data Distillation with TabPFNs](https://arxiv.org/abs/2402.06971)
   3. Consider locality; i.e. relevant and proximite experience to our query $\{(\lambda_j, t_j, .)\}_j$ but collected from $ D^{related}$

# Installation

```bash
conda create -n ft-pfn python=3.10
cd ExperiencedFT-PFNs

root=$(pwd)
pip install -e .


# FIXME: move this install directly to setup.py
cd src
git clone git@github.com:automl/ifBO.git
git checkout icml-2024
mv ifBO ifBO_icml24
pip install -U ifBO_icml24

git clone git@github.com:automl/ifBO.git
git checkout main
mv ifBO ifBO_main
#pip install -U ifBO
pip install -e ifBO_main


# LCBench benchmark data: 
#ExperiencedFT-PFNs/src/ifBO_icml2024$ python -m mfpbench download --benchmark lcbench-tabular
cd src/ifBO_icml24
pip install -r requirements.txt



python -m mfpbench download --benchmark lcbench-tabular  --data-dir $root/ExperiencedFT-PFNs/data/
python -m mfpbench download --benchmark pd1-tabular  --data-dir $root/ExperiencedFT-PFNs/data/
python -m mfpbench download --benchmark taskset  --data-dir $root/ExperiencedFT-PFNs/data/



```

# How to run the Experiments: 


```shell
python ExperiencedFT-PFNs/main_ftpfn.py \
   device=cpu \
   model=pfn_mixture \
   benchmark=sanity_same \
   seed=1 \
   allocation_seeds=[42,43] # this will run over multiple instantiations of budget per task and alpha allocations

```

Plotting: 
```python

```






# Credit: 
We build upon the work of ifBO directly and use their code and models.

```bibtex
@inproceedings{
  rakotoarison-icml24,
  title={In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization},
  author={H. Rakotoarison and S. Adriaensen and N. Mallik and S. Garibov and E. Bergman and F. Hutter},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=VyoY3Wh9Wd}
}

```

# Citation 
```bibtex

```