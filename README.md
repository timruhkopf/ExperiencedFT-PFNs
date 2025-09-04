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

```shell
conda create -n ft-pfn python=3.10
cd ExperiencedFT-PFNs


git clone git@github.com:timruhkopf/ifBO.git
git checkout main
<<<<<<< HEAD
mv ifBO ifBO_main
#pip install -U ifBO
pip install -e ifBO_main
=======
pip install -U ifBO_main
# collect the padding changes

PYTHONPATH=[...]/ExperiencedFT-PFNs/ifBO_main/ifbo:$PYTHONPATH


# LCBench benchmark data: 
#ExperiencedFT-PFNs/src/ifBO_icml2024$ python -m mfpbench download --benchmark lcbench-tabular
cd src/ifBO_icml24
pip install -r requirements.txt


`
python -m mfpbench download --benchmark lcbench-tabular  --data-dir $root/ExperiencedFT-PFNs/data/
python -m mfpbench download --benchmark pd1-tabular  --data-dir $root/ExperiencedFT-PFNs/data/
python -m mfpbench download --benchmark taskset  --data-dir $root/ExperiencedFT-PFNs/data/
`
>>>>>>> origin/version/0.7.0-jbc-err-variants

./setup.sh

```

# a potential bug
in line 100 of 
ExperiencedFT-PFNs/ifBO_icml2024/src/mf-prior-bench/src/mfpbench/taskset_tabular/processing/process.py

```python
    df[["optimizer", "config_id"]] = df.apply(
        lambda row: row.name.split("_seed"),
        axis=1,
        result_type="expand",
    )
    
    parsed = df.index.to_series().str.extract(r"^(.*)_seed(\d+)$")
    parsed.columns = ["optimizer", "config_id"]

    # Identify invalid config_id values
    invalid_rows = parsed[~parsed["config_id"].str.fullmatch(r"\d+")]
    if not invalid_rows.empty:
        print("Invalid rows detected:")
        print(invalid_rows)
    # Safe conversion
    df["optimizer"] = parsed["optimizer"].astype("category")
    df["config_id"] = pd.to_numeric(parsed["config_id"], errors="coerce").astype("Int32")

```

Then in data folder:


```python
import ifbo
model = ifbo.surrogate.FTPFN(version="0.0.1")
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
