## Setup

1. Following the instructions in the the main directory in README.md file

2. Install ifbo as well as nessary dependencies from icml24 branch
```
cd src
git clone https://github.com/automl/ifBO.git
cd ifBO
pip install -U ifBO
git checkout icml-2024
mkdir ../ifBO_icml24
cp -r * ../ifBO_icml24
git checkout main
cd ..
pip install -U ifBO_icml24/src/neps_lcpfn_hpo
pip install -U ifBO_icml24/src/mf-prior-bench
pip install -U ifBO_icml24/src/PFNs4HPO
```

3. Download lcbench-tabular
```
python -m mfpbench download --benchmark lcbench-tabular
```
