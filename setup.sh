root=$(pwd)
git clone git@github.com:timruhkopf/ifBO.git ifBO_icml2024
cd ifBO_icml2024
git checkout icml-2024
pip install -r requirements.txt
pip install -e ./src/mf-prior-bench
pip install -e ./src/neps_lcpfn_hpo
pip install -e ./src/PFNs4HPO
pip install -e ./src/pfns_hpo


python -m mfpbench download --benchmark lcbench-tabular  --data-dir $root/data/
python -m mfpbench download --benchmark pd1-tabular  --data-dir $root/data/
python -m mfpbench download --benchmark taskset-tabular  --data-dir $root/data/
