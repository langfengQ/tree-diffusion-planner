<h1 align="center">
<b>
Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree 🌳
</b>
</h1>

<div align="center">
  <b>
    MuJoCo locomotion environment
  </b>
</div>

This repository contains the complete code of TAT built upon the pre-trained [Diffuser](https://github.com/jannerm/diffuser) in MuJoCo locomotion environment.

## 🛠️ Installation
Follow these steps to set up the environment and install dependencies:
```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```
⚠️ Note: the environment setup matches the instructions provided in the [Diffuer's Main README](https://github.com/jannerm/diffuser/blob/main/README.md).

## 📥 Pretrained Model

Download pretrained diffusion models and value functions with:
```
./scripts/download_pretrained.sh
```
Then, you will obtain `logs/` in the root folder of the project directory.

## 🚀 Usage Guide
Evaluate `Diffuser` baseline:
```
python scripts/plan_guided.py --dataset hopper-medium-v2 --logbase logs/pretrained --use_tree False
```

Evaluate `TAT`:
```
python scripts/plan_guided.py --dataset hopper-medium-v2 --logbase logs/pretrained --use_tree True
```

Evaluate `TAT with vanilla warm-start`:
```
python scripts/plan_guided.py --dataset hopper-medium-v2 --logbase logs/pretrained --use_tree True --vanilla_warm_start True
```

Evaluate `TAT with TAT-reinforced warm-start`:
```
python scripts/plan_guided.py --dataset hopper-medium-v2 --logbase logs/pretrained --use_tree True --tree_warm_start True
```

## 📈 Viewing results

To view the experimental results, run:
```
python scripts/read_results.py
```

## 📝 Citation
```bibtex
@inproceedings{feng2024resisting,
  title={Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree},
  author={Feng, Lang and Gu, Pengjie and An, Bo and Pan, Gang},
  booktitle={International Conference on Machine Learning},
  pages={13175--13198},
  volume={235},
  year={2024},
  organization={PMLR},
}
```

## 🙏 Acknowledgements

This implementation is based on the [Diffuser repo](https://github.com/jannerm/diffuser).


