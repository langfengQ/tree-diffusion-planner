<h1 align="center">
<b>
Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree 🌳
</b>
</h1>

<div align="center">
  <b>
    Maze2D environment
  </b>
</div>

This repository contains the complete code of TAT built upon the pre-trained [Diffuser](https://github.com/jannerm/diffuser) in Maze2D environment.

## 🛠️ Installation
Follow these steps to set up the environment and install dependencies:
```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```
⚠️ Note: the environment setup matches the instructions provided in the [Diffuer's Maz2d README](https://github.com/jannerm/diffuser/blob/maze2d/README.md).

## 📥 Pretrained Model

Download maze2d pretrained models released by Janner et al. [here](https://www.dropbox.com/s/za14rwp8to1bosn/maze2d-logs.zip?e=2&dl=0).
After downloading, place the `maze2d-logs.zip` file in the main folder of the project directory: `tree-diffusion-planner/maze2d-logs.zip`.
```bash
unzip maze2d-logs.zip
# Archive:  maze2d-logs.zip
#    creating: logs/
#   ...
#   inflating: logs/maze2d-umaze-v1/diffusion/H128_T64/trainer_config.pkl  
#   inflating: logs/maze2d-umaze-v1/diffusion/H128_T64/model_config.pkl
```

## 🚀 Usage Guide
Run `Diffuser` baseline on large & multi-task map via
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --multi_task True --use_tree False
```

Evaluate `TAT` on large & multi-task map via
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --multi_task True --use_tree True
```
Evaluate `TAT` on large & single-task map via
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --multi_task False --use_tree True
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


