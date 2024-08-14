<h1 align="center">
<b>
Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree 🌳
</b>
</h1>

<div align="center">
  <b>
    ICML 2024 (Spotlight)
  </b>
</div>

<p align="center">
| <a href="https://github.com/langfengQ/tree-diffusion-planner"><b>Website</b></a> | <a href="https://arxiv.org/abs/2405.17879"><b>Arxiv</b></a> | <a href="https://proceedings.mlr.press/v235/feng24b.html"><b>Paper</b></a> | <a href="https://icml.cc/media/PosterPDFs/ICML%202024/34197.png?t=1719634719.7916923"><b>Poster</b></a> |
</p>

This repository contains the complete code of TAT built upon the pre-trained [Diffuser](https://github.com/jannerm/diffuser). The codes are organized into three separate sub-branches for easy access:

- **Maze2D environment**: access the code directly [here](https://github.com/langfengQ/tree-diffusion-planner/tree/maze2d) (`maze2d` branch).
- **Kuka Block Stacking environment**: access the code directly [here](https://github.com/langfengQ/tree-diffusion-planner/tree/kuka) (`kuka` branch).
- **MuJoCo Locomotion environment**: access the code directly [here](https://github.com/langfengQ/tree-diffusion-planner/tree/locomotion) (`locomotion` branch).

⚠️ Note: the `master` branch does not contain any codes. Please refer to the `maze2d`, `kuka`, and `locomotion` branches for the full implementations.

## 🚀 Quick Start
To get started with this repository, follow these steps:
1. **Clone the repository** and verify the available branches:
```bash
git clone https://github.com/langfengQ/tree-diffusion-planner.git
cd tree-diffusion-planner/
git branch -a
# * master
#   remotes/origin/HEAD -> origin/master
#   remotes/origin/kuka
#   remotes/origin/locomotion
#   remotes/origin/master
#   remotes/origin/maze2d
```
The command `git branch -a` lists all the branches, and you should see the following branches: `master`, `maze2d`, `kuka`, and `locomotion`.

2. **Switch to a specific branch** to access the corresponding environment's code:

For the Maze2D environment:
```bash
git checkout maze2d
# Branch 'maze2d' set up to track remote branch 'maze2d' from 'origin'.
# Switched to a new branch 'maze2d'
```

For the Kuka Block Stacking environment:
```bash
git checkout kuka
# Branch 'kuka' set up to track remote branch 'kuka' from 'origin'.
# Switched to a new branch 'kuka'
```
For the MuJoCo Locomotion environment:
```bash
git checkout locomotion
# Branch 'locomotion' set up to track remote branch 'locomotion' from 'origin'.
# Switched to a new branch 'locomotion'
```

3. **Follow the branch-specific README**: Now you are ready to explore and experiment with TAT in your chosen environment! 🥳 Then, you can refer to that sub-branch's README file for instructions on setting up the environment, running experiments, and additional configurations.

## ❓ Issue
If you have any questions about the code, please feel free to open an issue!

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
