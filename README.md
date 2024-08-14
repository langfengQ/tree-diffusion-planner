<h1 align="center">
<b>
Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree 🌳
</b>
</h1>

<div align="center">
  <b>
    Kuka environment
  </b>
</div>

This repository contains the complete code of TAT built upon the pre-trained [Diffuser](https://github.com/jannerm/diffuser) in Kuka environment.

## 🛠️ Installation
⚠️ Note: the environment setup matches the instructions provided in the [Diffuer's Kuka README](https://github.com/jannerm/diffuser/blob/kuka/README.md).

## 📥 Pretrained Model

Download kuka pretrained models released by Janner et al. [here](https://www.dropbox.com/s/zofqvtkwpmp4v44/metainfo.tar.gz?dl=0).
After downloading, place the `metainfo.tar.gz` file in the root folder of the project directory: `tree-diffusion-planner/metainfo.tar.gz`.
```bash
tar -xzf metainfo.tar.gz
```
Then, you will obtain `kuka_dataset/` and `logs/` in the root folder.

## 🚀 Usage Guide
Evaluate `TAT` on unconditional stacking via
```
python scripts/unconditional_kuka_planning_eval.py --use_tree
```
or conditional stacking via
```
python scripts/conditional_kuka_planning_eval.py --use_tree
```
or rearrangement stacking via
```
python scripts/rearrangment_kuka_planning_eval.py --use_tree
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


