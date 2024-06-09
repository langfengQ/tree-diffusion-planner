# Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree

This branch contains the implementation of TAT in **Maze2D environment**.
<p align="center">
    <img src="./images/fig_traj_agg_tree.png" width="95%" title="Planning with TAT">
</p>

## Using pretrained models

Download maze2d pretrained models released by Janner et al. [here](https://www.dropbox.com/s/za14rwp8to1bosn/maze2d-logs.zip?e=2&dl=0).

## Plan using TAT:
Evaluate TAT on large & multi-task map via
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --multi_task True --use_tree True
```
or large & single-task map via
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --multi_task False --use_tree True
```
Other details are the same as the Diffuser Maze2d branch, see the Appendix.

## Citation
```bibtex
@article{feng2024resisting,
  title={Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree},
  author={Feng, Lang and Gu, Pengjie and An, Bo and Pan, Gang},
  journal={arXiv preprint arXiv:2405.17879},
  year={2024}
}
```

## Acknowledgements

This implementation is based on the [Diffuser repo](https://github.com/jannerm/diffuser).

## Appendix

This appendix is a copy from [Diffuser README in maze2d branch](https://github.com/jannerm/diffuser/blob/maze2d/README.md).

Training and visualizing of diffusion models from [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).
This branch has the Maze2D experiments and will be merged into main shortly.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing).


## Installation

```
conda env create -f environment.yml
conda activate diffusion
pip install -e .
```

## Usage

Train a diffusion model with:
```
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
```

The default hyperparameters are listed in [`config/maze2d.py`](config/maze2d.py).
You can override any of them with runtime flags, eg `--batch_size 64`.

Plan using the diffusion model with:
```
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```


## Docker

1. Build the container:
```
docker build -f azure/Dockerfile . -t diffuser
```

2. Test the container:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --logbase logs/docker"
```


## Running on Azure

#### Setup

1. Launching jobs on Azure requires one more python dependency:
```
pip install git+https://github.com/JannerM/doodad.git@janner
```

2. Tag the image built in [the previous section](#Docker) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`
