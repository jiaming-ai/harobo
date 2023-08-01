

## OVMM 

### Installation

Install home-robot by following instructions at [here](https://github.com/facebookresearch/home-robot)
For detic, it seems that the default installation works fine. (need to install detic requirements)

```bash
# first create a conda environment
mamba env create -p ./.venv -f harobo/environment.yml
conda activate ./.venv
# mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia


# install pytorch3d, build from source
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
# install vis tools for pytorch3d TODO: only required during debugging
pip install scikit-image matplotlib imageio plotly opencv-python

# install home-robot
cd home-robot
python -m pip install -e src/home_robot

# intall habitat-sim
mamba env update -f src/home_robot_sim/environment.yml 

# install habitat-lab and baseline
python -m pip install -e src/third_party/habitat-lab/habitat-lab
python -m pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install -e src/home_robot_sim

# install detectron by build from source
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# checkout code, note the --recurse-submodule flag
git clone --recurse-submodule https://github.com/jiaming-robot-learning/harobo.git

# install requirements for detic
cd ${DETIC_ROOT}
pip install -r harobo/perception/detection/Detic/requirements.txt



# install torch scatter
# Ensure that at least PyTorch 1.4.0 is installed and verify that cuda/bin and cuda/include are in your $PATH and $CPATH respectively
pip install torch-scatter

```

### Trobleshooting

## ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
We need to build pytorch3d from source
```bash
# first remove conda installed pytorch3d
mamba uninstall pytorch3d
# then install pytorch3d from source

# install necessary dependencies if they are removed when uninstalling pytorch3d
# we assume pytorch 1.13 is installed (tested with 1.13.1)
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath

# IMPORTANT: don't install the following package as indicated on the pytorch3d github page!
# conda install -c bottler nvidiacub

# install pytorch3d from source
# 0.7.2 is the version that works with pytorch 1.13.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
```


