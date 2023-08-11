

## OVMM 

### Installation

Install home-robot by following instructions at [here](https://github.com/facebookresearch/home-robot) for the latest version. Or follow the instructions below for the version we used.

For detic, it seems that the default installation works fine. (need to install detic requirements)

#### 1. Create Your Environment

```
mamba env create -n home-robot -f src/home_robot/environment.yml

conda activate home-robot
```

This should install pytorch; if you run into trouble, you may need to edit the installation to make sure you have the right CUDA version. See the [pytorch install notes](docs/install_pytorch.md) for more.



#### 2. Install Home Robot Packages
```
conda activate home-robot

# Install the core home_robot package
python -m pip install -e src/home_robot

```

#### 3. Download third-party packages
```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet
```

#### 5. Install Detic

Install [detectron2](https://detectron2.readthedocs.io/tutorials/install.html). 
```
python -m pip install -e detectron2
```

If you installed our default environment above, you may need to [download CUDA11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive).


Download Detic checkpoint as per the instructions [on the Detic github page](https://github.com/facebookresearch/Detic):

```bash
cd $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/detic/Detic/
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate
```

You should be able to run the Detic demo script as per the Detic instructions to verify your installation was correct:
```bash
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

#### 6. Download pretrained skills
```
mkdir -p data/checkpoints
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip
unzip ovmm_baseline_home_robot_challenge_2023.zip
cd ../../
```

# first checkout harobo and home-robot
```bash
git clone https://github.com/facebookresearch/home-robot.git
cd home-robot
git reset --hard 5e9fdc7b2e88899061eecf03ec68f35607f772d5
cd ..

git clone https://github.com/jiaming-robot-learning/harobo.git
```

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
cd .. # go to the workspace directory
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


