

## OVMM 

### Installation

Install home-robot by following instructions at [here](https://github.com/facebookresearch/home-robot)
For detic, it seems that the default installation works fine. (need to install detic requirements)

```bash
# note the --recurse-submodule flag
git clone --recurse-submodule https://github.com/jiaming-robot-learning/harobo.git
```

### Trobleshooting

## ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
We need to build pytorch3d from source
```bash
# first remove conda installed pytorch3d
conda uninstall pytorch3d
# then install pytorch3d from source

# install necessary dependencies if they are removed when uninstalling pytorch3d
# we assume pytorch 1.13 is installed (tested with 1.13.1)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

# IMPORTANT: don't install the following package as indicated on the pytorch3d github page!
# conda install -c bottler nvidiacub

# install pytorch3d from source
# 0.7.2 is the version that works with pytorch 1.13.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
```