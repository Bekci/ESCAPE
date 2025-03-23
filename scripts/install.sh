#/usr/bin/env sh
sudo apt-get update -y
sudo apt-get install -y ninja-build
pip install argparse
pip install easydict
pip install h5py
pip install matplotlib
pip install numpy
pip install open3d
pip install opencv-python
pip install pyyaml
pip install scipy
pip install tensorboardX
pip install timm
pip install tqdm
pip install transforms3d
pip install einops
pip install gdown
pip install Ninja
pip install ipdb
pip install seaborn
pip install plyfile
pip install trimesh
# GPU kNN
pip install --upgrade "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"

# PointNet++
pip install --verbose --upgrade "git+https://github.com/Bekci/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"


python setup.py install --user

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'