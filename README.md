# python-image-processing
 # NOTE chay scrfd cua insightFace
ha version pytorch

link https://pytorch.org/get-started/previous-versions/

`` CUDA_VISIBLE_DEVICES=0 python -u tools/test_widerface.py ./configs/scrfd/scrfd_2.5g.py ./work_dirs/scrfd_2.5g/model.pth --mode 0 --out wouts ``

# install libtorch thirdparty

```
#!/bin/bash

sudo apt-get install libgoogle-glog-dev libgflags-dev libjsoncpp-dev libeigen3-dev nvidia-cuda-toolkit

wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

mv libtorch introspective_ORB_SLAM/Thirdparty/libtorch
```
