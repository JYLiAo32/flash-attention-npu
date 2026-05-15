#!/usr/bin/bash
set -eo pipefail

ENV_NAME="fa_demo2"


source /root/miniconda3/envs/myenv2/Ascend/cann/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh


echo "========== Create Conda Env =========="

if conda env list | grep -q "^${ENV_NAME} "; then
    conda remove -n ${ENV_NAME} --all -y
fi

conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME} 

echo "========== Install Python Packages =========="

pip cache purge
pip install wheel setuptools pyyaml "numpy<2.0.0"  
pip install torch==2.1.0 
pip install torch-npu==2.1.0.post17
pip install pytest
python -m pip install --upgrade setuptools wheel pip
conda install -y gxx_linux-aarch64

echo "========== Setup Compiler Env =========="
GCC_INCLUDE_BASE=$(dirname $(find $CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu -name iostream | head -n 1))
export CPLUS_INCLUDE_PATH=${GCC_INCLUDE_BASE}
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${GCC_INCLUDE_BASE}/aarch64-conda-linux-gnu

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib"

echo "========== Verify =========="

python -c "
import torch
import torch_npu

print('torch      :', torch.__version__)
print('torch_npu  :', torch_npu.__version__)
print('npu avail  :', torch.npu.is_available())
"

echo "========== DONE =========="
echo
echo "Run the following command to activate env:"
echo
echo "    conda activate ${ENV_NAME}"
echo