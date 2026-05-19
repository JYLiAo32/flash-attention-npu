###############
# 为了避免麻烦，仓库上已经提供了whl包和.pt文件（真值数据）
# 拉取本项目的代码后，建议reset到上一个提交，将softcap演示相关的代码处于改动状态，方便在IDE上体现“改动”
git reset --mixed HEAD~1


############# 配置镜像源（可选）
conda config --add channels https://mirrors.huaweicloud.com/repository/conda/pkgs/main
conda config --add channels https://mirrors.huaweicloud.com/repository/conda/pkgs/free
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host mirrors.huaweicloud.com
############# 构建环境
conda create -n demo python=3.10 -y
conda activate demo 

pip install wheel setuptools pyyaml "numpy<2.0.0"  
pip install torch==2.1.0 # 目前仅release适配2.1.0版本torch的whl包
pip install torch-npu==2.1.0.post17

# 避免 No module named 'pkg_resources' 的错误
python -m pip install --upgrade setuptools wheel pip

# 安装工具包，配置环境变量
conda install  gxx_linux-aarch64 -y
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/15.2.0/include/c++
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/15.2.0/include/c++/aarch64-conda-linux-gnu
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib"

############# PyPI安装
pip install flash_attn_npu --no-build-isolation

############# 编译
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_BUILD_VERSION=v2 python setup.py bdist_wheel

############# 重新安装
# NOTE: 建议预先编译出whl后将其移出dist/目录，以免上一步编译时覆盖它
# 这里默认将whl包移到项目根目录下
pip install --force-reinstall --no-deps flash_attn_npu-0.1.1-cp310-cp310-linux_aarch64.whl
# pip install --force-reinstall --no-deps dist/flash_attn_npu-0.1.1-cp310-cp310-linux_aarch64.whl


############# 预先执行构建真值数据
python demo/demo_ref.py
# 取消line198~199的注释行，再次执行，生成带softcap的真值数据
python demo/demo_ref.py

############# 执行demo
# part 1
python demo/llama_demo.py

# part2: 修改所用真值数据，启用softcap测试
# 修改line79，80行注释
python demo/llama_demo.py

# 取消line 88的注释，在FA算子中开启softcap功能
python demo/llama_demo.py


