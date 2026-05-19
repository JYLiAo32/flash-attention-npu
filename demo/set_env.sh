export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/15.2.0/include/c++
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$CONDA_PREFIX/lib/gcc/aarch64-conda-linux-gnu/15.2.0/include/c++/aarch64-conda-linux-gnu
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib"
