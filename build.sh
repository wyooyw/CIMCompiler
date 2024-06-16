PREFIX=/home/wangyiou/opensource/llvm-project/build
BUILD_DIR=/home/wangyiou/opensource/llvm-project/build

mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_CCACHE_BUILD=ON
cmake --build . --target main