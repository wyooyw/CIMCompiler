
PREFIX=/home/wangyiou/opensource/llvm_new/llvm-project/build
BUILD_DIR=/home/wangyiou/opensource/llvm_new/llvm-project/build

start=$(date +%s)

mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS_RELEASE="-O0" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O0" \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_ENABLE_RTTI=ON
cmake --build . --target main

    # -DCMAKE_BUILD_TYPE=Debug

end=$(date +%s)
echo "Running time: $((end-start)) seconds"