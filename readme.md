
## Environment

Build [llvm](https://github.com/llvm/llvm-project). 

The commit-id I use is 0977504537b4dd945fd91fe11eb1a3165297e64a

The script I use to build llvm is:

```
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_C_FLAGS_RELEASE="-O3" \
   -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
   -DLLVM_ENABLE_PROJECTS="mlir;lld"\
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_USE_SPLIT_DWARF=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_ENABLE_RTTI=ON
cmake --build . --target check-mlir -j8

```

## Quick Start

### Build the compiler

```
bash build.sh
```

### Run the test

```
bash test.sh
```

### Use the compiler

#### Compile .cim file

```
bash run.sh
```

#### Compile Neural Network

```
bash run_model.sh
```

Model files are not pushed yet.

## Operator Template

Operator templates are written in CIM DSL. Here are the convolution operator templates:

- `Dense Conv2d`: op/dense/dense_conv2d_group/code_template.cim
- `Value Sparse Conv2d`: op/value_sparse/value_sparse_group_longer/code_template.cim
- `Bit Sparse Conv2d`: op/bit_sparse/bit_sparse_conv2d_group/code_template.cim
- `Value&Bit Sparse Conv2d`: op/value_bit_sparse/value_bit_sparse_base/code_template.cim
