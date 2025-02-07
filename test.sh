source log_config.sh

export PYTHONPATH=${PWD}/src:${PWD}
export FAST_MODE=0
export IM2COL_SMALL_INPUT_MEMORY=1
export CIM_COMPILER_BASE=${PWD}
# pytest test
# pytest test/compiler/pimcompute/test_pimcompute_dwconv.py
# pytest test/compiler/pimcompute/test_pimcompute_linear.py
# pytest test/compiler/pimcompute/test_pimcompute_value_sparse.py
# pytest -v test/compiler/pimcompute/test_pimcompute_resmul_quantize.py
# pytest test/compiler/pimcompute/test_pimcompute_resadd_quantize.py
# pytest test/compiler/pimcompute/test_pimcompute_max_pooling.py
# pytest test/compiler/pimcompute/test_pimcompute_depthwise_simd.py
# pytest test/compiler/pimcompute/test_pimcompute_dwconv.py
# pytest test/op/test_pooling.py
# pytest test/op/test_linear.py
# pytest test/op/test_resadd_quantize.py
# pytest test/inst
# pytest test/compiler/arith/test_arith.py
# pytest -v test/compiler/control_flow/for_loop/test_for_loop.py
# pytest test/compiler/simd/test_simd.py
# pytest test/op/test_dwconv_cim.py
# python test/data_processor/test_value_sparse_data_processor_new.py
# pytest test/op/test_conv.py
# pytest test/
# exit
# python test/compiler/control_flow/for_loop/test_for_loop.py
# export FAST_MODE=1
# pytest test/compiler/pimcompute/test_pimcompute_value_sparse.py
# pytest test/compiler/pimcompute/test_pimcompute_depthwise_simd.py

# python src/python/cli/cim_compiler.py convert \
# --src-type legacy \
# --dst-type asm \
# --src-file test/inst/case1/legacy \
# --dst-file ./asm

# pytest test/inst

# python cli/cim_compiler.py convert \
# --src-type asm \
# --dst-type legacy \
# --src-file ./asm \
# --dst-file ./legacy

# python src/python/cli/cim_compiler.py compile \
# -i test/compiler/control_flow/if/if/code.cim \
# -o ./temp \
# -c test/compiler/config.json

# python src/python/cli/cim_compiler.py simulate \
# -i /home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-12-14/AlexNet/bit_sparse/0_conv/final_code.json \
# -d /home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-12-14/AlexNet/bit_sparse/0_conv/global_image \
# -o temp/output \
# -c config/config.json \
# --code-format legacy \
# --save-unrolled-code \
# --save-stats

# input_file=$(pwd)/.result/origin_code.cim
# output_path=$(pwd)/.result/test
# config_path=$(pwd)/config/config.json
# bash compile.sh isa $input_file $output_path $config_path

# pytest test/inst/test_inst_parser_dumper.py
# pytest test/compiler
# pytest test/op/test_conv.py
pytest test/utils