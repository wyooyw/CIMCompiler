export PYTHONPATH=${PWD}
export FAST_MODE=0
export IM2COL_SMALL_INPUT_MEMORY=1
export OP_BASE_DIR=/home/wangyiou/project/cim_compiler_frontend/playground/op
# pytest test
# pytest test/compiler/pimcompute/test_pimcompute_dwconv.py
# pytest test/compiler/pimcompute/test_pimcompute_linear.py
pytest test/compiler/pimcompute/test_pimcompute_value_sparse.py
# python test/compiler/control_flow/for_loop/test_for_loop.py