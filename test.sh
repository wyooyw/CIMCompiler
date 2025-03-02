source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
pytest -n 4 test/compiler

# pytest -n 4 test/
# pytest -n 4 test/compiler
# pytest test/compiler/conv
# pytest -n 4 test/op/test_conv.py 
# python test/op/test_conv.py 
# pytest test/utils
# pytest test/data_processor