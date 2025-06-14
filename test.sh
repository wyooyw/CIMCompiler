source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}

pytest -n 4 test/compiler
pytest -n 4 test/polycim/end2end/test_cimflow_op.py
pytest test/polycim/end2end/test_cimflow_network.py
pytest test/op/llm/test_attn_decode.py