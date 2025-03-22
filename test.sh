source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}

pytest -n 4 test/compiler
pytest -n 4 test/op/llm/test_attn_decode.py