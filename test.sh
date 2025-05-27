source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}

python3 test/op/test_conv.py
# pytest -n 4 test/compiler
# python3 test/op/llm/test_attn_decode.py
# cmcp compile -i test/compiler/memory/image/trans/code_.cim -o test_result -c test/compiler/config.json