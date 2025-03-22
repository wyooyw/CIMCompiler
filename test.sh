source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}

# cmcp compile -i ./test.cim -o .temp2 -c test/op/llm/config.json

export CIM_COMPILER_OUTPUT_DIR=result/seqlen8192_cp8
python test/op/llm/test_attn_decode.py

# pytest -n 4 test/op/llm/test_attn_decode.py
# python test/op/llm/test_attn_decode.py
# pytest test/op/llm/test_attn_decode.py
# pytest test/compiler/comm/test_send_recv.py
# python test/op/llm/softmax/test_softmax.py
# python test/op/llm/test_attn_decode.py
# pytest -n 4 test/op/llm/test_attn_decode.py
# pytest test/compiler
# pytest -n 4 test/compiler/memory/test_memory.py
# python test/compiler/comm/test_send_recv.py

# cmcp compile -i test.cim -o .temp -c test/op/llm/config.json

# pytest -n 4 test/
# pytest -n 4 test/compiler
# pytest test/compiler/conv
# pytest -n 4 test/op/test_conv.py
# python test/op/test_conv.py 
# pytest test/utils
# pytest test/data_processor