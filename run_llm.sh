export ATTN_HEAD_HIDDEN=128
export ATTN_SEQLEN=1024
export ATTN_WORLD_SIZE=32
export ATTN_CP_GROUP_SIZE=1
export CHECK_RESULT=0
NAME=hdim${ATTN_HEAD_HIDDEN}_seq${ATTN_SEQLEN}_cp${ATTN_CP_GROUP_SIZE}
export CIM_COMPILER_OUTPUT_DIR=./result/${NAME}
python test/op/llm/test_attn_decode.py
python gather_llm_code.py -i ./result/${NAME} -n 31 -o ./result/${NAME}/all_codes.json