source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

# 6.7B: n_head = 32 = 32
# 13B: n_head = 40 = 32 + 8
# 30B: n_head = 56 = 32 + 16 + 8
# 66B: n_head = 72 = 32 + 32 + 8

python3 exp/llm_cim/main.py \
--n-head 32 \
--hidden-size 4096 \
--seqlen 2048 \
--mapping-cp-sizes 1 \
--world-size 32 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_2k_32head

python3 exp/llm_cim/main.py \
--n-head 16 \
--hidden-size 2048 \
--seqlen 2048 \
--mapping-cp-sizes 2 \
--world-size 32 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_2k_16head

python3 exp/llm_cim/main.py \
--n-head 8 \
--hidden-size 1024 \
--seqlen 2048 \
--mapping-cp-sizes 4 \
--world-size 32 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_2k_8head