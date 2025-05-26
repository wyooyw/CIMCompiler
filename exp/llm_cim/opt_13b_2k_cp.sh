source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

# n_head = 40 = 16 + 16 + 8

python3 exp/llm_cim/main.py \
--n-head 1 \
--hidden-size 128 \
--seqlen 2048 \
--mapping-cp-sizes 16 \
--world-size 16 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_13b_2k_cp_onehead
