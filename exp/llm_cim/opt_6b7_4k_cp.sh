source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

python3 exp/llm_cim/main.py \
--n-head 32 \
--hidden-size 4096 \
--seqlen 4096 \
--mapping-cp-sizes 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 \
--world-size 32 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_6b7_4k_cp