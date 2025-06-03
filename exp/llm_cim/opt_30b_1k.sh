source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

python3 exp/llm_cim/main.py \
--n-head 56 \
--hidden-size 7168 \
--seqlen 1024 \
--mapping-cp-sizes 1 2 4 \
--world-size 32 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_30b_1k \
--split-stages