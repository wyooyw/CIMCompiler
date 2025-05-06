source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

python3 exp/llm_cim/main.py \
--n-head 32 \
--hidden-size 4096 \
--seqlen 1024 \
--mapping-cp-sizes 1 \
--world-size 32 \
--split-stages \
--config-path ${PWD}/test/op/llm/config.json