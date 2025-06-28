source log_config.sh

export PYTHONPATH=${PWD}
export CIM_COMPILER_BASE=${PWD}
export LD_LIBRARY_PATH=${PWD}/thirdparty/glog/build:$LD_LIBRARY_PATH

python3 exp/llm_cim/prefill.py \
--hidden-size 256 \
--seqlen 1024 \
--world-size 2 \
--config-path ${PWD}/test/op/llm/config.json \
--name-prefix opt_6b7_1k_prefill