# input_file="${PWD}/test/compiler/pimcompute/value_bit_sparse/value_bit_sparse_quantify/code.cim"
# input_file="${PWD}/test/compiler/pimcompute/value_sparse/value_sparse_group_longer_quantify/code.cim"
input_file="/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/control_flow/for_loop/accumulate_in_double_loop/code.cim"
output_path="${PWD}/.result"
config_path=${PWD}/config/config.json
bash compile.sh isa $input_file $output_path $config_path
# bash compile.sh ast $input_file $output_path