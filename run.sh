# input_file="${PWD}/test/compiler/pimcompute/value_bit_sparse/value_bit_sparse_quantify/code.cim"
# input_file="${PWD}/test/compiler/pimcompute/value_sparse/value_sparse_group_longer_quantify/code.cim"
input_file="${PWD}/test/compiler/pimcompute/value_sparse/value_sparse_group_longer_quantify/code.cim"
output_path="${PWD}/.result"
config_path=${PWD}/config/config.json
bash compile.sh isa $input_file $output_path $config_path
# bash compile.sh ast $input_file $output_path