input_file="${PWD}/op/v1/conv2d_dense.cim"
output_path="${PWD}/.result"
bash compile.sh isa $input_file $output_path
# bash compile.sh ast $input_file $output_path