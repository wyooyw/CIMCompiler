input_file="${PWD}/test/compiler/memory/image/trans/code.cim"
output_path="${PWD}/.result"
config_path=${PWD}/config/config.json
bash compile.sh isa $input_file $output_path $config_path
# bash compile.sh ast $input_file $output_path