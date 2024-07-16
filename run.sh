set -e

input_file="${PWD}/op/v1/conv2d_dense.cim"
output_path="${PWD}/result"
mkdir -p $output_path
# copy source file
cp $input_file $output_path/origin_code.cim

# precompile
python precompile/remove_comment.py --in-file $output_path/origin_code.cim --out-file $output_path/precompile_1_remove_comment.cim
python precompile/macro_replace.py --in-file $output_path/precompile_1_remove_comment.cim --out-file $output_path/precompile_2_replace_macro.cim
cp $output_path/precompile_2_replace_macro.cim $output_path/precompile.cim

# antlr: code -> ast(json)
antlr CIM.g -o .temp
echo "ANTLR:Generate done!"
cp antlr_to_json/Examples.java .temp/
antlr-compile .temp/CIM*.java .temp/Examples.java
echo "ANTLR:Compile done!"
cd .temp
antlr-grun-json Examples $output_path/precompile.cim $output_path/ast.json
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
cd ..
echo "ANTLR Down."

# ast(json) -> mlir
if [ $# -lt 1 ]; then
    ./build/bin/main $output_path/ast.json $output_path/final_code.json
fi