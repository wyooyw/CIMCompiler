set -e

# copy source file
cp op/v1/conv2d_dense.cim ./result/conv2d_dense.cim

# precompile
python precompile/remove_comment.py --in-file ./result/conv2d_dense.cim --out-file ./result/conv2d_dense_precompile.cim

# antlr: code -> ast(json)
antlr CIM.g -o .temp
echo "ANTLR:Generate done!"
cp antlr_to_json/Examples.java .temp/
antlr-compile .temp/CIM*.java .temp/Examples.java
echo "ANTLR:Compile done!"
cd .temp
antlr-grun-json Examples
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
cd ..
echo "ANTLR Down."

# ast(json) -> mlir
./build/bin/main