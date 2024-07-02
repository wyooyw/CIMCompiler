antlr CIM.g -o .temp
echo "Generate done!"
cp antlr_to_json/Examples.java .temp/
antlr-compile .temp/CIM*.java .temp/Examples.java
echo "Compile done!"
cd .temp
antlr-grun-json Examples
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
# antlr-grun CIM program -tree ../op/v1/conv2d_dense.cim
# echo "Run done!"