set -e

input_file=$2
output_path=$3
config_path=$4
mkdir -p $output_path

# copy source file
cp $input_file $output_path/origin_code.cim

# precompile
python precompile/remove_comment.py --in-file $output_path/origin_code.cim --out-file $output_path/precompile_1_remove_comment.cim
python precompile/macro_replace.py --in-file $output_path/precompile_1_remove_comment.cim --out-file $output_path/precompile_2_replace_macro.cim
cp $output_path/precompile_2_replace_macro.cim $output_path/precompile.cim

ANTLR_HOME=${PWD}/antlr

# antlr: code -> ast(json)
mkdir .temp
java -cp $ANTLR_HOME/antlr-4.7.1-complete.jar org.antlr.v4.Tool CIM.g -o .temp
echo "ANTLR:Generate done!"
cp $ANTLR_HOME/AntlrToJson.java .temp/
javac -cp "$ANTLR_HOME/*" .temp/CIM*.java .temp/AntlrToJson.java
echo "ANTLR:Compile done!"
cd .temp
java -cp .:$ANTLR_HOME/antlr-4.7.1-complete.jar:$ANTLR_HOME/gson-2.11.0.jar AntlrToJson $output_path/precompile.cim $output_path/ast.json

cd ..
echo "ANTLR Down."

# ast(json) -> mlir
if [ "$1" == "isa" ]; then
    ./build/bin/main $output_path/ast.json $output_path $config_path
fi