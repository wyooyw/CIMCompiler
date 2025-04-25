export FAST_MODE=1
export IM2COL_SMALL_INPUT_MEMORY=1
export CIM_COMPILER_BASE=${PWD}
export PYTHONPATH=${PWD}

model_name="AlexNet"
model_path_dense="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_ori_data_0525"
model_path_bit_sparse="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_csd_th2_data_0729"
model_path_value_sparse="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_0.6_csd_th2_data_0522"
model_path_value_bit_sparse_0_6="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_0.6_csd_th2_data_0522"
model_path_value_bit_sparse_0_4="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_0.4_csd_th2_data_0522"
model_path_value_bit_sparse_0_2="${CIM_COMPILER_BASE}/models/alexnet/AlexNet_0.2_csd_th2_data_0709"
quantify="true"

python cim_compiler/engine/model_runner.py \
--model_name $model_name \
--quantify $quantify \
--model_path_value_sparse $model_path_value_sparse \
--model_path_value_bit_sparse_0_6 $model_path_value_bit_sparse_0_6 \

# --model_path_bit_sparse $model_path_bit_sparse \
# python engine/model_runner.py \
# --model_name $model_name \
# --model_path_dense $model_path_dense \
# --model_path_bit_sparse $model_path_bit_sparse \
# --model_path_value_sparse $model_path_value_sparse \
# --model_path_value_bit_sparse_0_6 $model_path_value_bit_sparse_0_6 \
# --model_path_value_bit_sparse_0_4 $model_path_value_bit_sparse_0_4 \
# --model_path_value_bit_sparse_0_2 $model_path_value_bit_sparse_0_2 \
# --quantify $quantify
