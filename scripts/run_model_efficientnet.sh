export FAST_MODE=1
export IM2COL_SMALL_INPUT_MEMORY=1
export CIM_COMPILER_BASE=${PWD}
export PYTHONPATH=${PWD}

model_name="EfficientNet"
model_path_dense="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_ori_data_0730"
model_path_bit_sparse="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_csd_th2_0803_data"
model_path_value_sparse="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_0.6_csd_th2_0616"
model_path_value_bit_sparse_0_6="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_0.6_csd_th2_0616"
model_path_value_bit_sparse_0_4="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_0.4_csd_th2_0615_data"
model_path_value_bit_sparse_0_2="${CIM_COMPILER_BASE}/models/efficientnet/EfficientNet_0.2_csd_th2_data_0704"
quantify=true

python engine/model_runner.py \
--model_name $model_name \
--model_path_dense $model_path_dense \
--model_path_bit_sparse $model_path_bit_sparse \
--model_path_value_sparse $model_path_value_sparse \
--model_path_value_bit_sparse_0_6 $model_path_value_bit_sparse_0_6 \
--model_path_value_bit_sparse_0_4 $model_path_value_bit_sparse_0_4 \
--model_path_value_bit_sparse_0_2 $model_path_value_bit_sparse_0_2 \
--quantify $quantify