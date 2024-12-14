export FAST_MODE=1
export IM2COL_SMALL_INPUT_MEMORY=1
export CIM_COMPILER_BASE=${PWD}
export PYTHONPATH=${PWD}

model_name="MobileNetV2"
model_path_dense="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet-ori-data-0801"
model_path_bit_sparse="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet_csd_th2_data_0801"
model_path_value_sparse="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet_0.6_csd_th2_data_0518"
model_path_value_bit_sparse_0_6="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet_0.6_csd_th2_data_0518"
model_path_value_bit_sparse_0_4="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet_0.4_csd_th2_data_0518"
model_path_value_bit_sparse_0_2="${CIM_COMPILER_BASE}/models/mobilenet/MobileNet_0.2_csd_th2_data_0516"
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
