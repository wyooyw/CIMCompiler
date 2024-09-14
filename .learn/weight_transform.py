import torch
import numpy as np
import random
from utils.df_layout import tensor_bits_to_int8
import ipdb
int_to_csd_nonzero_count_loopup_table = None
def int_to_csd(n, w=8):
    if n >= 0:
        bin_str = format(int(n), f'0{w}b')
    else:
        bin_str = format(-int(n), f'0{w}b')
    
    csd_list = list(bin_str)

    i = len(csd_list) - 1
    while i > 0:

        if csd_list[i] == '1' and csd_list[i - 1] == '1':
            csd_list[i] = '-'
            csd_list[i - 1] = '0'

            j = i - 2
            while j >= 0:
                if csd_list[j] == '0':
                    csd_list[j] = '1'
                    break
                else:
                    csd_list[j] = '0'
                    j -= 1
        i -= 1

    if n < 0:
        csd_list = ['-' if x == '1' else ('1' if x == '-' else '0') for x in csd_list]

    return ''.join(csd_list)

def int_to_csd_nonzero_count(n):
    csd_result = int_to_csd(n)
    count = csd_result.count('1') + csd_result.count('-')
    return count

def int_to_csd_nonzero_count_tensor(tensor):
    """
    Args:
        tensor: torch.Tensor, shape=[num_filter, -1] , 每一行是一个filter
    
    查找表:
        int_to_csd_nonzero_count_loopup_table: torch.Tensor, shape=[256,]
        int_to_csd_nonzero_count_loopup_table[i+128] = 表示数字i的csd编码中非零值的个数, -127<=i<=128
    """
    global int_to_csd_nonzero_count_loopup_table

    assert type(tensor)==np.ndarray
    assert tensor.dtype=="int8"
    tensor = torch.from_numpy(tensor)

    # 如果第一次进入这个函数,则创建查找表
    if int_to_csd_nonzero_count_loopup_table==None:
        int_to_csd_nonzero_count_loopup_table = torch.zeros((256,)).to(torch.int8).to(tensor.device)
        for i in range(-128,128):
            csd_result = int_to_csd(i)
            count = csd_result.count('1') + csd_result.count('-') #记录csd编码中非零值的个数   
            int_to_csd_nonzero_count_loopup_table[i+128] = count
    
    shape = tensor.shape
    tensor_flatten = tensor.to(torch.int32).reshape(-1) + 128

    # 从查找表中进行查找，并返回结果
    count = torch.index_select(int_to_csd_nonzero_count_loopup_table,0,tensor_flatten)
    count = count.reshape(shape)

    return count.numpy()

def parse_fold_weight(cim_cfg, op_cfg, weight_bit_num):
        """
        [1,2,1,2,1,1,2,2,2,2, 1,1,1,1,2,2,2,2,1,1,1,1,....]
        =>
        [
            [
                1,2,1,2,1,1,2,2,2,2, # one macro
                1,1,1,1,2,2,2,2,1,1,1,1, # one macro
            ], # one time step
            [
                .....
            ], # one time step
        ]
        """
        weight_bit_num = weight_bit_num.reshape(-1).tolist()

        assert type(weight_bit_num)==list and len(weight_bit_num)==op_cfg.out_channel, f"{len(weight_bit_num)}, {op_cfg.out_channel}"
        whole = []
        channel_idx = 0
        while channel_idx < op_cfg.out_channel:
            one_time_step = []
            max_bit_width = cim_cfg.n_macro * cim_cfg.bits_column
            # print("max_bit_width:",max_bit_width)
            bit_idx = 0
            while (channel_idx < op_cfg.out_channel) and (bit_idx+weight_bit_num[channel_idx] <= max_bit_width):
                one_time_step.append(weight_bit_num[channel_idx])
                bit_idx += weight_bit_num[channel_idx]
                channel_idx += 1
            # print(one_time_step)
            whole.append(one_time_step)
        return whole

def weight_transform(weight, cim_cfg, op_cfg, weight_dtype="OIHW"):
    assert len(weight.shape)==4
    assert type(weight)==np.ndarray
    if weight_dtype=="OIHW":
        out_channel,in_channel,ker_height,ker_width = weight.shape
        ele_in_filter = in_channel*ker_height*ker_width
        weight = np.transpose(weight, (2,3,1,0)) # HWIO
    elif weight_dtype=="HWIO":
        ker_height,ker_width,in_channel,out_channel = weight.shape
        ele_in_filter = in_channel*ker_height*ker_width
    else:
        assert False

    # Check threshold
    weight_bit_num = int_to_csd_nonzero_count_tensor(weight)
    weight_bit_num_per_out_channel = int_to_csd_nonzero_count_tensor(weight[0:1,0:1,0:1,:])
    # ipdb.set_trace()
    assert (weight_bit_num==weight_bit_num_per_out_channel).all(), "Element in one filter should have same threshold."
    
    fold = parse_fold_weight(cim_cfg, op_cfg, weight_bit_num_per_out_channel)

    filter_idx = 0
    # Elements in 'new_weight' and 'info' are 0 or 1. 
    # Datatype of these tensors is bit. 
    # Setting dtype='int8' just for save memory.
    new_weight = np.zeros((len(fold), ele_in_filter, cim_cfg.bits_column*cim_cfg.n_macro), dtype="int8")
    info = np.zeros((len(fold), ele_in_filter, cim_cfg.bits_column*cim_cfg.n_macro,3), dtype="int8")
    for time_id, one_time_filters in enumerate(fold):
        bit_idx = 0
        for filter_threshold in one_time_filters:
            if filter_threshold>0:
                assert filter_threshold in [1,2]
                filter_tensor = weight[:,:,:,filter_idx].reshape(-1)
                
                # value : [threshold, elem_in_filter,1], each value is 0 or 1
                # sign : [threshold, elem_in_filter,1], each value is 0 or 1
                # location : [threshold, elem_in_filter, 2], each value is in [0,1]
                value,sign,location = parse_tensor(filter_tensor, filter_threshold)

                # Put 'value' into 'new_weight',
                # Put 'sign' and 'location' into 'info'
                for bit in range(filter_threshold):
                    new_weight[time_id, :, bit_idx] = value[bit]
                    info[time_id, :, bit_idx,0] = sign[bit]
                    info[time_id, :, bit_idx,1:3] = location[bit]
                    bit_idx += 1
            
            filter_idx += 1

    # Turn 'new_weight' and 'info' to tensor of bytes.
    assert (cim_cfg.bits_column*cim_cfg.n_macro) % 8==0
    new_weight = new_weight.reshape(len(fold), ele_in_filter, cim_cfg.bits_column*cim_cfg.n_macro//8, 8)
    info = info.reshape(len(fold), ele_in_filter, cim_cfg.bits_column*cim_cfg.n_macro*3//8, 8)

    new_weight = tensor_bits_to_int8(new_weight)
    info = tensor_bits_to_int8(info)

    return new_weight, info, fold

def generate_valid_weight(shape):
    assert len(shape)==4
    out_channel,in_channel,ker_height,ker_width = shape
    num_in_filter = in_channel*ker_height*ker_width

    # random select 'out_channel' value as threshold
    # threshold_per_channel = np.random.randint(1,3,(out_channel,),"int8")
    # threshold_per_channel = np.ones((out_channel,),dtype="int8")+np.ones((out_channel,),dtype="int8")
    threshold_per_channel = np.ones((out_channel,),dtype="int8")

    # collect number for each threshold
    num_for_threshold = dict()
    for i in range(-127,128):
        count = int_to_csd_nonzero_count(i)
        if count not in num_for_threshold:
            num_for_threshold[count] = []
        num_for_threshold[count].append(i)
    
    # sample from each threshold to make filters
    weight = []
    for oc in range(out_channel):
        weight_filter = []
        threshold = threshold_per_channel[oc]
        sample_max = len(num_for_threshold[threshold]) - 1
        candidate = num_for_threshold[threshold]

        for i in range(num_in_filter):
            weight_filter.append(candidate[random.randint(0,sample_max)])

        weight.append(weight_filter)
    
    # pack to numpy array and return
    weight = np.array(weight,dtype="int8").reshape(*shape)
    return weight

def npint8(ls):
    return np.array(ls,dtype="int8")

location_lookup_table = [npint8([1,1]),npint8([1,0]),npint8([0,1]),npint8([0,0])]
sign_lookup_table = {"01":npint8([0]),"10":npint8([0]),"0-":npint8([1]),"-0":npint8([1])}
value_lookup_table = {"01":npint8([0]),"0-":npint8([0]),"10":npint8([1]),"-0":npint8([1])}
def parse(num):
    global location_lookup_table
    global sign_lookup_table
    global value_lookup_table
    csd = int_to_csd(num)
    result = []
    for i in range(0,len(csd),2):
        pair = csd[i:i+2]
        if pair=="00":
            continue
        value = value_lookup_table[pair]
        sign = sign_lookup_table[pair]
        location = location_lookup_table[i//2]
        result.append((value, sign, location))
    return result

def parse_tensor(tensor, threshold):
    # value = # [threshold, elem_in_filter,1], each value is 0 or 1
    # sign = # [threshold, elem_in_filter,1], each value is 0 or 1
    # location = # [threshold, elem_in_filter, 2], each value is in [0,1]
    assert len(tensor.shape)==1
    size = tensor.shape[0]
    
    value = np.zeros((threshold, size), dtype="int8")
    sign = np.zeros((threshold, size), dtype="int8")
    location = np.zeros((threshold, size, 2), dtype="int8")
    
    for n in range(tensor.shape[0]):
        results = parse(tensor[n])
        for j,item in enumerate(results):
            value[j, n] = item[0]
            sign[j, n] = item[1]
            location[j, n, 0:2] = item[2]
    
    return value, sign, location

def recover_csd(value, sign, location):
    assert type(value)==np.uint8
    assert value==1 or value==0
    assert type(sign)==np.uint8
    assert sign==1 or sign==0
    assert type(location)==np.ndarray
    assert len(location.shape)==1
    assert location.shape[0]==2
    assert location.dtype=="uint8",location.dtype
    assert location[0] == 0 or location[0] == 1
    assert location[1] == 0 or location[1] == 1
    
    lookup_table = {"00":"01","01":"0-","10":"10","11":"-0"}
    key = str(value) + str(sign)
    csd_value = lookup_table[key]

    location_lookup_table = {"00":3,"01":2,"10":1,"11":0}
    key = str(location[0]) + str(location[1])
    real_location = location_lookup_table[key]

    csd = ["00","00","00","00"]
    csd[real_location] = csd_value
    csd = "".join(csd)

    return csd

csd_to_int_lookup_table = None
def csd_to_int(csd):
    global csd_to_int_lookup_table
    if csd_to_int_lookup_table==None:
        csd_to_int_lookup_table = dict()
        for num in range(-128,129):
            _csd = int_to_csd(num, 8)
            csd_to_int_lookup_table[_csd] = num
    
    return csd_to_int_lookup_table[csd]


def parse_out_mask(fold_weight_info):
    flatten_fold_weight_info = fold_weight_info
    out_mask = []
    for bw in flatten_fold_weight_info:
        if bw==1:
            out_mask.append(0)
        elif bw==2:
            out_mask.append(1)
            out_mask.append(0)
        elif bw==0:
            pass
        else:
            assert False
    return out_mask

def parse_col_to_filter(fold_weight_info):
    col_to_filter = []
    for i,cols in enumerate(fold_weight_info):
        for col in range(cols):
            col_to_filter.append(i)
    return col_to_filter

if __name__=="__main__":
    for i in range(-128,129):
        # csd_8 = int_to_csd(i)
        # csd_16 = int_to_csd(i,16)
        # assert csd_16[8:]==csd_8
        print(f"{i}:",int_to_csd(i,8))
    # result = parse(15)
    # for i in result:
    #     print("value:",i[0]," sign:",i[1]," location:",i[2])
    # weight = generate_valid_weight([1,1,2,2])
    # print("weight:",weight)
    # value, sign, location = parse_tensor(weight[0].reshape(-1), int_to_csd_nonzero_count(weight[0,0,0,0]))
    # print("value:")
    # print(value)
    # print("sign:")
    # print(sign)
    # print("location:")
    # print(location)
    
    # weight = generate_valid_weight([256,128,3,3])
    # bsp_conv2d_cfg = BitSparseConv2dConfig(in_name="I",weight_name="W",out_name="O",\
    #                 in_dtype="int8",out_dtype="int32",\
    #                 in_height=4,in_width=4,in_channel=128,\
    #                 out_channel=256,ker_height=3,ker_width=3,\
    #                 input_layout="NHWC",kernel_layout="HWIO",output_layout="NHWC",\
    #                 padding=(0,0,0,0),stride=(1,1))
    # cim_cfg = CimConfig(n_macro=2,n_compartment=4,n_row=32,bytes_column=1,local_memory_size=131072,pim_buffer_size=4096)
    # new_weight, info, fold = weigth_transform(weight, cim_cfg, bsp_conv2d_cfg)
    # print(new_weight.shape)
    # print(info.shape)
    # print(fold)

    
    # print(new_weight.shape)
    
    