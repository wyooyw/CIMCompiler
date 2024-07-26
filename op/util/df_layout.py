import numpy as np
from bitarray import bitarray
import math
import struct

def _float_to_bits(num:float, endian:str, width:int):
    assert type(endian)==str and (endian=="little" or endian=="big"), f"endian={str(endian)}"
    assert type(num)==float
    assert type(width)==int and width in [16,32,64]
    fmt_width = {16:"e",32:"f",64:"d"}[width]
    fmt_endian = {"big":">","little":"<"}[endian]
    fmt = f"{fmt_endian}{fmt_width}"
    num_bytes = struct.pack(fmt, num)
    bits = bitarray(endian=endian)
    bits.frombytes(num_bytes)
    return bits.tolist()

def _float16_to_bits(num:float, endian="big"):
    return _float_to_bits(num=num, endian=endian, width=16)

def _float32_to_bits(num:float,endian="big"):
    return _float_to_bits(num=num, endian=endian, width=32)

def _float64_to_bits(num:float,endian="big"):
    return _float_to_bits(num=num, endian=endian, width=64)


def _bits_to_float(bits_list:list, endian:str):
    assert type(endian)==str and (endian=="little" or endian=="big"), f"endian={str(endian)}"
    assert len(bits_list) in [16,32,64]
    bits = bitarray(bits_list,endian=endian)
    b = bits.tobytes()
    fmt_width = {16:"e",32:"f",64:"d"}[len(bits_list)]
    fmt_endian = {"big":">","little":"<"}[endian]
    fmt = f"{fmt_endian}{fmt_width}"
    num = struct.unpack(fmt, b)
    return num

def _bits_to_float16(bits_list:list, endian:str="big"):
    assert len(bits_list) == 16
    return _bits_to_float(bits_list, endian)

def _bits_to_float32(bits_list:list, endian:str="big"):
    assert len(bits_list) == 32
    return _bits_to_float(bits_list, endian)

def _bits_to_float64(bits_list:list, endian:str="big"):
    assert len(bits_list) == 64
    return _bits_to_float(bits_list, endian)

if __name__=="__main__":
    # print(_float64_to_bits(0.5))
    bits = _float64_to_bits(1.2345678)
    print(bits)
    print(_bits_to_float64(bits, "big"))
    # print(num)
    # print(bits[0])
    # print(bits[1:12])
    # print(bits[12:])
    # print(_float64_to_bits(-1.5))

def _int_to_bits(num:int, width:int, endian:str, signed:bool):
    assert type(width)==int and width > 0, f"width={str(width)}" 
    assert type(endian)==str and (endian=="little" or endian=="big"), f"endian={str(endian)}"
    assert type(signed)==bool, f"signed={str(signed)}"

    if signed:
        lower_bound = -(1<<(width-1)) 
        upper_bound = (1<<(width-1)) - 1
    else:
        lower_bound = 0
        upper_bound = (1<<width) - 1
    assert num>=lower_bound and num<=upper_bound,f"When width={width} and signed={signed}, num should in [{lower_bound},{upper_bound}], but got num={num}"
    
    bits = bitarray(endian=endian)
    bits.frombytes(int(num).to_bytes(math.ceil(width/8),byteorder=endian, signed=signed))
    if endian=="big":
        bits = bits[-width:]
    else:
        bits = bits[0:width]
    return bits.tolist()

def _uint32_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=32, signed=False)

def _uint16_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=16, signed=False)

def _uint8_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=8, signed=False)

def _int32_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=32, signed=True)

def _int16_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=16, signed=True)

def _int8_to_bits(num, endian="little"):
    return _int_to_bits(num=num, endian=endian, width=8, signed=True)



def _bits_to_int(bits_list, endian, signed):
    assert type(bits_list)==list and len(bits_list)%8==0, f"{type(bits_list)},{len(bits_list)}"
    bits = bitarray(bits_list,endian=endian)
    b = bits.tobytes()
    value = int.from_bytes(b,byteorder=endian, signed=True)
    return value

def _bits_to_int32(bits_list, endian="little"):
    assert len(bits_list)==32
    return _bits_to_int(bits_list, endian, signed=True)

def _bits_to_int16(bits_list, endian="little"):
    assert len(bits_list)==16
    return _bits_to_int(bits_list, endian, signed=True)

def _bits_to_int8(bits_list, endian="little"):
    assert len(bits_list)==8
    return _bits_to_int(bits_list, endian, signed=True)

def _bits_to_uint32(bits_list, endian="little"):
    assert len(bits_list)==32
    return _bits_to_int(bits_list, endian, signed=False)

def _bits_to_uint16(bits_list, endian="little"):
    assert len(bits_list)==16
    return _bits_to_int(bits_list, endian, signed=False)

def _bits_to_uint8(bits_list, endian="little"):
    assert len(bits_list)==8
    return _bits_to_int(bits_list, endian, signed=False)


def _tensor_int_to_bits(tensor, width:int, endian:str, signed:True):
    """ Turn int tensor to bits tensor
    """
    assert type(width)==int and width%8==0, f"width={str(width)}"
    assert type(endian)==str and (endian=="little" or endian=="big"), f"endian={str(endian)}"
    assert type(signed)==bool, f"signed={str(signed)}"
    assert type(tensor)==np.ndarray, f"type(tensor)={str(type(tensor))}"
    
    old_shape = tensor.shape
    tensor = tensor.reshape(-1)
    length = tensor.shape[0]
    new_tensor = np.zeros((length,width),dtype=np.uint8)
    for num in range(length):
        bits = _int_to_bits(tensor[num], endian=endian, width=width, signed=signed)
        new_tensor[num,:] = bits 
    new_tensor = new_tensor.reshape(*old_shape, width)
    return new_tensor

def tensor_uint32_to_bits(tensor, endian="little"):
    """ Turn uint32 tensor to bits tensor
    """
    assert tensor.dtype==np.uint32
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=32, signed=False)

def tensor_int32_to_bits(tensor, endian="little"):
    """ Turn uint32 tensor to bits tensor
    """
    assert tensor.dtype==np.int32
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=32, signed=True)


def tensor_uint16_to_bits(tensor, endian="little"):
    """ Turn uint16 tensor to bits tensor
    """
    assert tensor.dtype==np.uint16
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=16, signed=False)

def tensor_int16_to_bits(tensor, endian="little"):
    """ Turn uint16 tensor to bits tensor
    """
    assert tensor.dtype==np.int16
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=16, signed=True)

def tensor_uint8_to_bits(tensor, endian="little"):
    """ Turn uint8 tensor to bits tensor
    """
    assert tensor.dtype==np.uint8
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=8, signed=False)

def tensor_int8_to_bits(tensor, endian="little"):
    """ Turn int8 tensor to bits tensor
    """
    assert tensor.dtype==np.int8
    return _tensor_int_to_bits(tensor=tensor, endian=endian, width=8, signed=True)


def _tensor_bits_to_int(tensor, endian:str, signed:True, dtype:type):
    """ Turn bits tensor to int tensor
    """
    assert type(tensor)==np.ndarray, f"type(tensor)={str(type_tensor)}"
    assert tensor.shape[-1]%8==0, f"tensor.shape={tensor.shape}"
    assert type(endian)==str and endian=="little" or endian=="big", f"endian={str(endian)}"
    assert type(signed)==bool, f"signed={str(signed)}"
    assert dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32], f"dtype={dtype}"

    old_shape = tensor.shape[0:-1]
    tensor = tensor.reshape(-1, tensor.shape[-1])
    length = tensor.shape[0]
    new_tensor = np.zeros((length,),dtype=dtype)
    for num in range(length):
        new_tensor[num] = _bits_to_int(bits_list=tensor[num,:].tolist(),
                                        endian=endian,
                                        signed=signed)
    new_tensor = new_tensor.reshape(*old_shape)
    return new_tensor

def tensor_bits_to_uint32(tensor, endian="little"):
    """ Turn bits tensor to uint32 tensor
    """
    assert tensor.shape[-1]==32
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=False, dtype=np.uint32)

def tensor_bits_to_int32(tensor, endian="little"):
    """ Turn bits tensor to int32 tensor
    """
    assert tensor.shape[-1]==32
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=True, dtype=np.int32)

def tensor_bits_to_uint16(tensor, endian="little"):
    """ Turn bits tensor to uint16 tensor
    """
    assert tensor.shape[-1]==16
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=False, dtype=np.uint16)
    
def tensor_bits_to_int16(tensor, endian="little"):
    """ Turn bits tensor to int16 tensor
    """
    assert tensor.shape[-1]==16
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=True, dtype=np.int16)

def tensor_bits_to_uint8(tensor,endian="little"):
    """ Turn bits tensor to uint8 tensor
    """
    assert tensor.shape[-1]==8
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=False, dtype=np.uint8)

def tensor_bits_to_int8(tensor, endian="little"):
    """ Turn bits tensor to int8 tensor
    """
    assert tensor.shape[-1]==8
    return _tensor_bits_to_int(tensor=tensor, endian=endian, signed=True, dtype=np.int8)


def _loop_nest(kernel, bound):
    """ use recursion to do loop nest
    """
    depth = len(bound)
    new_bound = []
    for item in bound:
        if type(item)==int:
            new_bound.append((item,))
        else:
            new_bound.append(item)
    bound = new_bound
    index = [0] * len(bound)
    def loop(loop_depth):
        if loop_depth==depth:
            kernel(*index)
            return
        for i in range(*bound[loop_depth]):
            index[loop_depth] = i
            loop(loop_depth+1)
    loop(0)

def loop_nest_layout(tensor,bound,index):
    result = []
    def kernel(*index_tuple):
        index_tuple = tuple(index(*index_tuple))
        result.append(tensor.__getitem__(index_tuple).tolist())
    _loop_nest(kernel,bound=bound)
    result_np = np.array(result)
    return result_np

def im2col_layout(tensor,kernel_size,padding,stride):
    assert len(tensor.shape)==4
    assert kernel_size>0
    assert padding>=0
    assert stride>0
    tensor = np.pad(tensor,((0,0),(0,0),(padding,padding),(padding,padding)))
    batch,channel,height,width = tensor.shape
    format_feature = loop_nest_layout(tensor, 
        bound=[      batch, (0, height-kernel_size+1, stride), (0, width-kernel_size+1, stride), channel, kernel_size, kernel_size],
        index=lambda b    , filter_y                         , filter_x                        , c      , filter_h   , filter_w: 
            (b, c, filter_y+filter_h, filter_x+filter_w)
    )
    format_feature = format_feature.reshape(batch,-1,channel*kernel_size*kernel_size)
    format_feature = format_feature.astype(tensor.dtype)
    return format_feature


def padding_multiple_of(tensor, factor, axis):
    old_shape = tensor.shape
    assert axis < len(old_shape)
    pad_width = [(0,0)]*len(old_shape)
    old_len = old_shape[axis]
    new_len = int(math.ceil(old_len / factor)) * factor
    pad_width[axis] = (0,new_len-old_len)
    pad_width = tuple(pad_width)
    tensor = np.pad(tensor,pad_width)
    return tensor

def split_axis(tensor, factor, axis):
    old_shape = list(tensor.shape)
    axis_len = old_shape[axis]
    assert axis_len % factor == 0
    axis_len_1 = axis_len // factor
    axis_len_2 = factor
    old_shape[axis] = axis_len_1
    old_shape.insert(axis + 1, axis_len_2)
    return tensor.reshape(old_shape)

def safely_astype(tensor, dtype):
    assert dtype in [np.int8, np.int32, np.float32]
    width = {
        np.int8: 8, 
        np.int32: 32, 
        np.float32: 32
    }[dtype]
    signed = {
        np.int8: True,
        np.int32: True,
        np.float32: True
    }[dtype]

    if signed:
        lower_bound = -(1<<(width-1))
        upper_bound = (1<<(width-1)) - 1
    else:
        lower_bound = 0
        upper_bound = (1<<width) - 1

    assert tensor.max() <= upper_bound and tensor.min() >= lower_bound, f"tensor's value should in [{lowerbound,upperboud}], but got tensor.min()={tensor.min()}, tensor.max()={tensor.max()}"

    tensor_cast = tensor.astype(dtype)
    if not (tensor_cast==tensor).all():
        import ipdb
        ipdb.set_trace()

    return tensor_cast

if __name__=="__main__":
    a = np.arange(0,16).reshape(1,1,4,4)
    b = im2col_layout(a,kernel_size=3,padding=0,stride=1)
    print(a)
    print(b)