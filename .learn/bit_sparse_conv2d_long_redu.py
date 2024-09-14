
import torch
from op.bit_sparse.weight_transform import weight_transform, generate_valid_weight, recover_csd, csd_to_int
import random
from config.cim_config import CimConfig
from config.datatype import DATA_TYPE_BYTES
import numpy as np
import ipdb
import json
from functools import reduce
import pickle
from utils.df_layout import tensor_int8_to_bits
from op.bit_sparse.bit_sparse_conv2d_strategy_base import BitSparseConv2dStrategyBase, BitSparseConv2dConfig
from op.code_emitter import Tensor
from op.bit_sparse.weight_transform import parse_out_mask,parse_col_to_filter
from utils.check import Checker
from op.code_emitter import CodeEmitter
from utils.register import register_op_strategy
from utils.ceil import ceil

class BitSparseConv2dLongReduceStrategy(BitSparseConv2dStrategyBase):
    def __init__(self):
        super().__init__()

    def check_config(self, cim_cfg, op_cfg):
        checker = Checker()

        checker.check(op_cfg.is_depthwise==False, f"BitSparseConv2dLongReduceStrategy not support dw-conv, but got op_cfg.is_depthwise={op_cfg.is_depthwise}")

        # Align check
        checker.check(op_cfg.in_channel >= cim_cfg.n_compartment, f"Expect op_cfg.in_channel > cim_cfg.n_compartment, but got op_cfg.in_channel={op_cfg.in_channel}, cim_cfg.n_compartment={cim_cfg.n_compartment}")
        checker.check(op_cfg.in_channel % cim_cfg.n_compartment == 0, f"Expect op_cfg.in_channel % cim_cfg.n_compartment==0, but got op_cfg.in_channel={op_cfg.in_channel} , cim_cfg.n_compartment={cim_cfg.n_compartment}")

        # Row capacity check
        total_row = (op_cfg.in_channel // cim_cfg.n_compartment) * op_cfg.ker_height * op_cfg.ker_width
        checker.check(total_row > cim_cfg.n_row, f"Expect total_row > cim_cfg.n_row, but got total_row={total_row} cim_cfg.n_row={cim_cfg.n_row}")
        checker.check(self.decide_tile_in_channel(cim_cfg,op_cfg)>0,"cannot find a tile_in_channel")
        
        # Pim buffer capacity check
        input_pim_buffer = cim_cfg.n_compartment * op_cfg.in_bytes
        output_pim_buffer = cim_cfg.bits_column * cim_cfg.n_macro * op_cfg.out_bytes
        use_pim_buffer = input_pim_buffer + output_pim_buffer
        checker.check(use_pim_buffer <= cim_cfg.pim_buffer_size, f"use_pim_buffer <= cim_cfg.pim_buffer_size, but got use_pim_buffer={use_pim_buffer}, cim_cfg.pim_buffer_size={cim_cfg.pim_buffer_size}")

        return checker.check_pass()

    def decide_tile_in_channel(self,cim_cfg,op_cfg):
        tile_in_channel = op_cfg.in_channel
        while True:
            use_row = (tile_in_channel // cim_cfg.n_compartment) * op_cfg.ker_width * op_cfg.ker_height
            if use_row <= cim_cfg.n_row:
                break
            if not tile_in_channel%cim_cfg.n_compartment==0:
                return -1
            tile_in_channel //= 2
            if not tile_in_channel%cim_cfg.n_compartment==0:
                return -1
        print("tile_in_channel:",tile_in_channel)
        return tile_in_channel
        
    def get_low_level_code(self, cim_cfg, op_cfg, param_cfg, input_on_local=True, output_on_local=True):
        """
        No tiling when load input from global memory to local memory.

        new_weight: [timestep, num_ele_in_filter, n_macro*bytes_column]
        """
        Tensor.clear_unique_name()
        emitter = CodeEmitter()

        # Generate a ramdom weight
        # Later, this should be replaced by real weight, maybe store in op_cfg.
        # weight = generate_valid_weight([op_cfg.out_channel, op_cfg.in_channel, op_cfg.ker_height, op_cfg.ker_width])
        # new_weight, info, fold = weight_transform(weight, cim_cfg, op_cfg)

        # bytes_weight = reduce(lambda x,y:x*y, new_weight.shape)
        # bytes_info = reduce(lambda x,y:x*y, info.shape)

        tile_out_channel_max = cim_cfg.bits_column * cim_cfg.n_macro
        tile_in_channel = self.decide_tile_in_channel(cim_cfg, op_cfg)

        # Alloc input
        if input_on_local:
            input_local_memory = Tensor(name=op_cfg.in_name, shape=[op_cfg.in_height, op_cfg.in_width, op_cfg.in_channel])
        else:
            input_size = op_cfg.in_height * op_cfg.in_width * op_cfg.in_channel
            input_global_memory = Tensor(name=op_cfg.in_name, shape=[op_cfg.in_height, op_cfg.in_width, op_cfg.in_channel])
            input_local_memory = emitter._emit_malloc(dtype=op_cfg.in_dtype, scope="local.mem", name=f"{op_cfg.in_name}_localmem",
                                                    size=input_size, shape=[op_cfg.in_height, op_cfg.in_width, op_cfg.in_channel])
            # Load input from global memory to local memory
            emitter._emit_trans(saddr_base=input_global_memory, saddr_offset=0, sscope="global",\
                            daddr_base=input_local_memory, daddr_offset=0, dscope="local.mem",
                            size=input_size
                            )
        
        # Alloc weight memory, and load weight from global memory to local memory
        weight_global_memory = Tensor(name=f"{op_cfg.weight_name}", shape=op_cfg.weight_shape)
        weight_local_memory = emitter._emit_malloc(dtype="int8", scope="local.wmem", size=op_cfg.bytes_weight, name=f"{op_cfg.weight_name}", 
                                    shape=[op_cfg.weight_shape[0],
                                            op_cfg.ker_height, op_cfg.ker_width,op_cfg.in_channel,
                                            op_cfg.weight_shape[2]])
        emitter._emit_trans(saddr_base=weight_global_memory, saddr_offset=0,sscope="global",
                        daddr_base=weight_local_memory, daddr_offset=0,dscope="local.wmem",
                        size=op_cfg.bytes_weight)

        # Alloc pim info, and load pim info from global memory to local memory
        info_global_memory = emitter._emit_malloc(dtype="int8", scope="global", name=f"{op_cfg.info_name}", size=op_cfg.bytes_info, shape=[op_cfg.bytes_info,])
        info_local_memory = emitter._emit_malloc(dtype="int8", scope="local.info", name=f"{op_cfg.info_name}_localmem", 
                                                size=op_cfg.bytes_info, 
                                                shape=[ op_cfg.info_shape[0], 
                                                        op_cfg.in_channel // tile_in_channel,
                                                        op_cfg.ker_height, op_cfg.ker_width, tile_in_channel,  # info.shape[1]
                                                        cim_cfg.n_macro,
                                                        op_cfg.info_shape[2] // cim_cfg.n_macro
                                                ])
        emitter._emit_trans(saddr_base=info_global_memory, saddr_offset=0,sscope="global",
                        daddr_base=info_local_memory, daddr_offset=0,dscope="local.info",
                        size=op_cfg.bytes_info)


        # Alloc output
        if output_on_local:
            output_local_memory = Tensor(name=f"{op_cfg.out_name}", shape=[op_cfg.out_height, op_cfg.out_width, op_cfg.out_channel])
        else:
            output_size = op_cfg.out_channel * op_cfg.out_height * op_cfg.out_width
            output_global_memory = Tensor(name=f"{op_cfg.out_name}", shape=[op_cfg.out_height, op_cfg.out_width, op_cfg.out_channel])
            output_local_memory = emitter._emit_malloc(dtype=op_cfg.out_dtype, scope="local.mem", name=f"{op_cfg.out_name}_localmem", 
                                                    size=output_size, shape=[op_cfg.out_height, op_cfg.out_width, op_cfg.out_channel])
            
        # Output temp memory for sum
        output_temp_local_memory = emitter._emit_malloc(dtype=op_cfg.out_dtype, scope="local.mem", name=f"{op_cfg.out_name}_temp_localmem", 
                                                size=op_cfg.out_channel, shape=[op_cfg.out_channel,])

        # Generate infomation about input and output of the program
        # emitter._emit_input(name=input_global_memory.name, shape=[1,op_cfg.in_height, op_cfg.in_width, op_cfg.in_channel], dtype=op_cfg.in_dtype)
        # emitter._emit_output(name=output_global_memory.name, shape=[1,op_cfg.out_height, op_cfg.out_width, op_cfg.out_channel], dtype=op_cfg.out_dtype)

        # Alloc input buffer
        input_pim_buffer = emitter._emit_malloc(dtype=op_cfg.in_dtype, scope="local.pbuf", name=f"{op_cfg.in_name}_pbuf", 
                                            size=cim_cfg.n_compartment, shape=[cim_cfg.n_compartment,])

        # Alloc output buffer
        output_pim_buffer_size = cim_cfg.bits_column * cim_cfg.n_macro
        output_pim_buffer = emitter._emit_malloc(dtype=op_cfg.out_dtype, scope="local.pbuf", size=output_pim_buffer_size, name=f"{op_cfg.out_name}_pbuf")

        # Alloc macro
        weight_macro_size = cim_cfg.n_row * cim_cfg.n_compartment * cim_cfg.n_macro * cim_cfg.bytes_column
        weight_macro = emitter._emit_malloc(dtype="int8", scope="local.macro", name=f"{op_cfg.weight_name}_macro", 
                                        size=weight_macro_size, shape=[cim_cfg.n_row, cim_cfg.n_compartment, cim_cfg.n_macro , cim_cfg.bytes_column])

        fold = op_cfg.fold
        out_channel_idx = 0
        # Outer loops : iterate over weights
        for oc_time_step, oc in enumerate(fold):
            for o_ic in range(op_cfg.in_channel // tile_in_channel):

                column_use = reduce(lambda i,j:i+j, oc)
                use_macro = ceil(column_use,cim_cfg.bits_column)
                
                n_element_in_one_filter = tile_in_channel * op_cfg.ker_height * op_cfg.ker_width
                weight_macro_use_size = n_element_in_one_filter * cim_cfg.n_macro * cim_cfg.bytes_column
                assert weight_macro_use_size<=weight_macro_size
                assert tile_in_channel % cim_cfg.n_compartment==0, f"tile_in_channel={tile_in_channel},cim_cfg.n_compartment={cim_cfg.n_compartment}"

                out_mask = parse_out_mask(oc)
                col_to_filter = parse_col_to_filter(oc)
                
                # Load weights from global memory to macro
                for hk in range(op_cfg.ker_height):
                    for wk in range(op_cfg.ker_width):
                        for i_ic in range(tile_in_channel):
                            total_row = hk * op_cfg.ker_width * tile_in_channel + wk * tile_in_channel + i_ic
                            row_idx = total_row // cim_cfg.n_compartment
                            comp_idx = total_row % cim_cfg.n_compartment
                            ic_idx = o_ic * tile_in_channel + i_ic
                            oc_idx = out_channel_idx
                            # Load bytes_column * n_macro B
                            emitter._emit_pim_store(saddr_base=weight_local_memory, 
                                            saddr_offset=weight_local_memory.get(oc_time_step, hk, wk, ic_idx, 0),
                                            daddr_base=weight_macro, 
                                            daddr_offset=weight_macro.get(row_idx, comp_idx, 0, 0),
                                            size=cim_cfg.bytes_column * use_macro)

                # Load info from info memory to reg
                num_in_filter = op_cfg.ker_height * op_cfg.ker_width * tile_in_channel
                pim_load_size = num_in_filter * cim_cfg.n_macro * cim_cfg.bits_column * 3 // 8
                # pim_load_size = num_in_filter * use_macro * cim_cfg.bits_column * 3 // 8
                emitter._emit_pim_load_info(addr_base=info_local_memory, 
                                        addr_offset=info_local_memory.get(oc_time_step, o_ic, 0, 0, 0, 0, 0), 
                                        size = pim_load_size)

                # Inner loops : iterate over inputs
                for ho in range(0, op_cfg.out_height):
                    for wo in range(0, op_cfg.out_width):
                        
                        # Clear output buffer
                        emitter._emit_pim_buffer_clear(addr_base=output_pim_buffer, addr_offset=0, length=output_pim_buffer_size)
                        
                        # Accumulate on output buffer
                        for hk in range(0, op_cfg.ker_height):
                            for wk in range(0, op_cfg.ker_width):
                                for i_ic in range(0, tile_in_channel // cim_cfg.n_compartment):
                                    
                                    pad_hi = ho*op_cfg.stride_h+hk
                                    pad_wi = wo*op_cfg.stride_w+wk

                                    # Check padding
                                    if pad_hi < op_cfg.padding[0] or pad_hi >= op_cfg.in_height + op_cfg.padding[0] or \
                                        pad_wi < op_cfg.padding[1] or pad_wi >= op_cfg.in_width + op_cfg.padding[1]:
                                        continue
                                    
                                    # Some value need to use
                                    total_row = hk * op_cfg.ker_width * tile_in_channel + wk * tile_in_channel + i_ic * cim_cfg.n_compartment
                                    row_idx = total_row // cim_cfg.n_compartment
                                    comp_idx = total_row % cim_cfg.n_compartment

                                    # ic_idx: absolute in channel
                                    ic_idx = o_ic * tile_in_channel + i_ic * cim_cfg.n_compartment

                                    # Load input to buffer
                                    in_height_idx = pad_hi-op_cfg.padding_h
                                    in_width_idx = pad_wi-op_cfg.padding_w
                                    in_channel_idx = o_ic * tile_in_channel + i_ic * cim_cfg.n_compartment
                                    emitter._emit_ldbuf(saddr_base=input_local_memory, 
                                                    saddr_offset=input_local_memory.get(in_height_idx, in_width_idx, in_channel_idx),\
                                                    daddr_base=input_pim_buffer, 
                                                    daddr_offset=0, \
                                                    size=cim_cfg.n_compartment)
                                    
                                    total_row_in_filter = hk * op_cfg.ker_width * tile_in_channel + wk * tile_in_channel + i_ic * cim_cfg.n_compartment
                                    info_offset = total_row_in_filter * cim_cfg.bits_column * cim_cfg.n_macro * 3 // 8 # Byte

                                    
                                    # emitter._emit_pim_load_info(addr_base=info_local_memory, 
                                    #                         addr_offset=info_local_memory.get(oc_time_step, hk, wk, ic_idx, 0,0), 
                                    #                         col_n=cim_cfg.bits_column, comp_n=cim_cfg.n_compartment, macro_n=use_macro)
                                    emitter._emit_pim_mult_csd(iaddr_base=input_pim_buffer,
                                                                iaddr_offset=0,
                                                                oaddr_base=output_pim_buffer,
                                                                oaddr_offset=0,
                                                                waddr_base=weight_macro, 
                                                                waddr_offset=weight_macro.get(row_idx,0,0,0),
                                                                broadcast=1, 
                                                                valid_col_num=cim_cfg.bits_column,
                                                                comp_off=cim_cfg.n_compartment,
                                                                macro_idx=use_macro,
                                                                num_filter = len(oc),
                                                                info_offset = info_offset,
                                                                column_use = column_use
                                                                )
                                    # if use_macro==cim_cfg.n_macro:
                                    #     emitter._emit_pim_load_info(addr_base=info_local_memory, 
                                    #                         addr_offset=info_local_memory.get(oc_time_step, hk, wk, ic_idx, 0,0), 
                                    #                         col_n=cim_cfg.bits_column, comp_n=cim_cfg.n_compartment, macro_n=cim_cfg.n_macro)
                                    #     emitter._emit_pim_mult_csd(iaddr_base=input_pim_buffer,
                                    #                                 iaddr_offset=0,
                                    #                                 oaddr_base=output_pim_buffer,
                                    #                                 oaddr_offset=0,
                                    #                                 waddr_base=weight_macro, 
                                    #                                 waddr_offset=weight_macro.get(row_idx,0,0,0),
                                    #                                 broadcast=1, 
                                    #                                 valid_col_num=cim_cfg.bits_column,
                                    #                                 comp_off=cim_cfg.n_compartment,
                                    #                                 )
                                    # else:
                                    #     # Do the pim-mult
                                    #     for macro_idx in range(use_macro):
                                    #         emitter._emit_pim_load_info(addr_base=info_local_memory, 
                                    #                             addr_offset=info_local_memory.get(oc_time_step, hk, wk, ic_idx, macro_idx, 0), 
                                    #                             col_n=cim_cfg.bits_column, comp_n=cim_cfg.n_compartment, macro_n=1)
                                    #         emitter._emit_pim_mult_csd(iaddr_base=input_pim_buffer,
                                    #                                 iaddr_offset=0,
                                    #                                 oaddr_base=output_pim_buffer,
                                    #                                 oaddr_offset=macro_idx * cim_cfg.bits_column,
                                    #                                 waddr_base=weight_macro, 
                                    #                                 waddr_offset=weight_macro.get(row_idx,0,macro_idx,0),
                                    #                                 broadcast=0, 
                                    #                                 valid_col_num=cim_cfg.bits_column,
                                    #                                 comp_off=cim_cfg.n_compartment
                                    #                                 )
                        
                        if all([i==2 for i in oc]):
                            emitter._emit_pim_outsum_move(oaddr_base=output_pim_buffer, oaddr_offset=0, out_n=len(oc))
                            if o_ic==0:
                                emitter._emit_stbuf(saddr_base=output_pim_buffer, saddr_offset=0,\
                                                        daddr_base=output_local_memory, daddr_offset=output_local_memory.get(ho, wo, out_channel_idx), \
                                                        size=len(oc))
                            else:
                                emitter._emit_stbuf(saddr_base=output_pim_buffer, saddr_offset=0,\
                                                        daddr_base=output_temp_local_memory, daddr_offset=0, \
                                                        size=len(oc))
                                emitter._emit_veadd(saddr0_base=output_local_memory, saddr0_offset=output_local_memory.get(ho, wo, out_channel_idx), 
                                                    saddr1_base=output_temp_local_memory, saddr1_offset=0, 
                                                    daddr_base=output_local_memory, daddr_offset=output_local_memory.get(ho, wo, out_channel_idx), 
                                                                                                    length=len(oc))
                        else:
                            # Out sum
                            emitter._emit_pim_outsum(oaddr_base=output_pim_buffer, oaddr_offset=0, out_mask=out_mask)

                            
                            # ipdb.set_trace()
                            if o_ic==0:
                                # Write output to local memory

                                out_channel_delta_index = 0
                                for i in range(len(out_mask)):
                                    case1 = (i==0)
                                    case2 = (i >=1 and out_mask[i-1]==0 and out_mask[i]==0)
                                    case3 = (i >=1 and out_mask[i-1]==0 and out_mask[i]==1)
                                    if case1 or case2 or case3:
                                        co = out_channel_idx + col_to_filter[i]
                                        # assert co==out_channel_idx + out_channel_delta_index
                                        # co = out_channel_idx + out_channel_delta_index
                                        emitter._emit_stbuf(saddr_base=output_pim_buffer, saddr_offset=i,\
                                                        daddr_base=output_local_memory, daddr_offset=output_local_memory.get(ho, wo, co), \
                                                        size=1)
                                        
                                        out_channel_delta_index += 1
                            else:
                                # Save to a temp space, then add to the output
                                # ipdb.set_trace()
                                if 0 in oc:
                                    emitter._emit_memset(addr_base=output_temp_local_memory,
                                                    addr_offset=0,
                                                    length=output_temp_local_memory.size,
                                                    imm=0)
                                # ipdb.set_trace()
                                out_channel_delta_index = 0
                                for i in range(len(out_mask)):
                                    case1 = (i==0)
                                    case2 = (i >=1 and out_mask[i-1]==0 and out_mask[i]==0)
                                    case3 = (i >=1 and out_mask[i-1]==0 and out_mask[i]==1)
                                    if case1 or case2 or case3:
                                        out_channel_delta_index_filled = col_to_filter[i]
                                        # assert out_channel_delta_index==out_channel_delta_index_filled, f"out_channel_delta_index={out_channel_delta_index},out_channel_delta_index_filled={out_channel_delta_index_filled}"
                                        emitter._emit_stbuf(saddr_base=output_pim_buffer, saddr_offset=i,\
                                                        daddr_base=output_temp_local_memory, daddr_offset=output_temp_local_memory.get(out_channel_delta_index_filled), \
                                                        size=1)
                                        
                                        out_channel_delta_index += 1

                                # Add
                                length = col_to_filter[-1]+1
                                # assert length==out_channel_delta_index
                                emitter._emit_veadd(saddr0_base=output_local_memory, saddr0_offset=output_local_memory.get(ho, wo, out_channel_idx), 
                                                    saddr1_base=output_temp_local_memory, saddr1_offset=0, 
                                                    daddr_base=output_local_memory, daddr_offset=output_local_memory.get(ho, wo, out_channel_idx), 
                                                                                                    length=length)

                        


                

            out_channel_idx += len(oc)

        # Free Macro Weight
        emitter._emit_free(weight_macro)

        # Free Input and Output on pim buffer
        emitter._emit_free(input_pim_buffer)
        emitter._emit_free(output_pim_buffer)

        # Free Local weight and info
        emitter._emit_free(weight_local_memory)
        emitter._emit_free(info_local_memory)
        emitter._emit_free(info_global_memory)

        if not output_on_local:
            # Write output back to global memory
            emitter._emit_trans(saddr_base=output_local_memory, saddr_offset=0,sscope="local.mem",\
                            daddr_base=output_global_memory, daddr_offset=0,dscope="global",
                            size=output_size
                            )
            emitter._emit_free(output_local_memory)
        if not input_on_local:
            emitter._emit_free(input_local_memory)
        
        return emitter.get_code_string()

register_op_strategy("bit_sparse_conv2d",BitSparseConv2dLongReduceStrategy(),1)
# register_op_strategy("nn.dense",BitSparseConv2dLongReduceStrategy(),1)

def run(weight, bsp_conv2d_cfg):
    cim_cfg = CimConfig.global_cim_config()
    param_cfg = None
    
    # Transform weights
    new_weight, info, fold = weight_transform(weight, cim_cfg, bsp_conv2d_cfg)

    strategy = BitSparseConv2dLongReduceStrategy()
    code = strategy.get_low_level_code(cim_cfg, bsp_conv2d_cfg, param_cfg, new_weight, info, fold)
    print(code)
    
    # weight_dict = {
    #     "W":new_weight,
    #     "INFO":info
    # }
    # with open("bit_sparse_weight","wb") as f:
    #     pickle.dump(weight_dict, f)

def test(weight, bsp_conv2d_cfg):
    
    cim_cfg = CimConfig.global_cim_config()
    param_cfg = None
    
    # Transform weights
    new_weight, info, fold = weight_transform(weight, cim_cfg, bsp_conv2d_cfg)
    # print(new_weight.shape)
    # print(info.shape)
    # print(fold)
    # exit()
    print("new_weight:")
    print(new_weight)
    print("info:")
    print(info)
    new_weight = tensor_int8_to_bits(new_weight)
    info = tensor_int8_to_bits(info)
    new_weight = new_weight.reshape(new_weight.shape[0], new_weight.shape[1], cim_cfg.bits_column * cim_cfg.n_macro)
    info = info.reshape(new_weight.shape[0], new_weight.shape[1], cim_cfg.bits_column * cim_cfg.n_macro, 3)
    recovered_wtensor = np.zeros((new_weight.shape[0], new_weight.shape[1], cim_cfg.bits_column * cim_cfg.n_macro), dtype=f"int32")
    for time_step in range(new_weight.shape[0]):
        for i_comp in range(new_weight.shape[1]):
            for i_col_and_macro in range(cim_cfg.bits_column * cim_cfg.n_macro):
                val = new_weight[time_step,i_comp, i_col_and_macro]
                sign = info[time_step, i_comp, i_col_and_macro, 0]
                location = info[time_step, i_comp, i_col_and_macro, 1:3]
                # print(type(val),type(sign),type(location))
                csd = recover_csd(val, sign, location)
                int_val = csd_to_int(csd)
                # print(f"value:",val,"sign:",sign,"location:",location,"csd:",csd,"int:",int_val)
                recovered_wtensor[time_step, i_comp, i_col_and_macro] = int_val
    print(fold)
    print(recovered_wtensor)

if __name__=="__main__":
    info_size = 1024*1024 # 1MB
    local_memory_size = 4*1024*1024 # 4MB
    pim_buffer_size = 4*1024 # 4KB
    CimConfig.GLOBAL_CIM_CONFIG = CimConfig(n_macro=4,n_compartment=32,n_row=64,bytes_column=2,local_memory_size=local_memory_size,pim_buffer_size=pim_buffer_size,info_size=info_size)

    bsp_conv2d_cfg = BitSparseConv2dConfig(in_name="I",weight_name="W",info_name="INFO",out_name="O",\
                    in_dtype="int8",out_dtype="int32",\
                    in_height=4,in_width=4,in_channel=512,\
                    out_channel=512,ker_height=3,ker_width=3,\
                    input_layout="NHWC",kernel_layout="HWIO",output_layout="NHWC",\
                    padding=[1,1,1,1],stride=[1,1])
    weight = generate_valid_weight([bsp_conv2d_cfg.out_channel, bsp_conv2d_cfg.in_channel, bsp_conv2d_cfg.ker_height, bsp_conv2d_cfg.ker_width])
    # print(weight)
    # test(weight, bsp_conv2d_cfg)
    # exit()
    run(weight, bsp_conv2d_cfg)
    exit()
    conv = torch.nn.Conv2d(out_channels=bsp_conv2d_cfg.out_channel, 
                    in_channels=bsp_conv2d_cfg.in_channel,
                    kernel_size=bsp_conv2d_cfg.ker_height,
                    bias=False)
    conv.weight = torch.nn.Parameter(torch.Tensor(weight).to(torch.float))
    in_shape = bsp_conv2d_cfg.in_channel * bsp_conv2d_cfg.in_height * bsp_conv2d_cfg.in_width
    input = torch.Tensor(np.arange(1,in_shape+1,dtype="int8").reshape(bsp_conv2d_cfg.in_channel,bsp_conv2d_cfg.in_height,bsp_conv2d_cfg.in_width)).to(torch.float)
    out = conv(torch.Tensor(input).to(torch.float))
    # input = torch.ones(1,bsp_conv2d_cfg.in_channel,bsp_conv2d_cfg.in_height,bsp_conv2d_cfg.in_width)
    out = conv(input)
    print("Weight:",weight.shape)
    print("input:",input.shape)
    print("out:")
    print(out.to(torch.int))