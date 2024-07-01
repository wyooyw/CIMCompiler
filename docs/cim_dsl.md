## Trans

Trans(src, dst)

src & dst: memref

lower: memref::copy

## Slice

subview = Slice(src, slice)
src: subview
slice: ?

lower: memref::subview

## VVAdd

VVAdd(src1, src2, dst)

src1 & src2 & dst: memref

lower: ISA vvadd

## CIMCompute

CIMCompute(src_in, src_weight, dst_output, group=..., ...)

lower: ISA pimcompute

## Buffer

buf = Buffer(shape, dtype, memory)
shape: 1d array
dtype & memory: string

lower: alloc

## Shape

shape_i = Shape(buf, i)
