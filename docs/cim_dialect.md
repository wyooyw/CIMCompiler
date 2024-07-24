## CIM Dialect

一些需要额外支持的Operator的集合

#### Shape

- 输入：RankedTypeTensor类型的input
- 输出：output为一个rank=1的RankedTypeTensor（vector），长度为input的rank
    - 这里用shortvector和tensor哪个更好？

目前这一版所有shape均为静态，均需要在编译期完成常量传播，不会用额外的内存或寄存器来专门存储shape信息

#### Trans

- 输入：RankedTypeTensor类型的input和output
- 输出：output的副本

直接lower为memref::copy

tensor dialect里为什么没有类似的语义？

#### Slice

是否能直接使用tensor::extract_slice或tensor::insert_slice？

bufferlization时，
- tensor::extract_slice会lower为subview
- tensor::insert_slice会lower为subview + copy

其实Slice的语义和memref::Subview一致，Trans的语义和memref::copy一致

可以直接跳过tensor::extract_slice和tensor::insert_slice，Slice就直接lower为memref::subview
