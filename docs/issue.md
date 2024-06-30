
#### 1.insert_slice和extract_slice是怎么lower到bufferization的？

1.创建dest buffer的subview

```c++
// Take a subview of the destination buffer.
auto dstMemrefType = cast<MemRefType>(dstMemref->getType());
auto subviewMemRefType =
    cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        insertSliceOp.getSourceType().getShape(), dstMemrefType,
        mixedOffsets, mixedSizes, mixedStrides));
Value subView = rewriter.create<memref::SubViewOp>(
    loc, subviewMemRefType, *dstMemref, mixedOffsets, mixedSizes,
    mixedStrides);
```

2.将src buffer拷贝到dest subview buffer

```c++
if (failed(options.createMemCpy(rewriter, loc, *srcMemref, subView)))
    return failure();
```

问题：如果subview是不连续的，这里的拷贝怎么办？是一条拷贝里可以指定stride，还是生成多条连续的拷贝指令？

从memref::copy的[文档](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcopy-memrefcopyop)来看，可能是在进一步lower的时候才会拆:

```
operation ::= `memref.copy` $source `,` $target attr-dict `:` type($source) `to` type($target)
```

2.能否直接从memref开始，跳过tensor？没有SSA的话，还能做函数内联吗？