#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "cim/Dialect.h"
#include "cimisa/Dialect.h"
#include "cim/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <iostream>
using namespace mlir;

static Value getValue(OpFoldResult offset, PatternRewriter &rewriter){
    if (Attribute attr = llvm::dyn_cast_if_present<Attribute>(offset)) {
        Value value = rewriter.create<arith::ConstantIndexOp>(
                              rewriter.getUnknownLoc(), 
                              cast<IntegerAttr>(attr).getInt());
        return value;
    } else if(Value value = llvm::dyn_cast_if_present<Value>(offset)) {
        return value;
    }else{
        return nullptr;
    }
}

static int getBitWidth(mlir::Type type){
  if(type.isa<mlir::IntegerType>()){
    return type.getIntOrFloatBitWidth();
  }else if(type.isa<mlir::FloatType>()){
    return type.getIntOrFloatBitWidth();
  }else if(type.isa<mlir::IndexType>()){
    return 32;
  }else{
    std::cout << "getBitWidth fail" << std::endl;
    std::exit(1);
    return 0;
  }
}

static int getBitWidthMemRefOperand(mlir::Value operand){
  mlir::MemRefType type = llvm::cast<mlir::MemRefType>(operand.getType());
  return getBitWidth(type.getElementType());
}

static Value getBufferBaseAddr(Value buffer, PatternRewriter &rewriter){
  mlir::MemRefType type = llvm::cast<mlir::MemRefType>(buffer.getType());
  mlir::DictionaryAttr memory_space = llvm::cast<mlir::DictionaryAttr>(type.getMemorySpace());
  int address = llvm::cast<mlir::IntegerAttr>(memory_space.get("address")).getInt();
  mlir::Value base_addr = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), address);
  return base_addr;
}

static Value getAddrValue(Value operand, PatternRewriter &rewriter){
  if(auto alloc_op = operand.getDefiningOp<memref::AllocOp>()){
    mlir::Value addr = getBufferBaseAddr(alloc_op.getResult(), rewriter);
    return addr;
  }else if(auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()){
    auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
    if (!allocOp){
      std::cout << "getAddrValue allocOp==nullptr" << std::endl;
      return nullptr;
    }
    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();
    
    Value addr_offset = getValue(offsets[0], rewriter);
    for(int i = 1; i<offsets.size(); i++){
      if(Value offset_i = getValue(offsets[i],rewriter)){
        Value shape_i = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[i]);
        Value mul = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), addr_offset, shape_i);
        Value add = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), mul, offset_i);
        addr_offset = add;
      }else{
        return nullptr;
      }
    }
    int64_t bitwidth = getBitWidthMemRefOperand(operand);
    int64_t bytewidth = bitwidth / 8;
    Value bytewidth_value = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), bytewidth);
    Value byte_addr_offset = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), addr_offset, bytewidth_value);
    
    mlir::Value addr_base = getBufferBaseAddr(allocOp.getResult(), rewriter);
    mlir::Value real_address = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), addr_base, byte_addr_offset);
    return real_address;
  }else{
    std::cout << "getAddrValue fail" << std::endl;
    std::exit(1);
    return nullptr;
  }
  
}

static Value getLengthValue(Value operand, PatternRewriter &rewriter){
  if(auto allocOp = operand.getDefiningOp<memref::AllocOp>()){

    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    int64_t size = 1;
    for(int i = 0; i<allocShapes.size(); i++){
      size *= allocShapes[i];
    }
    mlir::Value zero = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), size);
    return zero;
  }else if(auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()){
    SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();
    
    Value size = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
    for(int i = 0; i<shapes.size(); i++){
      if(Value shape_i = getValue(shapes[i],rewriter)){
        size = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), size, shape_i);
      }else{
        return nullptr;
      }
    }
    return size;
  }else{
    std::cout << "getSizeValue fail" << std::endl;
    std::exit(1);
    return nullptr;
  }
}


static Value getSizeValue(Value operand, PatternRewriter &rewriter){
  if(auto allocOp = operand.getDefiningOp<memref::AllocOp>()){
    int bitwidth = getBitWidthMemRefOperand(operand);
    int bytewidth = bitwidth / 8;

    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    int64_t size = bytewidth;
    for(int i = 0; i<allocShapes.size(); i++){
      size *= allocShapes[i];
    }
    mlir::Value zero = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), size);
    return zero;
  }else if(auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()){
    SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();
    
    int bitwidth = getBitWidthMemRefOperand(operand);
    int bytewidth = bitwidth / 8;
    
    Value size = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), bytewidth);
    for(int i = 0; i<shapes.size(); i++){
      if(Value shape_i = getValue(shapes[i],rewriter)){
        size = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), size, shape_i);
      }else{
        return nullptr;
      }
    }
    return size;
  }else{
    std::cout << "getSizeValue fail" << std::endl;
    std::exit(1);
    return nullptr;
  }
}

static std::vector<Value> _getMacroActivatePositionBySubview(cim::CIMComputeOp op, PatternRewriter &rewriter, int operand_index){
  // <N_ROW, N_COMP, N_MACRO, N_VCOL>
  auto subViewOp = op.getOperand(operand_index).getDefiningOp<memref::SubViewOp>();
  auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
  llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
  SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();
  SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();
  
  // get activate row
  Value activate_row_begin = getValue(offsets[0], rewriter);
  // Value activate_row_length = getValue(shapes[0], rewriter);

  // get activate compartment
  // Value activate_comp_begin = getValue(shapes[1], rewriter);
  // Value activate_comp_length = getValue(shapes[1], rewriter);

  // get activate macro
  // Value activate_macro_begin = getValue(shapes[2], rewriter);
  Value activate_macro_length = getValue(shapes[2], rewriter);

  // get activate element in macro
  // Value activate_element_begin = getValue(shapes[3], rewriter);
  // Value activate_element_length = getValue(shapes[3], rewriter);

  Value element_per_macro = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), allocShapes[3]);
  Value activate_element_num = rewriter.create<arith::MulIOp>(op.getLoc(), activate_macro_length, element_per_macro);

  return {
    activate_row_begin, 
    activate_element_num
  };
}

static std::vector<Value> getMacroActivatePosition(cim::CIMComputeOp op, PatternRewriter &rewriter, int operand_index){
  if(op.getOperand(operand_index).getDefiningOp<memref::SubViewOp>()){
    return _getMacroActivatePositionBySubview(op, rewriter, operand_index);
  }else{
    // fail
    std::cout << "getMacroActivatePosition fail" << std::endl;
    return {};
  }
}

static IntegerAttr getI1IntegerAttr(int32_t value, PatternRewriter &rewriter) {
  return IntegerAttr::get(rewriter.getIntegerType(1), APInt(1, value));
}

// why need this namespace ?
namespace {

    struct LoadOpLowering : public OpRewritePattern<memref::LoadOp> {
      using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(memref::LoadOp op, PatternRewriter &rewriter) const final {
        /*
          Now, all index of load is 0.
        */
        std::cout << "LoadOpLowering::matchAndRewrite 1" << std::endl;
        Value addr_src = getAddrValue(op.getOperand(0), rewriter);
        std::cout << "LoadOpLowering::matchAndRewrite 4" << std::endl;
        if (!addr_src) {
          std::cout << "LoadOpLowering::matchAndRewrite fail" << std::endl;
          return failure();
        }
        std::cout << "LoadOpLowering::matchAndRewrite success" << std::endl;

        MemRefType memtype = llvm::cast<mlir::MemRefType>(op.getOperand(0).getType());
        Type type = memtype.getElementType();
        mlir::cimisa::LoadOp new_op = rewriter.create<mlir::cimisa::LoadOp>(op.getLoc(), type, addr_src);
        rewriter.replaceOp(op, {new_op.getResult()});
        // rewriter.replaceOpWithNewOp<mlir::cimisa::LoadOp>(op, type, addr_src);
        return success();
      }
    };

    struct StoreOpLowering : public OpRewritePattern<memref::StoreOp> {
      using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(memref::StoreOp op, PatternRewriter &rewriter) const final {
        /*
          Now, all index of load is 0.
        */
        std::cout << "StoreOpLowering::matchAndRewrite 1" << std::endl;
        Value value = op.getOperand(0);
        Value addr_dst = getAddrValue(op.getOperand(1), rewriter);
        std::cout << "StoreOpLowering::matchAndRewrite 4" << std::endl;
        if (!addr_dst || !value) {
          std::cout << "StoreOpLowering::matchAndRewrite fail" << std::endl;
          return failure();
        }
        std::cout << "StoreOpLowering::matchAndRewrite success" << std::endl;

        rewriter.replaceOpWithNewOp<cimisa::StoreOp>(op, addr_dst, value);
        return success();
      }
    };

    struct TransOpLowering : public OpRewritePattern<cim::CopyOp> {
      using OpRewritePattern<cim::CopyOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(cim::CopyOp op, PatternRewriter &rewriter) const final {

        Value addr_src = getAddrValue(op.getOperand(0), rewriter);
        Value addr_dst = getAddrValue(op.getOperand(1), rewriter);
        Value size = getSizeValue(op.getOperand(0), rewriter);
        std::cout << "TransOpLowering::matchAndRewrite" << std::endl;
        if (!addr_src || !addr_dst || !size) {
          std::cout << "TransOpLowering::matchAndRewrite fail" << std::endl;
          return failure();
        }
        std::cout << "TransOpLowering::matchAndRewrite success" << std::endl;
        
        rewriter.replaceOpWithNewOp<cimisa::TransOp>(op, addr_src, addr_dst, size);

        return success();
      }
    };

    struct CIMComputeOpLowering : public OpRewritePattern<cim::CIMComputeOp> {
      using OpRewritePattern<cim::CIMComputeOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(cim::CIMComputeOp op, PatternRewriter &rewriter) const final {
        std::cout << "CIMComputeOpLowering::matchAndRewrite 1" << std::endl;
        Value addr_input = getAddrValue(op.getOperand(0), rewriter);
        std::cout << "CIMComputeOpLowering::matchAndRewrite 2" << std::endl;
        Value addr_output = getAddrValue(op.getOperand(2), rewriter);
        std::cout << "CIMComputeOpLowering::matchAndRewrite 3" << std::endl;
        std::vector<Value> macro_activate = getMacroActivatePosition(op, rewriter, 1);
        std::cout << "CIMComputeOpLowering::matchAndRewrite 4" << std::endl;
        if(macro_activate.size() < 2){
          std::cout << "CIMComputeOpLowering::matchAndRewrite fail 1" << std::endl;
          return failure();
        }
        Value row_index = macro_activate[0];
        Value activate_element_num = macro_activate[1];

        
        if (!addr_input || !addr_output || !row_index || !activate_element_num) {
          std::cout << "CIMComputeOpLowering::matchAndRewrite fail 2" << std::endl;
          return failure();
        }
        std::cout << "CIMComputeOpLowering::matchAndRewrite success" << std::endl;

        IntegerAttr input_bw = rewriter.getI8IntegerAttr(8);
        IntegerAttr output_bw = rewriter.getI8IntegerAttr(32);
        IntegerAttr weight_bw = rewriter.getI8IntegerAttr(8);

        IntegerAttr acc_flag = getI1IntegerAttr(1, rewriter);
        IntegerAttr value_sparse_flag = getI1IntegerAttr(0, rewriter);
        IntegerAttr bit_sparse_flag = getI1IntegerAttr(0, rewriter);
        
        rewriter.replaceOpWithNewOp<cimisa::CIMComputeOp>(op, 
              addr_input,           // AnyTypeOf<[AnyInteger, Index]>:$input_addr, 
              addr_output,          // AnyTypeOf<[AnyInteger, Index]>:$output_addr, 
              row_index,            // AnyTypeOf<[AnyInteger, Index]>:$row_index,
              activate_element_num, // AnyTypeOf<[AnyInteger, Index]>:$activate_element_num,
              input_bw,             // I8:$input_bw,
              output_bw,            // I8:$output_bw, 
              weight_bw,            // I8:$weight_bw,
              acc_flag,             // I1:$acc_flag,
              value_sparse_flag,    // I1:$value_sparse_flag,
              bit_sparse_flag       // I1:$bit_sparse_flag
        );

        return success();
      }
    };

    struct VVAddOpLowering : public OpRewritePattern<cim::VVAddOp> {
      using OpRewritePattern<cim::VVAddOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(cim::VVAddOp op, PatternRewriter &rewriter) const final {

        Value lhs = getAddrValue(op.getOperand(0), rewriter);
        Value rhs = getAddrValue(op.getOperand(1), rewriter);
        Value result = getAddrValue(op.getOperand(2), rewriter);
        Value size = getLengthValue(op.getOperand(0), rewriter);
        
        std::cout << "VVAddOpLowering::matchAndRewrite" << std::endl;
        if (!lhs || !rhs || !result) {
          std::cout << "VVAddOpLowering::matchAndRewrite fail" << std::endl;
          return failure();
        }
        std::cout << "VVAddOpLowering::matchAndRewrite success" << std::endl;

        int64_t _bitwidth_lhs = getBitWidthMemRefOperand(op.getOperand(0));
        int64_t _bitwidth_rhs = getBitWidthMemRefOperand(op.getOperand(1));
        int64_t _bitwidth_out = getBitWidthMemRefOperand(op.getOperand(2));

        IntegerAttr bitwidth_lhs = rewriter.getI8IntegerAttr(_bitwidth_lhs);
        IntegerAttr bitwidth_rhs = rewriter.getI8IntegerAttr(_bitwidth_rhs);
        IntegerAttr bitwidth_out = rewriter.getI8IntegerAttr(_bitwidth_out);
        
        rewriter.replaceOpWithNewOp<cimisa::VVAddOp>(op, lhs, rhs, result, size, 
                bitwidth_lhs, bitwidth_rhs, bitwidth_out);

        return success();
      }
    };
}



namespace {
struct CIMLoweringPass
    : public PassWrapper<CIMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void CIMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  std::cout << "CIMLoweringPass::runOnOperation" << std::endl;
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  // target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
  //                        arith::ArithDialect, func::FuncDialect,
  //                        cimisa::CIMISADialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  // target.addIllegalDialect<cim::CIMDialect>();
//   target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
//     return llvm::none_of(op->getOperandTypes(),
//                          [](Type type) { return llvm::isa<TensorType>(type); });
//   });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<TransOpLowering,CIMComputeOpLowering,LoadOpLowering,StoreOpLowering,
              VVAddOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  // if (failed(
  //         applyPartialConversion(getOperation(), target, std::move(patterns))))
  //   signalPassFailure();
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  std::cout << "CIMLoweringPass::runOnOperation finish!" << std::endl;
}

std::unique_ptr<Pass> mlir::cim::createCIMLoweringPass() {
  return std::make_unique<CIMLoweringPass>();
}