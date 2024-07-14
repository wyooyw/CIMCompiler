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

static Value getAddrValue(cim::CopyOp op, PatternRewriter &rewriter, int isDst){
  auto subViewOp = op.getOperand(isDst).getDefiningOp<memref::SubViewOp>();
  auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
  llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
  SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();
  
  Value addr_offset = getValue(offsets[0], rewriter);
  for(int i = 1; i<offsets.size(); i++){
    if(Value offset_i = getValue(offsets[i],rewriter)){
      Value shape_i = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), allocShapes[i]);
      Value mul = rewriter.create<arith::MulIOp>(op.getLoc(), addr_offset, shape_i);
      Value add = rewriter.create<arith::AddIOp>(op.getLoc(), mul, offset_i);
      addr_offset = add;
    }else{
      return nullptr;
    }
  }
  return addr_offset;
}

static Value getSizeValue(cim::CopyOp op, PatternRewriter &rewriter){
  auto subViewOp = op.getOperand(0).getDefiningOp<memref::SubViewOp>();
  SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();
  
  Value size = getValue(shapes[0], rewriter);
  for(int i = 1; i<shapes.size(); i++){
    if(Value shape_i = getValue(shapes[i],rewriter)){
      size = rewriter.create<arith::MulIOp>(op.getLoc(), size, shape_i);
    }else{
      return nullptr;
    }
  }
  return size;
}

// why need this namespace ?
namespace {


    struct TransOpLowering : public OpRewritePattern<cim::CopyOp> {
      using OpRewritePattern<cim::CopyOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(cim::CopyOp op, PatternRewriter &rewriter) const final {

        Value addr_src = getAddrValue(op, rewriter, 0);
        Value addr_dst = getAddrValue(op, rewriter, 1);
        Value size = getSizeValue(op, rewriter);
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
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, cimisa::CIMISADialect>();

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
  patterns.add<TransOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  std::cout << "CIMLoweringPass::runOnOperation finish!" << std::endl;
}

std::unique_ptr<Pass> mlir::cim::createCIMLoweringPass() {
  return std::make_unique<CIMLoweringPass>();
}