//===- GaloisPasses.cpp - Galois passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Galois/GaloisOps.h"
#include "Galois/GaloisPasses.h"

namespace mlir::galois {
#define GEN_PASS_DEF_GALOISSWITCHBARFOO
#include "Galois/GaloisPasses.h.inc"

namespace {
class GaloisSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class GaloisSwitchBarFoo
    : public impl::GaloisSwitchBarFooBase<GaloisSwitchBarFoo> {
public:
  using impl::GaloisSwitchBarFooBase<
      GaloisSwitchBarFoo>::GaloisSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<GaloisSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace

struct GaloisAddOpLowering : public OpRewritePattern<galois::AddOp> {
  using OpRewritePattern<galois::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::AddOp op, 
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Location loc = op.getLoc();

    // XOR the two operands
    Value xorResult = rewriter.create<arith::XOrIOp>(loc, lhs, rhs);
    
    // Create 0xFF mask (i32) and apply to keep lower 8 bits
    Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xFF, 32);
    Value result = rewriter.create<arith::AndIOp>(loc, xorResult, mask);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertGaloisToArithPass 
    : public PassWrapper<ConvertGaloisToArithPass, OperationPass<>> {
  
  StringRef getArgument() const final { return "convert-galois-to-arith"; }
  StringRef getDescription() const final { 
    return "Convert Galois ops to Arith dialect operations"; 
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<galois::AddOp>();
    target.addLegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GaloisAddOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGaloisToArithPass() { 
  return std::make_unique<ConvertGaloisToArithPass>();
}
} // namespace mlir::galois
