//===- GaloisPasses.cpp - Galois passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "Galois/GaloisOps.h"
#include "Galois/GaloisPasses.h"
#include "Galois/GaloisHelpers.h"

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

struct GaloisMulOpLowering : public OpRewritePattern<galois::MulOp> {
  using OpRewritePattern<galois::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::MulOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) {
      return rewriter.notifyMatchFailure(op, "operation not in a module");
    }

    // Check if log_table exists; if not, parse the lookup tables module.
    if (!module.lookupSymbol<func::FuncOp>("log_table")) {
      // Parse the embedded module containing log and antilog tables.
      OwningOpRef<mlir::ModuleOp> lookupTablesModule =
          parseSourceString<mlir::ModuleOp>(
              mlir::galois::kGaloisLookupTables, rewriter.getContext());
      if (!lookupTablesModule)
        return failure();

      // Clone the functions into the current module.
      SymbolTable symbolTable(module);
      for (auto func : lookupTablesModule->getOps<func::FuncOp>()) {
        // Only clone if it doesn't already exist.
        if (!module.lookupSymbol(func.getName())) {
          auto insertPt = rewriter.saveInsertionPoint();//new
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*func.getOperation());//new
          rewriter.restoreInsertionPoint(insertPt);
        }
      }
    }

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Create constant zero (i32).
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    // Check if either operand is zero: (lhs == 0) OR (rhs == 0).
    Value lhsIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, zero);
    Value rhsIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, zero);
    Value eitherZero = rewriter.create<arith::OrIOp>(loc, lhsIsZero, rhsIsZero);

    // Obtain symbol references for the lookup table functions.
    auto logFuncSymbol = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto antilogFuncSymbol = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");

    // Call the log_table function to get the log table (tensor constant).
    // Adjust the tensor type if necessary. Here we assume tensor<256xi32>.
    auto logTableType = mlir::RankedTensorType::get({256}, rewriter.getIntegerType(32));
    auto logCall = rewriter.create<func::CallOp>(loc, logFuncSymbol, logTableType, ValueRange{});
    Value logTable = logCall.getResult(0);

    // Cast lhs and rhs to index type for tensor extraction.
    Value lhsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), lhs);
    Value rhsIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rhs);

    // Extract the log value for lhs and rhs using tensor.extract.
    Value logValLhs = rewriter.create<tensor::ExtractOp>(loc, logTable, ArrayRef<Value>{lhsIdx});
    Value logValRhs = rewriter.create<tensor::ExtractOp>(loc, logTable, ArrayRef<Value>{rhsIdx});

    // Compute the sum of logarithms.
    Value logSum = rewriter.create<arith::AddIOp>(loc, logValLhs, logValRhs);

    // Compute (logA + logB) mod 255.
    Value modConstant = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
    Value modSum = rewriter.create<arith::RemSIOp>(loc, logSum, modConstant);

    // Call the antilog_table function to get the antilog table (tensor constant).
    // Adjust the tensor type if necessary. For example, tensor<510xi32> for antilog.
    auto antilogTableType = mlir::RankedTensorType::get({510}, rewriter.getIntegerType(32));
    auto antilogCall = rewriter.create<func::CallOp>(loc, antilogFuncSymbol, antilogTableType, ValueRange{});
    Value antilogTable = antilogCall.getResult(0);

    // Cast modSum to index type for tensor extraction.
    Value modIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), modSum);

    // Extract the corresponding element from the antilog table using modSum as the index.
    Value prodVal = rewriter.create<tensor::ExtractOp>(loc, antilogTable, ArrayRef<Value>{modIdx});

    // Use a select op: if either input is zero, the result is zero; otherwise, use the extracted value.
    Value finalResult = rewriter.create<arith::SelectOp>(loc, eitherZero, zero, prodVal);

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

struct GaloisInvOpLowering : public OpRewritePattern<galois::InvOp> {
  using OpRewritePattern<galois::InvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::InvOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a module");

    // 1) Inject lookup‑table funcs if missing
    if (!module.lookupSymbol<func::FuncOp>("log_table")) {
      auto savePt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToEnd(module.getBody());
      OwningOpRef<ModuleOp> tableMod =
        parseSourceString<ModuleOp>(mlir::galois::kGaloisLookupTables,
                                    rewriter.getContext());
      if (!tableMod)
        return failure();
      SymbolTable symtab(module);
      for (auto fn : tableMod->getOps<func::FuncOp>())
        if (!module.lookupSymbol(fn.getName()))
          rewriter.clone(*fn.getOperation());
      rewriter.restoreInsertionPoint(savePt);
    }

    // 2) Zero check
    Value in = op.getOperand();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value isZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, in, zero);

    // 3) log lookup
    auto logSym = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto logTy = RankedTensorType::get({256}, rewriter.getIntegerType(32));
    Value logTbl = rewriter
                       .create<func::CallOp>(loc, logSym, logTy, ValueRange{})
                       .getResult(0);

    // index-cast input -> index
    Value inIdx =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), in);
    Value logVal = rewriter.create<tensor::ExtractOp>(
        loc, logTbl, ArrayRef<Value>{inIdx});

    // 4) compute (255 - logVal) mod 255
    Value c255 = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
    Value diff  = rewriter.create<arith::SubIOp>(loc, c255, logVal);
    Value invIdx = rewriter.create<arith::RemSIOp>(loc, diff, c255);

    // 5) antilog lookup
    auto antiSym = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");
    auto antiTy  = RankedTensorType::get({510}, rewriter.getIntegerType(32));
    Value antiTbl = rewriter
                        .create<func::CallOp>(loc, antiSym, antiTy, ValueRange{})
                        .getResult(0);

    // cast invIdx -> index and extract
    Value idx  = rewriter.create<arith::IndexCastOp>(loc,
                          rewriter.getIndexType(), invIdx);
    Value res  = rewriter.create<tensor::ExtractOp>(
        loc, antiTbl, ArrayRef<Value>{idx});

    // 6) select zero vs. result
    Value result = rewriter.create<arith::SelectOp>(loc, isZero, zero, res);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertGaloisToArithPass 
    : public PassWrapper<ConvertGaloisToArithPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-galois-to-arith"; }
  StringRef getDescription() const final { 
    return "Convert Galois ops to Arith dialect operations"; 
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<galois::AddOp>();
    target.addIllegalOp<galois::MulOp>();
    target.addIllegalOp<galois::InvOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GaloisAddOpLowering, 
                 GaloisMulOpLowering,
                 GaloisInvOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGaloisToArithPass() { 
  return std::make_unique<ConvertGaloisToArithPass>();
}
} // namespace mlir::galois
