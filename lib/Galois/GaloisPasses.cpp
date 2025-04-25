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
              mlir::galois::kLogAntilogTables, rewriter.getContext());
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
        parseSourceString<ModuleOp>(mlir::galois::kLogAntilogTables,
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


struct GaloisSBoxOpLowering : public OpRewritePattern<galois::SBoxOp> {
  using OpRewritePattern<galois::SBoxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::SBoxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // 1) Ensure sbox_table is in the module:
    if (!module.lookupSymbol<func::FuncOp>("sbox_table")) {
      auto savePt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToEnd(module.getBody());
      auto tblMod = parseSourceString<ModuleOp>(
          mlir::galois::kSBoxLookupTable, rewriter.getContext());
      if (!tblMod) return failure();
      SymbolTable sym(module);
      for (auto fn : tblMod->getOps<func::FuncOp>())
        if (!module.lookupSymbol(fn.getName()))
          rewriter.clone(*fn.getOperation());
      rewriter.restoreInsertionPoint(savePt);
    }

    // 2) Perform the table lookup:
    Value in = op.getInput();
    // call @sbox_table() -> tensor<256xi32>
    auto sym = SymbolRefAttr::get(rewriter.getContext(), "sbox_table");
    auto tblTy = RankedTensorType::get({256}, rewriter.getI32Type());
    Value table = rewriter
      .create<func::CallOp>(loc, sym, tblTy, ValueRange{})
      .getResult(0);

    // index‑cast the input byte to an index:
    Value idx = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), in);

    // extract the substituted byte:
    Value out = rewriter.create<tensor::ExtractOp>(
      loc, table, ArrayRef<Value>{idx});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct GaloisLFSRStepOpLowering : public OpRewritePattern<galois::LFSRStepOp> {
  using OpRewritePattern<galois::LFSRStepOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::LFSRStepOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // 1) Pull attrs
    auto widthAttr = op->getAttrOfType<IntegerAttr>("width");
    auto tapsAttr  = op->getAttrOfType<ArrayAttr>("taps");
    if (!widthAttr || !tapsAttr)
      return rewriter.notifyMatchFailure(op, "missing 'width' or 'taps' attr");
    int width = widthAttr.getInt();
    if (width < 1 || width > 32)
      return rewriter.notifyMatchFailure(op, "invalid LFSR width");

    // 2) Current state
    Value state = op.getInput();

    // 3) Compute feedback = XOR of all tapped bits
    //    feedback is an i32 0 or 1.
    Value feedback = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    for (Attribute a : tapsAttr) {
      int tap = cast<IntegerAttr>(a).getInt();
      // shift right by `tap` bits
      Value tapShift = rewriter.create<arith::ConstantIntOp>(loc, tap, 32);
      Value bit = rewriter.create<arith::ShRUIOp>(loc, state, tapShift);
      // mask to lowest bit
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value bit0 = rewriter.create<arith::AndIOp>(loc, bit, one);
      // XOR into feedback
      feedback = rewriter.create<arith::XOrIOp>(loc, feedback, bit0);
    }

    // 4) Shift the register right by one
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value shifted = rewriter.create<arith::ShRUIOp>(loc, state, one);

    // 5) Insert feedback into the MSB position (bit index = width-1)
    Value msbPos = rewriter.create<arith::ConstantIntOp>(loc, width - 1, 32);
    Value feedShifted = rewriter.create<arith::ShLIOp>(loc, feedback, msbPos);

    // 6) OR them together for next state
    Value nextState = rewriter.create<arith::OrIOp>(loc, shifted, feedShifted);

    rewriter.replaceOp(op, nextState);
    return success();
  }
};

struct GaloisRSEncodeOpLowering : public OpRewritePattern<galois::RSEncodeOp> {
  using OpRewritePattern<galois::RSEncodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::RSEncodeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Fetch attrs
    int64_t k = op->getAttrOfType<IntegerAttr>("messageLength").getInt();
    int64_t nsym = op->getAttrOfType<IntegerAttr>("paritySymbols").getInt();
    auto genCoeffs = op->getAttrOfType<ArrayAttr>("generatorPoly").getValue();

    // 1) Collect message operands
    SmallVector<Value> message(op.getOperands().begin(),
                                op.getOperands().end());
    // 2) Initialize parity[] = zero, length = nsym
    SmallVector<Value> parity(nsym,
      rewriter.create<arith::ConstantIntOp>(loc, 0, 32));

    // 3) Main loop: for each msg[i]
    for (int i = 0; i < k; ++i) {
      // feedback = msg[i] XOR parity[0]
      Value feedback = rewriter.create<arith::XOrIOp>(loc, message[i], parity[0]);
      // shift parity left by one
      for (int j = 0; j + 1 < nsym; ++j)
        parity[j] = parity[j+1];
      parity[nsym-1] = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      // update parity[j] ^= gf_mul(feedback, genCoeffs[j+1])
      for (int j = 0; j < nsym; ++j) {
        // genCoeffs[0] is the xⁿᵖᵒʷʳ⁻¹ term, so we start at [1]
        int64_t coeff = cast<IntegerAttr>(genCoeffs[j+1]).getInt();
        
        Value cst = rewriter.create<arith::ConstantIntOp>(loc, coeff, 32);
        Value prod = rewriter.create<galois::MulOp>(loc, feedback, cst);
        parity[j] = rewriter.create<arith::XOrIOp>(loc, parity[j], prod);
      }
    }

    // 4) Build the full codeword: message ++ parity
    SmallVector<Value> resultValues;
    resultValues.append(message.begin(), message.end());
    resultValues.append(parity.begin(), parity.end());

    rewriter.replaceOp(op, resultValues);
    return success();
  }
};

struct GaloisRSDecodeOpLowering : public OpRewritePattern<galois::RSDecodeOp> {
  using OpRewritePattern<galois::RSDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::RSDecodeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Fetch attrs
    int64_t k = op->getAttrOfType<IntegerAttr>("messageLength").getInt();
    int64_t nsym = op->getAttrOfType<IntegerAttr>("paritySymbols").getInt();

    // 1) Copy codeword operands into a working vector
    SmallVector<Value> work(op.getOperands().begin(), op.getOperands().end());
    SmallVector<Value> message(k);

    // 2) Polynomial “long division”:
    //    for i in 0..k-1:
    //      coef = work[i];
    //      message[i] = coef;
    //      for j in 1..nsym:
    //        work[i+j] = work[i+j] XOR (coef * g[j]);
    auto genCoeffs = op->getAttrOfType<ArrayAttr>("generatorPoly").getValue();
    for (int i = 0; i < k; ++i) {
      Value coef = work[i];
      message[i] = coef;
      for (int j = 1; j <= nsym; ++j) {
        int64_t coeff = cast<IntegerAttr>(genCoeffs[j]).getInt();
        
        Value cst = rewriter.create<arith::ConstantIntOp>(loc, coeff, 32);
        Value prod = rewriter.create<galois::MulOp>(loc, coef, cst);
        work[i + j] = rewriter.create<arith::XOrIOp>(loc, work[i + j], prod);
      }
    }

    // 3) Replace op with the first k entries (the recovered message)
    rewriter.replaceOp(op, message);
    return success();
  }
};

struct GaloisMatMulOpLowering : OpRewritePattern<galois::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // shape attrs
    int64_t M = op->getAttrOfType<IntegerAttr>("rowsA").getInt();
    int64_t K = op->getAttrOfType<IntegerAttr>("colsA").getInt();
    int64_t N = op->getAttrOfType<IntegerAttr>("colsB").getInt();

    // split operands into A and B
    auto operands = op.getOperands();
    SmallVector<Value> A(operands.begin(), operands.begin() + M*K);
    SmallVector<Value> B(operands.begin() + M*K, operands.end());

    // emit C entries
    SmallVector<Value> results;
    results.reserve(M * N);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) {
        Value acc = zero;
        for (int64_t k = 0; k < K; ++k) {
          Value prod = rewriter.create<galois::MulOp>(
              loc, A[i*K + k], B[k*N + j]);
          acc = rewriter.create<arith::XOrIOp>(loc, acc, prod);
        }
        results.push_back(acc);
      }

    rewriter.replaceOp(op, results);
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
    target.addIllegalOp<galois::SBoxOp>();
    target.addIllegalOp<galois::LFSRStepOp>();
    target.addIllegalOp<galois::RSEncodeOp>();
    target.addIllegalOp<galois::RSDecodeOp>();
    target.addIllegalOp<galois::MatMulOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GaloisAddOpLowering, 
                 GaloisMulOpLowering,
                 GaloisInvOpLowering,
                 GaloisSBoxOpLowering,
                 GaloisLFSRStepOpLowering,
                 GaloisRSEncodeOpLowering,
                 GaloisRSDecodeOpLowering,
                 GaloisMatMulOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGaloisToArithPass() { 
  return std::make_unique<ConvertGaloisToArithPass>();
}
} // namespace mlir::galois
