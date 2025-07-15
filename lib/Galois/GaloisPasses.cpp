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
#include "mlir/Parser/Parser.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"

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

struct GaloisSubOpLowering : public OpRewritePattern<galois::SubOp> {
  using OpRewritePattern<galois::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::SubOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs(), rhs = op.getRhs();

    // XOR the two operands
    Value x = rewriter.create<arith::XOrIOp>(loc, lhs, rhs);
    // Mask to 8 bits
    Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xFF, 32);
    Value result = rewriter.create<arith::AndIOp>(loc, x, mask);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GaloisMulOpLowering : public OpRewritePattern<galois::MulOp> {
    using OpRewritePattern<galois::MulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(galois::MulOp op,
                                  PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      auto module = op->getParentOfType<ModuleOp>();
      if (!module)
        return rewriter.notifyMatchFailure(op, "not in a module");

      // --- 1) Ensure the log/antilog globals have been injected into the module.
      using GlobalOp = memref::GlobalOp;

      // Only inject if the module doesn’t already have a GlobalOp named “log_table”
      if (!module.lookupSymbol<GlobalOp>("log_table")) {
        auto lookupM = parseSourceString<ModuleOp>(kLogAntilogTables, rewriter.getContext());
        if (!lookupM) return failure();
        SymbolTable symTable(module);
        // 1) Clone all memref.global @log_table / @antilog_table
        for (auto glob : lookupM->getOps<GlobalOp>()) {
          if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointToEnd(module.getBody());
            rewriter.clone(*glob.getOperation());
          }
        }
      }

    auto logSymAttr = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto antiSymAttr = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");

    // --- 2) Prepare constants
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);

    // --- 3) Fetch operands
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // --- 4) Early checks: zero or one
    Value lhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, zero);
    Value rhsIsZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs, zero);
    Value eitherZero = rewriter.create<arith::OrIOp>(loc, lhsIsZero, rhsIsZero);

    Value lhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhs, one);
    Value rhsIsOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rhs, one);

    // --- 5) Precompute result for early cases
    // If lhs == 1 -> rhs
    // Else if rhs == 1 -> lhs
    // Else 0
    Value partialResult = rewriter.create<arith::SelectOp>(
        loc, lhsIsOne, rhs,
        rewriter.create<arith::SelectOp>(
            loc, rhsIsOne, lhs, zero));

    // --- 6) Did we hit any early exit? (eitherZero OR lhsIsOne OR rhsIsOne)
    Value anyEarly1 = rewriter.create<arith::OrIOp>(loc, eitherZero, lhsIsOne);
    Value anyEarly = rewriter.create<arith::OrIOp>(loc, anyEarly1, rhsIsOne);

    // --- 7) Prepare index adjustment for log lookup (subtract 1)
    // (only used if anyEarly == false)
    Value oneI32 = one;
    Value lhsAdj = rewriter.create<arith::SubIOp>(loc, lhs, oneI32);
    Value rhsAdj = rewriter.create<arith::SubIOp>(loc, rhs, oneI32);

    Value lhsIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), lhsAdj);
    Value rhsIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), rhsAdj);

    // --- 8) Load tables
    auto i32Ty = rewriter.getIntegerType(32);
    auto logMemrefTy = MemRefType::get({255}, i32Ty);
    auto antiMemrefTy = MemRefType::get({255}, i32Ty);

    Value logTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, logMemrefTy, logSymAttr);
    Value antilogTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, antiMemrefTy, antiSymAttr);

    // --- 9) Conditionally load log values (safe dummy 0 if early)
    Value dummyLog = zero;
    Value logValLhs = rewriter.create<arith::SelectOp>(
        loc, anyEarly, dummyLog,
        rewriter.create<memref::LoadOp>(loc, logTablePtr, lhsIdx));
    Value logValRhs = rewriter.create<arith::SelectOp>(
        loc, anyEarly, dummyLog,
        rewriter.create<memref::LoadOp>(loc, logTablePtr, rhsIdx));

    // --- 10) Sum logs and mod 255
    Value logSum = rewriter.create<arith::AddIOp>(loc, logValLhs, logValRhs);
    Value modConst = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);
    Value modSum = rewriter.create<arith::RemUIOp>(loc, logSum, modConst);

    // --- 11) Index cast for antilog lookup
    Value modIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), modSum);

    // --- 12) Load antilog value (safe dummy 0 if early)
    Value prodVal = rewriter.create<arith::SelectOp>(
        loc, anyEarly, zero,
        rewriter.create<memref::LoadOp>(loc, antilogTablePtr, modIdx));

    // --- 13) If early, return partialResult; else return prodVal
    Value finalResult = rewriter.create<arith::SelectOp>(
        loc, anyEarly, partialResult, prodVal);

    // --- 14) Replace op
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

    using GlobalOp = memref::GlobalOp;

    // 1) Inject lookup‑table funcs if missing
    if (!module.lookupSymbol<GlobalOp>("log_table")) {
      auto lookupM = parseSourceString<ModuleOp>(
          kLogAntilogTables, rewriter.getContext());
      if (!lookupM) return failure();
      SymbolTable symtab(module);
      for (auto glob : lookupM->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }

    auto logSymAttr = SymbolRefAttr::get(rewriter.getContext(), "log_table");
    auto antiSymAttr = SymbolRefAttr::get(rewriter.getContext(), "antilog_table");

    // --- 2) Constants
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value c255 = rewriter.create<arith::ConstantIntOp>(loc, 255, 32);

    // --- 3) Zero check
    Value in = op.getOperand();
    Value isZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, in, zero);

    // --- 4) Adjust index for log lookup (subtract 1)
    Value inAdj = rewriter.create<arith::SubIOp>(loc, in, one);
    Value inIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), inAdj);

    // --- 5) Load from log_table
    auto i32Ty = rewriter.getIntegerType(32);
    auto logMemrefTy = MemRefType::get({255}, i32Ty);
    Value logTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, logMemrefTy, logSymAttr);
    Value logVal = rewriter.create<memref::LoadOp>(loc, logTablePtr, inIdx);

    // --- 6) Compute (255 - logVal) mod 255
    Value diff = rewriter.create<arith::SubIOp>(loc, c255, logVal);
    Value invIdxI32 = rewriter.create<arith::RemUIOp>(loc, diff, c255);
    Value invIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), invIdxI32);

    // --- 7) Load from antilog_table
    auto antiMemrefTy = MemRefType::get({255}, i32Ty);
    Value antiTablePtr = rewriter.create<memref::GetGlobalOp>(
        loc, antiMemrefTy, antiSymAttr);
    Value invVal = rewriter.create<memref::LoadOp>(loc, antiTablePtr, invIdx);

    // --- 8) Select: if zero, return zero, else return invVal
    Value result = rewriter.create<arith::SelectOp>(
        loc, isZero, zero, invVal);

    // --- 9) Replace
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GaloisDivOpLowering : public OpRewritePattern<galois::DivOp> {
  using OpRewritePattern<galois::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::DivOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs(), rhs = op.getRhs();
    // 1) Compute inv(rhs)
    Value invR = rewriter.create<galois::InvOp>(loc, rhs);
    // 2) Multiply by lhs
    Value res  = rewriter.create<galois::MulOp>(loc, lhs, invR);
    rewriter.replaceOp(op, res);
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
    using GlobalOp = memref::GlobalOp;

    // 1) Inject lookup‑table funcs if missing
    if (!module.lookupSymbol<GlobalOp>("sbox_table")) {
      auto lookupM = parseSourceString<ModuleOp>(
          kSBoxLookupTable, rewriter.getContext());
      if (!lookupM) return failure();
      SymbolTable symtab(module);
      for (auto glob : lookupM->getOps<GlobalOp>()) {
        if (!module.lookupSymbol<GlobalOp>(glob.getSymName())) {
          OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.clone(*glob.getOperation());
        }
      }
    }

    // 2) Get the memref:
    Value tableMemref = rewriter.create<memref::GetGlobalOp>(
      loc,
      MemRefType::get({256}, rewriter.getI32Type()),
      "sbox_table"
    );

    // 3) Cast the input byte to index:
    Value idx = rewriter.create<arith::IndexCastOp>(
      loc,
      rewriter.getIndexType(),
      op.getInput()
    );

    // 4) Load the substituted value:
    Value out = rewriter.create<memref::LoadOp>(
      loc,
      tableMemref,
      idx
    );

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

struct GaloisLagrangeInterpOpLowering
    : public OpRewritePattern<galois::LagrangeInterpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::LagrangeInterpOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto coords = op.getCoords();
    size_t k = coords.size() / 2;

    // Split into x[i], y[i]
    SmallVector<Value> x(k), y(k);
    for (size_t i = 0; i < k; ++i) {
      x[i] = coords[2*i];
      y[i] = coords[2*i+1];
    }

    // Prepare result coeffs initialized to zero
    SmallVector<Value> coeffs(k,
      rewriter.create<arith::ConstantIntOp>(loc, 0, 32));

    // Compute P(x) via standard Lagrange basis:
    //   for each i: li = Π_{j≠i} (x_input - x[j])/(x[i]-x[j])
    //   term = y[i] * li
    //   coeffs += term * [1, x, x^2, …]   <- full poly multiplication
    //
    // Here’s a minimal “evaluate at a fixed x_input” variant,
    // or you can unroll to build the coefficient array.

    // (You’d replace any subtraction with XOR:)
    //   Value diff = rewriter.create<arith::XOrIOp>(loc, x[i], x[j]);

    // For brevity, suppose we just compute the constant term of P(x):
    for (size_t i = 0; i < k; ++i) {
      // Build li = ∏_{j≠i} (x[i] XOR x[j])⁻¹
      Value li = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      for (size_t j = 0; j < k; ++j) {
        if (i == j) continue;
        Value diff = rewriter.create<arith::XOrIOp>(loc, x[i], x[j]);
        Value inv  = rewriter.create<galois::InvOp>(loc, diff);
        li = rewriter.create<galois::MulOp>(loc, li, inv);
      }
      // term = y[i] * li
      Value term = rewriter.create<galois::MulOp>(loc, y[i], li);
      // accumulate into coeffs[0]
      coeffs[0] = rewriter.create<arith::XOrIOp>(loc, coeffs[0], term);
    }

    rewriter.replaceOp(op, coeffs);
    return success();
  }
};

struct GaloisMixColumnsOpLowering : public OpRewritePattern<MixColumnsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MixColumnsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto vec = op.getCol();  // 4-element ValueRange

    // 1) Build the constant 4×4 AES matrix M as a flat list of 16 GF(2^8) bytes:
    auto mkC = [&](int v) {
      return rewriter.create<arith::ConstantIntOp>(loc, v, 32);
    };
    SmallVector<Value> mat = {
      mkC(2), mkC(3), mkC(1), mkC(1),
      mkC(1), mkC(2), mkC(3), mkC(1),
      mkC(1), mkC(1), mkC(2), mkC(3),
      mkC(3), mkC(1), mkC(1), mkC(2)
    };

    // 2) Prepare the operands (mat + vec)
    SmallVector<Value> operands(mat.begin(), mat.end());
    operands.append(vec.begin(), vec.end());

    // 3) Setup attributes for matrix dimensions
    int64_t rowsA = 4;
    int64_t colsA = 4;
    int64_t colsB = 1;

    auto rowsAAttr = rewriter.getI32IntegerAttr(rowsA);
    auto colsAAttr = rewriter.getI32IntegerAttr(colsA);
    auto colsBAttr = rewriter.getI32IntegerAttr(colsB);

    NamedAttrList attrs;
    attrs.set("rowsA", rowsAAttr);
    attrs.set("colsA", colsAAttr);
    attrs.set("colsB", colsBAttr);

    // 4) Create the MatMulOp
   // 4) Create the generic MatMulOp
  auto mm = rewriter.create<galois::MatMulOp>(
    loc,
    // 1) result types
    TypeRange(op.getResultTypes()),
    // 2) first operand segment: the 4×4 matrix
    mat,
    // 3) second operand segment: the 4×1 vector
    vec,
    // 4) the three I32 attributes in order
    rowsAAttr, colsAAttr, colsBAttr
  );

  // 5) Replace original MixColumnsOp with results of MatMulOp
  rewriter.replaceOp(op, mm.getResults());

    return success();
  }
};

struct GaloisHashOpLowering : public OpRewritePattern<galois::HashOp> {
  using OpRewritePattern<galois::HashOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::HashOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto data = op.getData();  // ValueRange of bytes

    // Fetch alpha from attribute and make it a constant
    int64_t alphaVal = op->getAttrOfType<IntegerAttr>("alpha").getInt();
    Value alphaConst = rewriter.create<arith::ConstantIntOp>(loc, alphaVal, 32);

    // Initialize hash H = 0
    Value h = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    // H = (H * alpha) ⊕ b  for each b
    for (Value b : data) {
      // Multiply in GF(2^8)
      h = rewriter.create<galois::MulOp>(loc, h, alphaConst);
      // Add (XOR) the byte
      h = rewriter.create<galois::AddOp>(loc, h, b);
    }

    // Replace op with final hash
    rewriter.replaceOp(op, h);
    return success();
  }
};


struct GaloisKeyExpansionOpLowering
    : public OpRewritePattern<galois::KeyExpansionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(galois::KeyExpansionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto bytes = op.getKeyBytes();  // 16 inputs

    // 1) Split into 4-byte words: W0, W1, W2, W3
    SmallVector<Value> W0(bytes.begin()+0,  bytes.begin()+4);
    SmallVector<Value> W1(bytes.begin()+4,  bytes.begin()+8);
    SmallVector<Value> W2(bytes.begin()+8,  bytes.begin()+12);
    SmallVector<Value> W3(bytes.begin()+12, bytes.begin()+16);

    // 2) “Temp” = RotWord(W3): rotate left by 1 byte
    SmallVector<Value> Temp = { W3[1], W3[2], W3[3], W3[0] };

    // 3) SubWord(Temp) via SBox
    for (auto &b : Temp)
      b = rewriter.create<galois::SBoxOp>(loc, b);

    // 4) XOR Temp[0] with RCON for this round
    int64_t rnd = op->getAttrOfType<IntegerAttr>("round").getInt();
    // RCON values for AES-128 rounds 1..10
    static const uint8_t RCON[11] = {
      0x00, 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
    };
    Value rcon = rewriter.create<arith::ConstantIntOp>(
        loc, RCON[rnd], 32);
    Temp[0] = rewriter.create<arith::XOrIOp>(loc, Temp[0], rcon);

    // 5) Compute new words:
    //    W0' = W0 XOR Temp
    //    W1' = W1 XOR W0'
    //    W2' = W2 XOR W1'
    //    W3' = W3 XOR W2'
    SmallVector<Value> OutW(4);
    OutW[0] = rewriter.create<arith::XOrIOp>(loc, W0[0], Temp[0]);
    for (unsigned i = 1; i < 4; ++i)
      OutW[0] = rewriter.create<arith::XOrIOp>(loc, OutW[0], Temp[i]);

    auto xorBytes = [&](ArrayRef<Value> A, ArrayRef<Value> B) {
      SmallVector<Value> R(4);
      for (unsigned i = 0; i < 4; ++i)
        R[i] = rewriter.create<arith::XOrIOp>(loc, A[i], B[i]);
      return R;
    };
    SmallVector<Value> W0p = xorBytes(W0, Temp);
    SmallVector<Value> W1p = xorBytes(W1, W0p);
    SmallVector<Value> W2p = xorBytes(W2, W1p);
    SmallVector<Value> W3p = xorBytes(W3, W2p);

    // 6) Flatten all 16 output bytes
    SmallVector<Value> result;
    result.append(W0p.begin(), W0p.end());
    result.append(W1p.begin(), W1p.end());
    result.append(W2p.begin(), W2p.end());
    result.append(W3p.begin(), W3p.end());

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
    registry.insert<
    arith::ArithDialect,
    func::FuncDialect, 
    LLVM::LLVMDialect,
    memref::MemRefDialect>();
    
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<galois::AddOp>();
    target.addIllegalOp<galois::MulOp>();
    target.addIllegalOp<galois::InvOp>();
    target.addIllegalOp<galois::DivOp>();
    target.addIllegalOp<galois::SBoxOp>();
    target.addIllegalOp<galois::LFSRStepOp>();
    target.addIllegalOp<galois::RSEncodeOp>();
    target.addIllegalOp<galois::RSDecodeOp>();
    target.addIllegalOp<galois::MatMulOp>();
    target.addIllegalOp<galois::MixColumnsOp>();
    target.addIllegalOp<galois::LagrangeInterpOp>();
    target.addIllegalOp<galois::HashOp>();
    target.addIllegalOp<galois::KeyExpansionOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GaloisAddOpLowering,
                 GaloisSubOpLowering,
                 GaloisMulOpLowering,
                 GaloisInvOpLowering,
                 GaloisDivOpLowering,
                 GaloisSBoxOpLowering,
                 GaloisLFSRStepOpLowering,
                 GaloisRSEncodeOpLowering,
                 GaloisRSDecodeOpLowering,
                 GaloisMatMulOpLowering,
                 GaloisLagrangeInterpOpLowering,
                 GaloisMixColumnsOpLowering,
                 GaloisHashOpLowering,
                 GaloisKeyExpansionOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertGaloisToArithPass() { 
  return std::make_unique<ConvertGaloisToArithPass>();
}
} // namespace mlir::galois
