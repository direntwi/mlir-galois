//===- GaloisOps.cpp - Galois dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Galois/GaloisOps.h"
#include "Galois/GaloisDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define GET_OP_CLASSES
#include "Galois/GaloisOps.cpp.inc"

using namespace mlir;
using namespace mlir::galois;


//===----------------------------------------------------------------------===//
// Type Conversion Ops
//===----------------------------------------------------------------------===//

LogicalResult ToIntegerOp::verify() {
    if (!mlir::isa<GF8Type>(getInput().getType()))
        return emitOpError("expects input to be of type !galois.gf8");
    return success();
}


LogicalResult FromIntegerOp::verify() {
    auto intType = mlir::dyn_cast<IntegerType>(getInput().getType());
    if (!intType || intType.getWidth() != 32)
        return emitOpError("expects a 32-bit integer input");

    // Ensure the value is in the range [0, 255]
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
    if (!getLhs().getType().isInteger(32) || !getRhs().getType().isInteger(32)) {
        return emitOpError("expects i32 input operands");
    }
    if (!getResult().getType().isInteger(32)) {
        return emitOpError("expects i32 output");
    }    
    auto checkOperand = [&](Value operand) -> LogicalResult {
        IntegerAttr value;
        if (matchPattern(operand, m_Constant(&value))) {
            int64_t val = value.getValue().getSExtValue();
            if (val < 0 || val > 255) {
                return emitOpError()
                    << "operand value " << val << " out of range [0,255]";
            }
        }
        return success();
    };
    if (failed(checkOperand(getLhs())) || failed(checkOperand(getRhs()))) {
        return failure();
    }
    return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() {
    if (!getLhs().getType().isInteger(32) || !getRhs().getType().isInteger(32)) {
        return emitOpError("expects i32 input operands");
    }
    if (!getResult().getType().isInteger(32)) {
        return emitOpError("expects i32 output");
    }    
    auto checkOperand = [&](Value operand) -> LogicalResult {
        IntegerAttr value;
        if (matchPattern(operand, m_Constant(&value))) {
            int64_t val = value.getValue().getSExtValue();
            if (val < 0 || val > 255) {
                return emitOpError()
                    << "operand value " << val << " out of range [0,255]";
            }
        }
        return success();
    };
    if (failed(checkOperand(getLhs())) || failed(checkOperand(getRhs()))) {
        return failure();
    }
    return success();
}

//===----------------------------------------------------------------------===//
// InvOp
//===----------------------------------------------------------------------===//

LogicalResult InvOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() {
    for (unsigned i = 0; i < 2; ++i) {
        Value v = getOperand(i);
        if (auto cst = v.getDefiningOp<arith::ConstantIntOp>()) {
          int64_t val = cst.value();
          if (val < 0 || val > 255)
            return emitOpError("input values must be in range [0, 255]");
          if (i == 1 && val == 0)
            return emitOpError("division by zero");
        }
      }
      return success();
    }

//===----------------------------------------------------------------------===//
// SBoxOp
//===----------------------------------------------------------------------===//

LogicalResult SBoxOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// LFSRStepOp
//===----------------------------------------------------------------------===//

LogicalResult LFSRStepOp::verify() {
    if (auto constantOp = getInput().getDefiningOp<arith::ConstantOp>()) {
        auto intValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
        if (!intValue)
            return emitOpError("expects a constant integer input");
        if (intValue.getInt() < 0 || intValue.getInt() > 255)
            return emitOpError("input value must be in range [0, 255]");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// RSEncodeOp
//===----------------------------------------------------------------------===//


LogicalResult RSEncodeOp::verify() {
    auto msgLenAttr = getOperation()->getAttrOfType<IntegerAttr>("messageLength");
    auto nsymAttr = getOperation()->getAttrOfType<IntegerAttr>("paritySymbols");
    auto genAttr = getOperation()->getAttrOfType<ArrayAttr>("generatorPoly");
    if (!msgLenAttr || !nsymAttr || !genAttr)
      return emitOpError("requires 'messageLength', 'paritySymbols', and 'generatorPoly' attrs");
  
    int64_t k    = msgLenAttr.getInt();
    int64_t nsym = nsymAttr.getInt();
    // operands == k
    if (getNumOperands() != k)
      return emitOpError("operand count (")
             << getNumOperands() << ") must equal messageLength (" << k << ")";
    // results == k + nsym
    if (getNumResults() != k + nsym)
      return emitOpError("result count (")
             << getNumResults() << ") must equal messageLength + paritySymbols ("
             << (k + nsym) << ")";
    // generatorPoly length == nsym + 1
    if ((int)genAttr.size() != nsym + 1)
      return emitOpError("generatorPoly length (")
             << genAttr.size() << ") must equal paritySymbols+1 (" << (nsym+1) << ")";

    for (Value opnd : getOperands()) {
    if (auto cst = opnd.getDefiningOp<arith::ConstantIntOp>()) {
        // extract the integer literal
        int64_t val = cst.value();
        if (val < 0 || val > 255)
        return emitOpError("constant operand out of GF(2^8) range [0,255]: ")
                << val;
        }
    }
    return success();
  }

//===----------------------------------------------------------------------===//
// RSDecodeOp
//===----------------------------------------------------------------------===//

LogicalResult RSDecodeOp::verify() {
    auto msgLenAttr = getOperation()->getAttrOfType<IntegerAttr>("messageLength");
    auto nsymAttr = getOperation()->getAttrOfType<IntegerAttr>("paritySymbols");
    if (!msgLenAttr || !nsymAttr)
      return emitOpError("requires 'messageLength' and 'paritySymbols' attrs");
  
    int64_t k = msgLenAttr.getInt();
    int64_t nsym = nsymAttr.getInt();
    int64_t n = k + nsym;
  
    if (getNumOperands() != n)
      return emitOpError("operand count (")
             << getNumOperands() << ") must equal messageLength + paritySymbols ("
             << n << ")";
    if (getNumResults() != k)
      return emitOpError("result count (")
             << getNumResults() << ") must equal messageLength (" << k << ")";

    for (Value opnd : getOperands()) {
      if (auto cst = opnd.getDefiningOp<arith::ConstantIntOp>()) {
        // extract the integer literal
        int64_t val = cst.value();
        if (val < 0 || val > 255)
          return emitOpError("constant operand out of GF(2^8) range [0,255]: ")
                 << val;
        }
    }
  
    return success();
  }

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
    auto rowsA = getOperation()->getAttrOfType<IntegerAttr>("rowsA");
    auto colsA = getOperation()->getAttrOfType<IntegerAttr>("colsA");
    auto colsB = getOperation()->getAttrOfType<IntegerAttr>("colsB");
    if (!rowsA || !colsA || !colsB)
      return emitOpError("requires 'rowsA', 'colsA', and 'colsB' attrs");
  
    int64_t M = rowsA.getInt(), K = colsA.getInt(), N = colsB.getInt();
    int64_t numA = M * K, numB = K * N, numC = M * N;
  
    if (getNumOperands() != numA + numB)
      return emitOpError("expected ") << (numA + numB)
        << " inputs (M*K + K*N), got " << getNumOperands();
    if (getNumResults() != numC)
      return emitOpError("expected ") << numC
        << " results (M*N), got " << getNumResults();
    return success();
  }

//===----------------------------------------------------------------------===//
// LagrangeInterpOp
//===----------------------------------------------------------------------===//

  LogicalResult LagrangeInterpOp::verify() {
    auto coords = getCoords();
    // Must be an even, non-zero number of ints.
    if (coords.size() == 0 || coords.size() % 2 != 0)
        return emitOpError("expects an even, non-zero number of coordinate values");

    size_t k = coords.size() / 2;
    // Results must have exactly k outputs.
    if (getNumResults() != k)
        return emitOpError("must return exactly ")
            << k << " coefficients but got " << getNumResults();

    // Range-check any constant operands.
    for (Value v : coords) {
        if (auto cst = v.getDefiningOp<arith::ConstantIntOp>()) {
        int64_t val = cst.value();
        if (val < 0 || val > 255)
            return emitOpError("coordinate out of GF(2^8) range [0,255]: ")
                << val;
        }
    }

  return success();
  }

//===----------------------------------------------------------------------===//
// MixColumnsOp
//===----------------------------------------------------------------------===//

LogicalResult MixColumnsOp::verify() {
    auto col = getCol();  // ValueRange of inputs
    // Must have exactly 4 inputs and 4 results.
    if (col.size() != 4)
      return emitOpError("expects exactly 4 input bytes, got ") << col.size();
    if (getNumResults() != 4)
      return emitOpError("expects exactly 4 result bytes, got ") << getNumResults();
  
    // Ensure any constant inputs lie in [0,255].
    for (unsigned i = 0; i < 4; ++i) {
      if (auto cst = col[i].getDefiningOp<arith::ConstantIntOp>()) {
        int64_t v = cst.value();
        if (v < 0 || v > 255)
          return emitOpError("input #") << i
                 << " out of GF(2^8) range [0,255]: " << v;
      }
    }
    return success();
  }
