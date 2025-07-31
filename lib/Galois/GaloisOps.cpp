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
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  // Both inputs must be bytes in [0,255].
  for (unsigned i = 0; i < 2; ++i) {
    Value v = getOperand(i);
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t val = c.value();  // int64_t
      if (val < 0 || val > 255)
        return emitOpError("operand #")
               << i << " out of GF(2^8) range [0,255]: " << val;
    }
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
  auto lhsVals = getLhs();
  auto rhsVals = getRhs();
  auto outputType = mlir::dyn_cast<MemRefType>(getOutput().getType());

  int64_t M = getRowsA();
  int64_t K = getColsA();
  int64_t N = getColsB();

  // Check dimensions are positive
  if (M <= 0 || K <= 0 || N <= 0)
    return emitOpError("all matrix dimensions must be positive");

  // --- LHS check ---
  bool lhsIsMemref = (lhsVals.size() == 1) && mlir::isa<MemRefType>(lhsVals[0].getType());
  if (!lhsIsMemref) {
    if ((int64_t)lhsVals.size() != M * K)
      return emitOpError("lhs operand size does not match rowsA × colsA");
  } else {
    auto lhsType = mlir::cast<MemRefType>(lhsVals[0].getType());
    if (lhsType.hasStaticShape() && lhsType.getNumElements() < M * K)
      return emitOpError("lhs memref too small for rowsA × colsA");
  }

  // --- RHS check ---
  bool rhsIsMemref = (rhsVals.size() == 1) && mlir::isa<MemRefType>(rhsVals[0].getType());
  if (!rhsIsMemref) {
    if ((int64_t)rhsVals.size() != K * N)
      return emitOpError("rhs operand size does not match colsA × colsB");
  } else {
    auto rhsType = mlir::cast<MemRefType>(rhsVals[0].getType());
    if (rhsType.hasStaticShape() && rhsType.getNumElements() < K * N)
      return emitOpError("rhs memref too small for colsA × colsB");
  }

  // --- Output memref check ---
  if (outputType.hasStaticShape()) {
    int64_t outSize = outputType.getNumElements();
    if (outSize < M * N)
      return emitOpError("output memref too small for result matrix");
  }

  if (!outputType.getElementType().isInteger(32))
    return emitOpError("output must be memref of i32");

  // --- Value range check only for constant scalars ---
  if (!lhsIsMemref) {
    for (auto val : lhsVals)
      if (auto cst = val.getDefiningOp<arith::ConstantOp>())
        if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
          if (intVal.getInt() < 0 || intVal.getInt() > 255)
            return emitOpError("lhs values must be in [0, 255]");
  }

  if (!rhsIsMemref) {
    for (auto val : rhsVals)
      if (auto cst = val.getDefiningOp<arith::ConstantOp>())
        if (auto intVal = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
          if (intVal.getInt() < 0 || intVal.getInt() > 255)
            return emitOpError("rhs values must be in [0, 255]");
  }

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

// LogicalResult MixColumnsOp::verify() {
//     auto col = getCol();  // ValueRange of inputs
//     // Must have exactly 4 inputs and 4 results.
//     if (col.size() != 4)
//       return emitOpError("expects exactly 4 input bytes, got ") << col.size();
//     if (getNumResults() != 4)
//       return emitOpError("expects exactly 4 result bytes, got ") << getNumResults();
  
//     // // Ensure any constant inputs lie in [0,255].
//     // for (unsigned i = 0; i < 4; ++i) {
//     //   if (auto cst = col[i].getDefiningOp<arith::ConstantIntOp>()) {
//     //     int64_t v = cst.value();
//     //     if (v < 0 || v > 255)
//     //       return emitOpError("input #") << i
//     //              << " out of GF(2^8) range [0,255]: " << v;
//     //   }
//     // }
//     return success();
//   }


//===----------------------------------------------------------------------===//
// HashOp
//===----------------------------------------------------------------------===//

LogicalResult HashOp::verify() {
  if (getNumOperands() == 0)
    return emitOpError("requires at least one input byte");

  // Alpha attribute must be present and in [1,255]
  auto alphaAttr = (*this)->getAttrOfType<IntegerAttr>("alpha");
  if (!alphaAttr)
    return emitOpError("requires an 'alpha' IntegerAttr");
  int64_t alpha = alphaAttr.getInt();
  if (alpha < 1 || alpha > 255)
    return emitOpError("alpha out of GF(2^8) range [1,255]: ") << alpha;

  // Check each input constant (if constant) lies in [0,255]
  for (auto v : getData()) {
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t val = c.value();
      if (val < 0 || val > 255)
        return emitOpError("input out of GF(2^8) range [0,255]: ") << val;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// KeyExpansionOp
//===----------------------------------------------------------------------===//

LogicalResult KeyExpansionOp::verify() {
  // Must have exactly 16 input bytes and 16 results.
  if (getKeyBytes().size() != 16)
    return emitOpError("expects 16 key bytes, got ") << getKeyBytes().size();
  if (getNumResults() != 16)
    return emitOpError("must return 16 expanded bytes, got ") << getNumResults();

  // Round attr in [1,10]
  auto roundAttr = (*this)->getAttrOfType<IntegerAttr>("round");
  if (!roundAttr)
    return emitOpError("requires a 'round' IntegerAttr");
  int64_t rnd = roundAttr.getInt();
  if (rnd < 1 || rnd > 10)
    return emitOpError("round out of range [1,10]: ") << rnd;

  // Check any constant input is in [0,255]
  for (auto v : getKeyBytes()) {
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t byte = c.value();
      if (byte < 0 || byte > 255)
        return emitOpError("key byte out of GF(2^8) range: ") << byte;
    }
  }
  return success();
}
