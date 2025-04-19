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
