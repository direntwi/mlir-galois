//===- GaloisAttributes.cpp - Galois dialect attributes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Galois/GaloisAttributes.h"

#include "Galois/GaloisDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::galois;

LogicalResult GF8ConstantAttr::verify(function_ref<InFlightDiagnostic()> emitError, 
                                      IntegerAttr value) {
    int64_t num = value.getInt();
    if (num < 0 || num > 255) {
        return emitError() << "GF8 constant must be in range [0, 255], but got: " << num;
    }
    return success();
}
