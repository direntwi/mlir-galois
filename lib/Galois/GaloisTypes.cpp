//===- GaloisTypes.cpp - Galois dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "Galois/GaloisAttributes.h"
#include "Galois/GaloisTypes.h"
#include "mlir/IR/Types.h"
#include "Galois/GaloisDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/LogicalResult.h"

#define GET_TYPEDEF_CLASSES
#include "Galois/GaloisOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::galois;

mlir::LogicalResult mlir::galois::GF8Type::verify(function_ref<InFlightDiagnostic()> emitError, mlir::galois::GF8ConstantAttr value) {
    // Get the integer value stored inside the GF8_ConstantAttr
    int64_t num = value.getValue().getValue().getZExtValue();

    // Ensure it's within the valid range [0, 255]
    if (num < 0 || num > 255) {
      return emitError() << "GF8 constant must be in range2 [0, 255], but got: " << num;
    }
    return success();
}

void GaloisDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Galois/GaloisOpsTypes.cpp.inc"
      >();
}
