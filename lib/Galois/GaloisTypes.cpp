//===- GaloisTypes.cpp - Galois dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Galois/GaloisTypes.h"

#include "Galois/GaloisDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::galois;

#define GET_TYPEDEF_CLASSES
#include "Galois/GaloisOpsTypes.cpp.inc"

void GaloisDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Galois/GaloisOpsTypes.cpp.inc"
      , GF8Type>();
}
