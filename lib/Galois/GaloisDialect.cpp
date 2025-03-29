//===- GaloisDialect.cpp - Galois dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Galois/GaloisDialect.h"
#include "Galois/GaloisAttributes.h"
#include "Galois/GaloisOps.h"
#include "Galois/GaloisTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::galois;

#include "Galois/GaloisOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Galois dialect.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Galois/GaloisAttributes.cpp.inc"


void GaloisDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Galois/GaloisAttributes.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "Galois/GaloisOps.cpp.inc"
      >();
  registerTypes();
}
