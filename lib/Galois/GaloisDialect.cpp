//===- GaloisDialect.cpp - Galois dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Galois/GaloisDialect.h"
#include "Galois/GaloisOps.h"
#include "Galois/GaloisTypes.h"

using namespace mlir;
using namespace mlir::galois;

#include "Galois/GaloisOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Galois dialect.
//===----------------------------------------------------------------------===//

void GaloisDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Galois/GaloisOps.cpp.inc"
      >();
  registerTypes();
}
