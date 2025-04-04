//===- GaloisAttributes.h - Galois dialect attributes -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_GALOISATTRIBUTES_H
#define GALOIS_GALOISATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#define GET_ATTRDEF_CLASSES
#include "Galois/GaloisAttributes.h.inc"

#endif // GALOIS_GALOISATTRIBUTES_H
