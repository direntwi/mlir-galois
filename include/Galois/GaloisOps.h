//===- GaloisOps.h - Galois dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_GALOISOPS_H
#define GALOIS_GALOISOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Galois/GaloisAttributes.h"
#include "Galois/GaloisTypes.h"

#define GET_OP_CLASSES
#include "Galois/GaloisOps.h.inc"

#endif // GALOIS_GALOISOPS_H
