//===- GaloisPasses.h - Galois passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GALOIS_GALOISPASSES_H
#define GALOIS_GALOISPASSES_H

#include "Galois/GaloisDialect.h"
#include "Galois/GaloisOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace galois {
#define GEN_PASS_DECL
#include "Galois/GaloisPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Galois/GaloisPasses.h.inc"
} // namespace galois
} // namespace mlir

#endif
