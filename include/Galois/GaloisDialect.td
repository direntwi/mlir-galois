//===- GaloisDialect.td - Galois dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_DIALECT
#define GALOIS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Galois dialect definition.
//===----------------------------------------------------------------------===//

def Galois_Dialect : Dialect {
    let name = "galois";
    let summary = "A galois out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::galois";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base galois operation definition.
//===----------------------------------------------------------------------===//

class Galois_Op<string mnemonic, list<Trait> traits = []> :
        Op<Galois_Dialect, mnemonic, traits>;

#endif // GALOIS_DIALECT
