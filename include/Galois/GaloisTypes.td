//===- GaloisTypes.td - Galois dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_TYPES
#define GALOIS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Galois/GaloisDialect.td"

//===----------------------------------------------------------------------===//
// Galois type definitions
//===----------------------------------------------------------------------===//

class Galois_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Galois_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Galois_GF8Type : Galois_Type<"GF8", "gf8"> {
    let summary = "Galois GF(2^8) type";
    let description = [{
      This is a custom type in the Galois dialect representing elements of GF(2^8).
      It ensures that operations within this dialect receive valid 8-bit values
      and prevents invalid input types.
    }];

    let parameters = (ins "unsigned":$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // GALOIS_TYPES
