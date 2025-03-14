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

def Galois_CustomType : Galois_Type<"Custom", "custom"> {
    let summary = "Galois custom type";
    let description = "Custom type in galois dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // GALOIS_TYPES
