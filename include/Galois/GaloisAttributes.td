//===- GaloisAttributes.td - Galois dialect ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_ATTRIBUTES
#define GALOIS_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "Galois/GaloisDialect.td"

class Galois_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Galois_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Galois_GF8ConstantAttr : Galois_Attr<"GF8Constant", "gf8constant"> {
  let summary = "A constant attribute for GF(2^8) field elements.";
  let description = [{
    This attribute represents an element in GF(2^8). It ensures that the stored
    value is always in the range `[0, 255]` (i.e., it fits within 8 bits).
  }];
  
  let parameters = (ins "IntegerAttr":$value);

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}
#endif // GALOIS_ATTRIBUTES
