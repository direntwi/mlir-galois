//===- GaloisOps.td - Galois dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_OPS
#define GALOIS_OPS

include "mlir/IR/BuiltinAttributes.td"
include "Galois/GaloisAttributes.td"
include "Galois/GaloisTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// include "mlir/Dialect/Arith/IR/ArithOps.td"

def Galois_FooOp : Galois_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `galois.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = galois.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Galois_ConstantOp : Galois_Op<"constant", [Pure]> {
    let summary = "Creates a constant GF(2^8) value.";
    let description = [{
      This operation defines a constant in GF(2^8) using a GF8ConstantAttr.
    }];

    let arguments = (ins Galois_GF8ConstantAttr:$value);
    let results = (outs I32:$res);

    let assemblyFormat = " $value attr-dict `:` type($res)";
}

def Galois_ToIntegerOp : Galois_Op<"to_integer", [Pure]> {
    let summary = "Converts GF(2^8) type to i32.";
    let description = [{}];
    let arguments = (ins Galois_GF8Type:$input);
    let results = (outs I32:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}

def Galois_FromIntegerOp : Galois_Op<"from_integer", [Pure]> {
    let summary = "Converts i32 to GF(2^8).";
    let description = [{}];
    let arguments = (ins I32:$input);
    let results = (outs Galois_GF8Type:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}
#endif // GALOIS_OPS
