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

def Galois_AddOp : Galois_Op<"add", [Pure]> {
    let summary = "Addition in GF(2^8) using XOR.";
    let description = [{}];
    let hasVerifier = 1;
    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def Galois_MulOp : Galois_Op<"mul", [Pure]> {
    let summary = "Multiplication in GF(2^8)";
    let description = [{
        Performs multiplication in GF(2^8) via log/antilog table lookups.
        The op takes two GF field elements and produces a GF field element result.
    }];
    let hasVerifier = 1;
    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def Galois_InvOp : Galois_Op<"inv", [Pure]> {
    let summary = "Multiplicative inverse in GF(2^8)";
    let description = [{
        Computes the multiplicative inverse of a non‑zero GF(2^8) element
        by performing a log lookup, subtracting from 255, then doing an antilog lookup.
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}

def Galois_SBoxOp : Galois_Op<"sbox", [Pure]> {
    let summary = "AES S‑Box byte substitution in GF(2^8)";
    let description = [{
        Replaces an input byte x with SBox[x], where SBox is the
        standard 256‑entry AES substitution box.
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}

def Galois_LFSRStepOp : Galois_Op<"lfsr_step", [Pure]> {
    let summary = "Compute the next state of an N‑bit LFSR given the current state";
    let description = [{
        Takes a single integer state and two attributes:
        - `width` : IntegerAttr, the bit‑width of the register (≤32).
        - `taps`  : ArrayAttr of IntegerAttr, listing bit positions whose XOR 
                    forms the feedback bit.
        Returns the next state.
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);
    let hasVerifier = 1;
    let assemblyFormat = "$input attr-dict `:` type($res)";
}

def Galois_RSEncodeOp : Galois_Op<"rs_encode", [Pure]> {
    let summary = "Systematic Reed–Solomon encoder over GF(2^8)";
    let description = [{
        Given a message of length k, computes the n‑symbol codeword of
        length k+(n−k), appending parity symbols so that c(x) ≡ 0 mod g(x).
    }];

    let arguments = (ins Variadic<I32>:$message);
    let results = (outs Variadic<I32>:$codeword);
    let hasVerifier = 1;
    let assemblyFormat = "$message attr-dict `:` type($message) `->` type($codeword)";
}

def Galois_RSDecodeOp : Galois_Op<"rs_decode", [Pure]> {
    let summary = "Reed–Solomon decoder for systematic code (no errors)";
    let description = [{
        Given an (k+nsym)-symbol codeword, returns the first k message symbols.
        Assumes codeword is valid (no errors) under the generator polynomial.
    }];

    let arguments = (ins Variadic<I32>:$codeword);
    let results = (outs Variadic<I32>:$message);
    let hasVerifier = 1;
    let assemblyFormat = " $codeword attr-dict `:` type($codeword) `->`type($message)";
}

#endif // GALOIS_OPS
