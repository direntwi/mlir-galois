//===- GaloisPsss.td - Galois dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GALOIS_PASS
#define GALOIS_PASS

include "mlir/Pass/PassBase.td"

def GaloisSwitchBarFoo: Pass<"galois-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def ConvertGaloisToArith : Pass<"convert-galois-to-arith"> {
  let summary = "Lower Galois dialect ops to Arith dialect";
  let constructor = "mlir::galois::createConvertGaloisToArithPass()";
  let dependentDialects = ["arith::ArithDialect"];
}

#endif // GALOIS_PASS
