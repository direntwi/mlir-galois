// RUN: mlir-opt %s --load-pass-plugin=%galois_libs/GaloisPlugin%shlibext --pass-pipeline="builtin.module(galois-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
