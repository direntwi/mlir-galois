// RUN: mlir-opt %s --load-dialect-plugin=%galois_libs/GaloisPlugin%shlibext --pass-pipeline="builtin.module(galois-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @galois_types(%arg0: !galois.custom<"10">)
  func.func @galois_types(%arg0: !galois.custom<"10">) {
    return
  }
}
