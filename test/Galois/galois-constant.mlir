// RUN: mlir-opt %s --mlir-print-op-generic | FileCheck %s

module {
  %c = galois.constant <42> : !galois.gf8
}


module {
  %c = galois.constant <256> : !galois.gf8
}
