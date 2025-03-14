# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_galois.ir import *
from mlir_galois.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_galois.dialects import galois_pybind11 as galois_d
elif sys.argv[1] == "nanobind":
    from mlir_galois.dialects import galois_nanobind as galois_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    galois_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = galois.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: galois.foo %[[C]] : i32
    print(str(module))
