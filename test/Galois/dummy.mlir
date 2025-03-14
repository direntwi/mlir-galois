// RUN: galois-opt %s | galois-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = galois.foo %{{.*}} : i32
        %res = galois.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @galois_types(%arg0: !galois.custom<"10">)
    func.func @galois_types(%arg0: !galois.custom<"10">) {
        return
    }
}
