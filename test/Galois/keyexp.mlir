// testKeyExp.mlir
module {
  func.func @testKeyExp() -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {  // 16 results
    %k0  = arith.constant 0x00 : i32
    %k1  = arith.constant 0x01 : i32
    %k2  = arith.constant 0x02 : i32
    %k3  = arith.constant 0x03 : i32
    %k4  = arith.constant 0x04 : i32
    %k5  = arith.constant 0x05 : i32
    %k6  = arith.constant 0x06 : i32
    %k7  = arith.constant 0x07 : i32
    %k8  = arith.constant 0x08 : i32
    %k9  = arith.constant 0x09 : i32
    %k10 = arith.constant 0x0a : i32
    %k11 = arith.constant 0x0b : i32
    %k12 = arith.constant 0x0c : i32
    %k13 = arith.constant 0x0d : i32
    %k14 = arith.constant 0x0e : i32
    %k15 = arith.constant 0x0f : i32

    %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7, %n8, %n9, %n10, %n11, %n12, %n13, %n14, %n15 = galois.key_expand 
      %k0, %k1, %k2, %k3, %k4, %k5, %k6, %k7, %k8, %k9, %k10, %k11, %k12, %k13, %k14 , %k15 { round = 1 : i32 }
      : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    return %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7, %n8, %n9, %n10, %n11, %n12, %n13, %n14, %n15 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  }
}
