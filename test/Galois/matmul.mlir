module {
  func.func @testMatMul() -> (i32, i32, i32, i32) {
    // A = [1, 2; 3, 4], B = [5, 6; 7, 8]
    %a0 = arith.constant 1 : i32
    %a1 = arith.constant 2 : i32
    %a2 = arith.constant 3 : i32
    %a3 = arith.constant 4 : i32
    %b0 = arith.constant 5 : i32
    %b1 = arith.constant 6 : i32
    %b2 = arith.constant 7 : i32
    %b3 = arith.constant 8 : i32

    %r0, %r1, %r2, %r3 = galois.matmul 
      [%a0, %a1, %a2, %a3] by [%b0, %b1, %b2, %b3] 
      {rowsA = 2 : i32, colsA = 2 : i32, colsB = 2 : i32} 
      : (i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32)

    return %r0, %r1, %r2, %r3 : i32, i32, i32, i32
  }
}
