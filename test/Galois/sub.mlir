module {
  func.func @testSub() -> i32 {
    %c30 = arith.constant 30 : i32
    %c10 = arith.constant 10 : i32
    %result = galois.sub %c30, %c10 : i32
    return %result : i32
  }
}
