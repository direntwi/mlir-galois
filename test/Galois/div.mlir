module {
  func.func @testDiv() -> i32 {
    %a = arith.constant 0 : i32
    %b = arith.constant 5  : i32
    %r = galois.div %a, %b : i32
    return %r : i32
  }
}
