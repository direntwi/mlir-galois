module {
  // An example function using galois.inv with a non-zero argument.
  func.func @testInv() -> i32 {
    %a = arith.constant 5 : i32
    %c = galois.inv %a : i32
    return %c : i32
  }
}
