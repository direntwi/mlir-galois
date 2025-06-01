module {
  // An example function using galois.mul with non-zero arguments.
  func.func @main() -> i32 {
    %a = arith.constant 5 : i32
    %b = arith.constant 7 : i32
    %c = galois.mul %a, %b : i32
    func.return %c : i32
  }
}
