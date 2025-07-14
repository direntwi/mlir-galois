module {
  // An example function using galois.sbox
  func.func @main() -> i32 {
    %a = arith.constant 5 : i32
    %c = galois.sbox %a : i32
    func.return %c : i32
  }
}
