module {
  // An example function using galois.lfsr_step
  func.func @testLFSR() -> i32 {
    %a = arith.constant 905 : i32
    %c = galois.lfsr_step %a {width = 8, taps = [7,5,4,3]} : i32
    return %c : i32
  }
}
