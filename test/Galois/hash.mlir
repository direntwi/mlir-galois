module {
  func.func @testHash() -> i32 {
    
    %b0 = arith.constant 1 : i32
    %b1 = arith.constant 2 : i32
    %b2 = arith.constant 3 : i32
    %b3 = arith.constant 4 : i32
    
    %h = galois.hash %b0, %b1, %b2, %b3 { alpha = 5 : i32 } : i32
    return %h : i32
  }
}
