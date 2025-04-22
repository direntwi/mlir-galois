module {
  func.func @testRSEncodeAndDecode() -> (i32, i32) {
    
    %m0 = arith.constant 1 : i32
    %m1 = arith.constant 2 : i32
    %c0, %c1, %p0, %p1 = galois.rs_encode %m0, %m1 
                      {messageLength = 2, paritySymbols = 2, generatorPoly = [3,1,2]}
                      : i32, i32 -> i32, i32, i32, i32

    
    %d0, %d1 = galois.rs_decode %c0, %c1, %p0, %p1 
                      {messageLength = 2, paritySymbols = 2, generatorPoly = [3,1,2]}
                      : i32, i32, i32, i32 -> i32, i32

    return %d0, %d1 : i32, i32
  }
}
