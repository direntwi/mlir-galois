// testRS-encode.mlir
module {
  func.func @testRSEncode() -> (i32, i32, i32, i32) {
    // Message symbols: 1, 2
    %m0 = arith.constant 1 : i32
    %m1 = arith.constant 2 : i32

    // rs_encode %m0, %m1 {messageLength=2, paritySymbols=2, generatorPoly=[3,1,2]}
    // -> %c0, %c1 (message) , %p0, %p1 (parity)
    %c0, %c1, %p0, %p1 = galois.rs_encode %m0, %m1 
                      {messageLength = 2, paritySymbols = 2, generatorPoly = [3,1,2]}
                      : i32, i32 -> i32, i32, i32, i32

    return %c0, %c1, %p0, %p1 : i32, i32, i32, i32
  }
}
