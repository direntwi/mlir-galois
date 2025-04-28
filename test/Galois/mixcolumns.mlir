module {
  func.func @testMixColumns() -> (i32, i32, i32, i32) {
    %a = arith.constant 0xd4 : i32
    %b = arith.constant 0xbf : i32
    %c = arith.constant 0x5d : i32
    %d = arith.constant 0x30 : i32

    %r0, %r1, %r2, %r3 = galois.mix_columns %a, %b, %c, %d
      : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

    return %r0, %r1, %r2, %r3 : i32, i32, i32, i32
  }
}
