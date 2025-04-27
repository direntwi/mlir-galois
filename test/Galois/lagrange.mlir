module {
  func.func @test_lagrange_interp() -> (i32, i32) {
    %x0 = arith.constant 1 : i32
    %x1 = arith.constant 2 : i32
    %y0 = arith.constant 3 : i32
    %y1 = arith.constant 5 : i32

    %c0, %c1 = galois.lagrange_interp %x0, %x1, %y0, %y1 : i32, i32, i32, i32 -> i32, i32

    return %c0, %c1 : i32, i32
  }
}
