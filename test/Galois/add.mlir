// test-add.mlir
// module {
//   func.func @test_add(%lhs: i32, %rhs: i32) -> i32 {
//     %result = galois.add %lhs, %rhs : i32
//     return %result : i32
//   }
// }
// module {
//     func.func @valid_add() -> i32 {
//     %c5 = arith.constant 5 : i16
//     %c10 = arith.constant 10 : i16
//     %result = galois.add %c5, %c10 : i16
//     return %result : i32
// }
// }

// // Should fail verification
// func.func @invalid() -> i32 {
//   %c300 = arith.constant 30 : i32  // expected-error {{operand value 300 out of range [0,255]}}
//   %c10 = galois.constant <10> : i32
//   %0 = galois.add %c300, %c10 : i32
//   return %0 : i32
// }

module {
  func.func @test() -> i32 {
    %c30 = arith.constant 30 : i32
    %c10 = arith.constant 10 : i32
    %result = galois.add %c30, %c10 : i32
    return %result : i32
  }
}