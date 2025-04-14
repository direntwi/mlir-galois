module {
  // ConstantOp test
  %c = galois.constant <42> : i32

  // FromIntegerOp test - fixed syntax
  %gf8_val = galois.from_integer %c : !galois.gf8

  // ToIntegerOp test - fixed syntax
  %i32_val = galois.to_integer %gf8_val : i32
}

module {
  // ConstantOp test
  %c42 = arith.constant 42 : i32

  // FromIntegerOp test
  %gf8_val = galois.from_integer %c42 : !galois.gf8

  // ToIntegerOp test
  %i32_val = galois.to_integer %gf8_val : i32
}


//Should fail
module {
  // ConstantOp test
  %c42 = arith.constant 256 : i32

  // FromIntegerOp test
  %gf8_val = galois.to_integer %c42 : i32

}


//Should fail
module {
  // ConstantOp test
  %c42 = arith.constant 256 : i32

  // FromIntegerOp test
  %gf8_val = galois.from_integer %c42 : !galois.gf8

}
