// RUN: mlir-opt %s -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// -----

// Valid case (should pass verification)
func.func @valid_add() -> i32 {
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  %result = galois.add %c5, %c10 : i32
  return %result : i32
}

// -----

// Invalid operand type (i16)
func.func @invalid_type_add() -> i32 {
  %c5 = arith.constant 5 : i16  // expected-error {{expects i32 input operands}}
  %c10 = arith.constant 10 : i32
  %result = galois.add %c5, %c10 : i32
  return %result : i32
}

// -----

// Value too low (negative)
func.func @negative_operand_add() -> i32 {
  %c_neg1 = arith.constant -1 : i32  // expected-error {{operand value must be 0-255}}
  %c10 = arith.constant 10 : i32
  %result = galois.add %c_neg1, %c10 : i32
  return %result : i32
}

// -----

// Value too high (256)
func.func @overflow_operand_add() -> i32 {
  %c256 = arith.constant 256 : i32  // expected-error {{operand value must be 0-255}}
  %c10 = arith.constant 10 : i32
  %result = galois.add %c256, %c10 : i32
  return %result : i32
}
