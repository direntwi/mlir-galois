// module {
//   func.func @main() -> i32 {
  
//     %a0 = arith.constant 1 : i32
//     %a1 = arith.constant 2 : i32
//     %a2 = arith.constant 3 : i32
//     %a3 = arith.constant 4 : i32

//     %b0 = arith.constant 5 : i32
//     %b1 = arith.constant 6 : i32
//     %b2 = arith.constant 7 : i32
//     %b3 = arith.constant 8 : i32

//     // Allocate output buffer
//     %output = memref.alloc() : memref<4xi32>

//     %d0 = arith.constant 0 : index
//     %d1 = arith.constant 1 : index
//     %d2 = arith.constant 2 : index
//     %d3 = arith.constant 3 : index
//     %zero = arith.constant 0 : i32

//     memref.store %zero, %output[%d0] : memref<4xi32>
//     memref.store %zero, %output[%d1] : memref<4xi32>
//     memref.store %zero, %output[%d2] : memref<4xi32>
//     memref.store %zero, %output[%d3] : memref<4xi32>


//     // Call matmul
//     galois.matmul 
//     [%a0, %a1, %a2, %a3 : i32, i32, i32, i32] 
//     by 
//     [%b0, %b1, %b2, %b3 : i32, i32, i32, i32] 
//     into %output 
//     {rowsA = 2 : i32, colsA = 2 : i32, colsB = 2 : i32}
//     : memref<4xi32>

//     // Constants for indices
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %c2 = arith.constant 2 : index
//     %c3 = arith.constant 3 : index

//     // Load output scalars
//     %r0 = memref.load %output[%c0] : memref<4xi32>
//     %r1 = memref.load %output[%c1] : memref<4xi32>
//     %r2 = memref.load %output[%c2] : memref<4xi32>
//     %r3 = memref.load %output[%c3] : memref<4xi32>


//     // func.return %r0, %r1, %r2, %r3 : i32, i32, i32, i32
//     %out = arith.xori %r0, %r1 : i32
//     %out2 = arith.xori %r2, %r3 : i32
//     %final = arith.xori %out, %out2 : i32
//     return %final : i32 //return single result for flat file testing in gem5 

//   }
// }

module {
  func.func @main() -> i32 {
    %zero = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c10 = arith.constant 10 : index

    // Allocate A, B, C on the stack
    %A = memref.alloca() : memref<100xi32>
    %B = memref.alloca() : memref<100xi32>
    %C = memref.alloca() : memref<100xi32>

    %one = arith.constant 1 : i32
    scf.for %i = %c0 to %c100 step %c1 {
      %val = arith.index_cast %i : index to i32
      %val_plus_one = arith.addi %val, %one : i32
      memref.store %val_plus_one, %A[%i] : memref<100xi32>
    }

    %two = arith.constant 2 : i32
    scf.for %i = %c0 to %c100 step %c1 {
      %val = arith.index_cast %i : index to i32
      %val_plus_one = arith.addi %val, %one : i32
      %double = arith.muli %val_plus_one, %two : i32
      memref.store %double, %B[%i] : memref<100xi32>
    }


    // Zero initialize C[i]
    scf.for %i = %c0 to %c100 step %c1 {
      memref.store %zero, %C[%i] : memref<100xi32>
    }

    // Call galois.matmul
    galois.matmul
      [%A : memref<100xi32>]
      by [%B : memref<100xi32>]
      into %C
      {rowsA = 10 : i32, colsA = 10 : i32, colsB = 10 : i32}
      : memref<100xi32>

    // XOR accumulate all values in C
    %acc0 = arith.constant 0 : i32
    %result = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc = %acc0) -> i32 {
      %val = memref.load %C[%i] : memref<100xi32>
      %next = arith.xori %acc, %val : i32
      scf.yield %next : i32
    }

    return %result : i32
  }
}
