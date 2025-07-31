// module {
//   func.func @main() -> (i32) {
//     // 1) Allocate memory for input and output columns
//     %col = memref.alloc() : memref<4xi32>
//     %out = memref.alloc() : memref<4xi32>

//     // 2) Constants for the column
//     %a = arith.constant 0xd4 : i32
//     %b = arith.constant 0xbf : i32
//     %c = arith.constant 0x5d : i32
//     %d = arith.constant 0x30 : i32

//     // 3) Store column values in the input memref
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %c2 = arith.constant 2 : index
//     %c3 = arith.constant 3 : index
//     memref.store %a, %col[%c0] : memref<4xi32>
//     memref.store %b, %col[%c1] : memref<4xi32>
//     memref.store %c, %col[%c2] : memref<4xi32>
//     memref.store %d, %col[%c3] : memref<4xi32>

//     // 4) Call MixColumns (memory-based)
//     galois.mix_columns %col into %out
//       : memref<4xi32>, memref<4xi32>

//     // 5) Load results from the output memref
//     %r0 = memref.load %out[%c0] : memref<4xi32>
//     %r1 = memref.load %out[%c1] : memref<4xi32>
//     %r2 = memref.load %out[%c2] : memref<4xi32>
//     %r3 = memref.load %out[%c3] : memref<4xi32>

//     // 6) Return results
//     // func.return %r0, %r1, %r2, %r3 : i32, i32, i32, i32
//     %out1 = arith.xori %r0, %r1 : i32
//     %out2 = arith.xori %r2, %r3 : i32
//     %final = arith.xori %out1, %out2 : i32
//     func.return %final : i32 //return single result for flat file testing in gem5 
//   }
// }

module {
  func.func @main() -> (i32) {
    // 1) Allocate memory for input and output states
    %state = memref.alloca() : memref<16xi32>
    %out   = memref.alloca() : memref<16xi32>

    // 2) Constants for the state (row-major 4x4)
    %c0  = arith.constant 0xd4 : i32
    %c1  = arith.constant 0xbf : i32
    %c2  = arith.constant 0x5d : i32
    %c3  = arith.constant 0x30 : i32
    %c4  = arith.constant 0xe0 : i32
    %c5  = arith.constant 0xb4 : i32
    %c6  = arith.constant 0x52 : i32
    %c7  = arith.constant 0xae : i32
    %c8  = arith.constant 0xb8 : i32
    %c9  = arith.constant 0x41 : i32
    %c10 = arith.constant 0x11 : i32
    %c11 = arith.constant 0xf1 : i32
    %c12 = arith.constant 0x1e : i32
    %c13 = arith.constant 0x27 : i32
    %c14 = arith.constant 0x98 : i32
    %c15 = arith.constant 0xe5 : i32

    // 3) Store constants into state
    %i0  = arith.constant 0  : index
    %i1  = arith.constant 1  : index
    %i2  = arith.constant 2  : index
    %i3  = arith.constant 3  : index
    %i4  = arith.constant 4  : index
    %i5  = arith.constant 5  : index
    %i6  = arith.constant 6  : index
    %i7  = arith.constant 7  : index
    %i8  = arith.constant 8  : index
    %i9  = arith.constant 9  : index
    %i10 = arith.constant 10 : index
    %i11 = arith.constant 11 : index
    %i12 = arith.constant 12 : index
    %i13 = arith.constant 13 : index
    %i14 = arith.constant 14 : index
    %i15 = arith.constant 15 : index

    memref.store %c0,  %state[%i0]  : memref<16xi32>
    memref.store %c1,  %state[%i1]  : memref<16xi32>
    memref.store %c2,  %state[%i2]  : memref<16xi32>
    memref.store %c3,  %state[%i3]  : memref<16xi32>
    memref.store %c4,  %state[%i4]  : memref<16xi32>
    memref.store %c5,  %state[%i5]  : memref<16xi32>
    memref.store %c6,  %state[%i6]  : memref<16xi32>
    memref.store %c7,  %state[%i7]  : memref<16xi32>
    memref.store %c8,  %state[%i8]  : memref<16xi32>
    memref.store %c9,  %state[%i9]  : memref<16xi32>
    memref.store %c10, %state[%i10] : memref<16xi32>
    memref.store %c11, %state[%i11] : memref<16xi32>
    memref.store %c12, %state[%i12] : memref<16xi32>
    memref.store %c13, %state[%i13] : memref<16xi32>
    memref.store %c14, %state[%i14] : memref<16xi32>
    memref.store %c15, %state[%i15] : memref<16xi32>

    // 4) Loop through the 4 columns (each column has 4 rows)
    %c4_idx = arith.constant 4 : index
    scf.for %col = %i0 to %c4_idx step %i1 {
      %col_mem = memref.alloca() : memref<4xi32>
      %out_col = memref.alloca() : memref<4xi32>

      %base = arith.muli %col, %c4_idx : index
      %row0 = arith.addi %base, %i0 : index
      %row1 = arith.addi %base, %i1 : index
      %row2 = arith.addi %base, %i2 : index
      %row3 = arith.addi %base, %i3 : index

      %v0 = memref.load %state[%row0] : memref<16xi32>
      %v1 = memref.load %state[%row1] : memref<16xi32>
      %v2 = memref.load %state[%row2] : memref<16xi32>
      %v3 = memref.load %state[%row3] : memref<16xi32>

      memref.store %v0, %col_mem[%i0] : memref<4xi32>
      memref.store %v1, %col_mem[%i1] : memref<4xi32>
      memref.store %v2, %col_mem[%i2] : memref<4xi32>
      memref.store %v3, %col_mem[%i3] : memref<4xi32>

      galois.mix_columns %col_mem into %out_col
        : memref<4xi32>, memref<4xi32>

      %o0 = memref.load %out_col[%i0] : memref<4xi32>
      %o1 = memref.load %out_col[%i1] : memref<4xi32>
      %o2 = memref.load %out_col[%i2] : memref<4xi32>
      %o3 = memref.load %out_col[%i3] : memref<4xi32>

      memref.store %o0, %out[%row0] : memref<16xi32>
      memref.store %o1, %out[%row1] : memref<16xi32>
      memref.store %o2, %out[%row2] : memref<16xi32>
      memref.store %o3, %out[%row3] : memref<16xi32>

      // memref.dealloc %col_mem : memref<4xi32>
      // memref.dealloc %out_col : memref<4xi32>
    }

    // 5) XOR reduction using loop-carried value
    %r16 = arith.constant 16 : index
    %zero = arith.constant 0 : i32
    %final = scf.for %j = %i0 to %r16 step %i1 iter_args(%acc = %zero) -> i32 {
      %val = memref.load %out[%j] : memref<16xi32>
      %new_acc = arith.xori %acc, %val : i32
      scf.yield %new_acc : i32
    }

    func.return %final : i32
  }
}
