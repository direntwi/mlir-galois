get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRGalois
        MLIRConvertToLLVMPass
        )
add_llvm_executable(galois-opt galois-opt.cpp)

llvm_update_compile_flags(galois-opt)
target_link_libraries(galois-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(galois-opt)
