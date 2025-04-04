add_mlir_dialect(GaloisOps galois)
add_mlir_doc(GaloisDialect GaloisDialect Galois/ -gen-dialect-doc)
add_mlir_doc(GaloisOps GaloisOps Galois/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GaloisPasses.td)
mlir_tablegen(GaloisPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRGaloisPassesIncGen)
