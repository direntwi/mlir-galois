add_mlir_dialect_library(MLIRGalois
        GaloisTypes.cpp
        GaloisDialect.cpp
        GaloisOps.cpp
        GaloisPasses.cpp
        GaloisAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Galois

        DEPENDS
        MLIRGaloisOpsIncGen
        MLIRGaloisPassesIncGen
        MLIRGaloisAttributesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIRTensorDialect
        MLIRUBDialect
        MLIRFuncAllExtensions
        MLIRTensorAllExtensions
        )
