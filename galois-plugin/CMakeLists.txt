add_llvm_library(GaloisPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        galois-plugin.cpp

        DEPENDS
        MLIRGalois
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRGalois
        )
target_include_directories(
        GaloisPlugin
        PRIVATE
        "${GALOIS_BINARY_DIR}/include"
)
