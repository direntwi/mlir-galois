add_mlir_public_c_api_library(GaloisCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRGalois
)
