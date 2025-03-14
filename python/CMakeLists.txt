include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_galois`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_galois.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(GaloisPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT GaloisPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_galois"
  TD_FILE dialects/GaloisOps.td
  SOURCES
    dialects/galois_pybind11.py
    dialects/galois_nanobind.py
  DIALECT_NAME galois)


declare_mlir_python_extension(GaloisPythonSources.Pybind11Extension
  MODULE_NAME _galoisDialectsPybind11
  ADD_TO_PARENT GaloisPythonSources
  SOURCES
    GaloisExtensionPybind11.cpp
  EMBED_CAPI_LINK_LIBS
    GaloisCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(GaloisPythonSources.NanobindExtension
  MODULE_NAME _galoisDialectsNanobind
  ADD_TO_PARENT GaloisPythonSources
  SOURCES
    GaloisExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    GaloisCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(GaloisPythonCAPI
  INSTALL_COMPONENT GaloisPythonModules
  INSTALL_DESTINATION python_packages/galois/mlir_galois/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/galois/mlir_galois/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    GaloisPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(GaloisPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/galois/mlir_galois"
  INSTALL_PREFIX "python_packages/galois/mlir_galois"
  DECLARED_SOURCES
    GaloisPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    GaloisPythonCAPI
  )
