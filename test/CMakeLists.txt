llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GALOIS_TEST_DEPENDS
        FileCheck count not
        galois-capi-test
        galois-opt
        galois-translate
        GaloisPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND GALOIS_TEST_DEPENDS GaloisPythonModules)
endif()

add_lit_testsuite(check-galois "Running the galois regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GALOIS_TEST_DEPENDS}
        )
set_target_properties(check-galois PROPERTIES FOLDER "Tests")

add_lit_testsuites(GALOIS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GALOIS_TEST_DEPENDS})

add_subdirectory(CAPI)
