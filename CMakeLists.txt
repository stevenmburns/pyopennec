cmake_minimum_required(VERSION 3.26)
# C only: the package is OpenNEC (C) plus pyopennec._porty (C on the CPython
# Limited API), so no C++ compiler is ever involved.
project(pyopennec LANGUAGES C)

set(OPENNEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opennec)
if(NOT EXISTS ${OPENNEC_DIR}/src/opennec.h)
  message(FATAL_ERROR "opennec/ submodule missing: git submodule update --init")
endif()

# Development.SABIModule (CMake >= 3.26) builds _porty against the Limited
# API, so one cp312-abi3 wheel serves every CPython >= 3.12.
find_package(Python 3.12 COMPONENTS Interpreter Development.SABIModule REQUIRED)

# Every OpenNEC library source except the CLI driver (main.c) and the
# MSVC-only compat_time.c: the same set the Makefile archives into libonec.a.
file(GLOB OPENNEC_SOURCES ${OPENNEC_DIR}/src/*.c ${OPENNEC_DIR}/src/import-export/*.c)
list(FILTER OPENNEC_SOURCES EXCLUDE REGEX "/(main|compat_time)\\.c$")

add_library(opennec_core STATIC ${OPENNEC_SOURCES})
set_target_properties(opennec_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_STANDARD 11
  C_EXTENSIONS ON)
target_include_directories(opennec_core PUBLIC ${OPENNEC_DIR} ${OPENNEC_DIR}/src)
# OpenNEC's own -O2; asserts stay compiled in (porty refuses load types
# outside 0-5 before the engine sees them, which keeps calculations.c's
# load-type assert unreachable).
target_compile_options(opennec_core PRIVATE -O2 -Wno-unused-parameter -Wno-unused-function
  -Wno-format-truncation -Wno-stringop-truncation -U NDEBUG)

find_package(OpenMP COMPONENTS C)
if(OpenMP_C_FOUND)
  target_link_libraries(opennec_core PUBLIC OpenMP::OpenMP_C)
endif()

find_package(PkgConfig)
set(PYOPENNEC_BACKEND "original")
if(PkgConfig_FOUND)
  pkg_check_modules(OPENBLAS IMPORTED_TARGET openblas)
endif()
if(OPENBLAS_FOUND)
  find_library(LAPACK_LIB lapack)
  find_library(GFORTRAN_LIB gfortran)
  target_link_libraries(opennec_core PUBLIC PkgConfig::OPENBLAS)
  if(LAPACK_LIB)
    target_link_libraries(opennec_core PUBLIC ${LAPACK_LIB})
  endif()
  if(GFORTRAN_LIB)
    target_link_libraries(opennec_core PUBLIC ${GFORTRAN_LIB})
  endif()
  target_compile_definitions(opennec_core PRIVATE HAVE_OPENBLAS)
  set(PYOPENNEC_BACKEND "openblas")
  set(PYOPENNEC_LAPACK_DEFS HAVE_OPENBLAS)
else()
  target_compile_definitions(opennec_core PRIVATE USE_ORIGINAL_MATRIX)
endif()
target_link_libraries(opennec_core PUBLIC m)
message(STATUS "pyopennec: matrix backend ${PYOPENNEC_BACKEND}, OpenMP ${OpenMP_C_FOUND}")

# pyopennec._porty: the N-port admittance interface.  It calls LAPACK itself
# (multi-RHS zgetrs), so it needs the same backend macro the engine was
# built with.
Python_add_library(_porty MODULE USE_SABI 3.12 WITH_SOABI csrc/porty.c)
set_target_properties(_porty PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_link_libraries(_porty PRIVATE opennec_core)
target_compile_definitions(_porty PRIVATE ${PYOPENNEC_LAPACK_DEFS}
  PYOPENNEC_BACKEND="${PYOPENNEC_BACKEND}"
  PYOPENNEC_OPENMP=$<BOOL:${OpenMP_C_FOUND}>)
target_compile_options(_porty PRIVATE -Wall -Wextra -Wno-unused-parameter)
install(TARGETS _porty DESTINATION pyopennec)
