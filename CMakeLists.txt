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
# input.c calls strcasestr, which glibc declares only under _GNU_SOURCE; GCC
# 14 (manylinux_2_28's toolset) makes the implicit declaration an error.
if(NOT APPLE)
  target_compile_definitions(opennec_core PRIVATE _GNU_SOURCE)
endif()
# OpenNEC's own -O2; asserts stay compiled in (porty refuses load types
# outside 0-5 before the engine sees them, which keeps calculations.c's
# load-type assert unreachable).
target_compile_options(opennec_core PRIVATE -O2 -Wno-unused-parameter -Wno-unused-function
  -Wno-format-truncation -U NDEBUG)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_compile_options(opennec_core PRIVATE -Wno-stringop-truncation)
endif()

# --- OpenMP ------------------------------------------------------------------
# A redistributable wheel passes -DPYOPENNEC_REQUIRE_OPENMP=ON so a build that
# lost OpenMP fails instead of shipping a single-threaded fill.  On macOS the
# runtime is Homebrew's libomp (Apple clang has none): pass
# -DOpenMP_ROOT=$(brew --prefix libomp).  The wheel links it by its absolute
# install name and does not vendor it, so every extension in the process
# shares one libomp image (see pyproject.toml's cibuildwheel sections).
option(PYOPENNEC_REQUIRE_OPENMP "fail the build when OpenMP is not found" OFF)
find_package(OpenMP COMPONENTS C)
if(OpenMP_C_FOUND)
  target_link_libraries(opennec_core PUBLIC OpenMP::OpenMP_C)
elseif(PYOPENNEC_REQUIRE_OPENMP)
  message(FATAL_ERROR "pyopennec: OpenMP not found and PYOPENNEC_REQUIRE_OPENMP=ON "
    "(macOS: brew install libomp and pass -DOpenMP_ROOT=$(brew --prefix libomp))")
endif()

# --- BLAS / LAPACK -------------------------------------------------------------
# PYOPENNEC_BLAS picks the LAPACK behind the LU factor and solve:
#   auto            pkg-config openblas, else Accelerate on Apple, else
#                   OpenNEC's own matrix code (the only value that may fall back)
#   openblas        a system OpenBLAS via pkg-config (+ liblapack if present)
#   scipy-openblas32  the scipy-openblas32 PyPI wheel's libscipy_openblas
#                   (SCIPY_OPENBLAS32_DIR = its package dir, else imported
#                   from the build interpreter)
#   accelerate      Apple's Accelerate framework (new LAPACK interface)
#   original        OpenNEC's own LU, no LAPACK (far slower)
# Any explicit value other than `original` is a hard requirement: if it cannot
# be satisfied the configure fails, so a wheel never ships the fallback.
set(PYOPENNEC_BLAS "auto" CACHE STRING "auto|openblas|scipy-openblas32|accelerate|original")
set(_blas ${PYOPENNEC_BLAS})
if(_blas STREQUAL "auto")
  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    pkg_check_modules(OPENBLAS IMPORTED_TARGET openblas)
  endif()
  if(OPENBLAS_FOUND)
    set(_blas openblas)
  elseif(APPLE)
    set(_blas accelerate)
  else()
    set(_blas original)
  endif()
endif()

set(PYOPENNEC_LAPACK_DEFS "")
if(_blas STREQUAL "openblas")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
  find_library(LAPACK_LIB lapack)
  find_library(GFORTRAN_LIB gfortran)
  target_link_libraries(opennec_core PUBLIC PkgConfig::OPENBLAS)
  if(LAPACK_LIB)
    target_link_libraries(opennec_core PUBLIC ${LAPACK_LIB})
  endif()
  if(GFORTRAN_LIB)
    target_link_libraries(opennec_core PUBLIC ${GFORTRAN_LIB})
  endif()
  set(PYOPENNEC_LAPACK_DEFS HAVE_OPENBLAS)
elseif(_blas STREQUAL "scipy-openblas32")
  if(NOT SCIPY_OPENBLAS32_DIR)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import os, scipy_openblas32 as o; print(os.path.dirname(o.__file__))"
      OUTPUT_VARIABLE SCIPY_OPENBLAS32_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rc ERROR_QUIET)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "pyopennec: PYOPENNEC_BLAS=scipy-openblas32 but SCIPY_OPENBLAS32_DIR "
        "is unset and the build interpreter cannot import scipy_openblas32")
    endif()
  endif()
  find_library(SCIPY_OPENBLAS_LIB scipy_openblas PATHS ${SCIPY_OPENBLAS32_DIR}/lib NO_DEFAULT_PATH)
  if(NOT SCIPY_OPENBLAS_LIB)
    message(FATAL_ERROR "pyopennec: libscipy_openblas not found under ${SCIPY_OPENBLAS32_DIR}/lib")
  endif()
  target_link_libraries(opennec_core PUBLIC ${SCIPY_OPENBLAS_LIB})
  # scipy-openblas is built with BLAS_SYMBOL_PREFIX=scipy_ (its pkg-config
  # Cflags say so), so its LAPACK is scipy_zgetrf_ / scipy_zgetrs_.  Its own
  # headers apply the prefix by macro; OpenNEC and porty declare the Fortran
  # entry points themselves, so the same rename is done here, for both
  # targets, without touching opennec/.  These two are the only LAPACK
  # symbols either calls (matrix.c factor_matrix / solve, porty.c
  # engine_factor / engine_solve).
  set(PYOPENNEC_LAPACK_DEFS HAVE_OPENBLAS zgetrf_=scipy_zgetrf_ zgetrs_=scipy_zgetrs_)
elseif(_blas STREQUAL "accelerate")
  if(NOT APPLE)
    message(FATAL_ERROR "pyopennec: PYOPENNEC_BLAS=accelerate is macOS only")
  endif()
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(opennec_core PUBLIC ${ACCELERATE_FRAMEWORK})
  # the same macros OpenNEC's Makefile sets for BACKEND=accelerate
  set(PYOPENNEC_LAPACK_DEFS HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
elseif(_blas STREQUAL "original")
  set(PYOPENNEC_LAPACK_DEFS USE_ORIGINAL_MATRIX)
else()
  message(FATAL_ERROR "pyopennec: unknown PYOPENNEC_BLAS '${PYOPENNEC_BLAS}'")
endif()
set(PYOPENNEC_BACKEND ${_blas})
target_compile_definitions(opennec_core PRIVATE ${PYOPENNEC_LAPACK_DEFS})
target_link_libraries(opennec_core PUBLIC m)
message(STATUS "pyopennec: matrix backend ${PYOPENNEC_BACKEND} (PYOPENNEC_BLAS=${PYOPENNEC_BLAS}), OpenMP ${OpenMP_C_FOUND}")

# pyopennec._porty: the N-port admittance interface.  It calls LAPACK itself
# (multi-RHS zgetrs), so it needs the same backend macro the engine was
# built with.
Python_add_library(_porty MODULE USE_SABI 3.12 WITH_SOABI csrc/porty.c)
# INSTALL_RPATH_USE_LINK_PATH keeps a non-system BLAS (scipy-openblas32's
# lib dir) findable from an unrepaired local build; auditwheel / delocate
# rewrite it when they vendor the library into a wheel.
set_target_properties(_porty PROPERTIES C_STANDARD 11 C_EXTENSIONS ON
  INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(_porty PRIVATE opennec_core)
target_compile_definitions(_porty PRIVATE ${PYOPENNEC_LAPACK_DEFS}
  PYOPENNEC_BACKEND="${PYOPENNEC_BACKEND}"
  PYOPENNEC_OPENMP=$<BOOL:${OpenMP_C_FOUND}>)
target_compile_options(_porty PRIVATE -Wall -Wextra -Wno-unused-parameter)
install(TARGETS _porty DESTINATION pyopennec)
