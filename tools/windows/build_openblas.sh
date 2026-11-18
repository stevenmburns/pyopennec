#!/usr/bin/env bash
# A static OpenBLAS for the MinGW (MSYS2 UCRT64) builds of onec and _porty.
#
#   tools/windows/build_openblas.sh OUTDIR win32|omp
#       -> OUTDIR/libopenblas.a, OUTDIR/LICENSE.OpenBLAS, OUTDIR/LICENSE.LAPACK
#
# win32  OpenBLAS's own Win32 thread pool (USE_OPENMP=0): the fill's OpenMP
#        team and the factor's pool are two sets of threads.
# omp    the factor runs on the OpenMP team (USE_OPENMP=1): one set of threads,
#        and none of blas_server_win32.c's exit-time TerminateThread
#        (OpenMathLib/OpenBLAS#6262).
#
# DYNAMIC_ARCH, so one library serves any x86-64 CPU. The list stops at
# HASWELL, as momwire's static build does: SKYLAKEX's kernels would be the only
# ones no CI runner here executes (the runners are AVX2), and AVX-512 bought the
# LU no speed in momwire's measurements. NOFORTRAN=1 takes OpenBLAS's C
# translation of LAPACK, so no libgfortran or libquadmath is linked.
set -euo pipefail
out=$(realpath -m "${1:?usage: build_openblas.sh OUTDIR win32|omp}")
variant=${2:?usage: build_openblas.sh OUTDIR win32|omp}
TAG=v0.3.34
COMMIT=e0166008be8e466242aa76b2ff75ce3f0fbf574a
case "$variant" in
  win32) use_openmp=0 ;;
  omp) use_openmp=1 ;;
  *) echo "unknown variant $variant" >&2; exit 2 ;;
esac
mkdir -p "$out"
src="$out/src"
rm -rf "$src"
git clone -q --depth 1 --branch "$TAG" https://github.com/OpenMathLib/OpenBLAS "$src"
got=$(git -C "$src" rev-parse HEAD)
if [ "$got" != "$COMMIT" ]; then
  echo "OpenBLAS $TAG is $got, expected $COMMIT" >&2
  exit 1
fi
cd "$src"
CFLAGS="-Wno-uninitialized -Wa,-muse-unaligned-vector-move" make -j"$(nproc)" \
  BUFFERSIZE=20 DYNAMIC_ARCH=1 QUIET_MAKE=1 USE_THREAD=1 USE_OPENMP="$use_openmp" \
  NUM_THREADS=64 DYNAMIC_LIST="PRESCOTT NEHALEM SANDYBRIDGE HASWELL" BINARY=64 \
  TARGET=PRESCOTT NOFORTRAN=1 NO_SHARED=1 NO_LAPACKE=1 > "$out/build.log" 2>&1 || {
    tail -60 "$out/build.log"; exit 1; }
cp "$(ls libopenblas*.a | head -1)" "$out/libopenblas.a"
cp LICENSE "$out/LICENSE.OpenBLAS"
cp lapack-netlib/LICENSE "$out/LICENSE.LAPACK"
for s in zgetrf_ zgetrs_ openblas_get_config; do
  nm -g --defined-only "$out/libopenblas.a" 2>/dev/null | grep -q " T $s\$" || {
    echo "libopenblas.a does not define $s" >&2; exit 1; }
done
if nm "$out/libopenblas.a" 2>/dev/null | grep -qi '_gfortran\|quadmath'; then
  echo "libopenblas.a references libgfortran/libquadmath" >&2
  exit 1
fi
ls -l "$out/libopenblas.a"
