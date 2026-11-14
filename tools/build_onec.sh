#!/usr/bin/env bash
# Build the onec CLI from the opennec/ submodule into an out-of-tree directory,
# with the flags `make RELEASE=1 OPENMP=1 BACKEND=openblas` uses, so the
# submodule working tree stays clean.  The test suite compares porty
# against this binary at print precision (tests/test_porty_vs_cli.py).
#
#   tools/build_onec.sh <outdir>      -> <outdir>/onec
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/opennec/src"
out="${1:-$here/build/onec}"
mkdir -p "$out/obj"
cflags=(-I"$here/opennec" -I"$src" -O2)
if [[ "$(uname -s)" == Darwin ]]; then
  # Apple clang has no OpenMP of its own: Homebrew's libomp, and Accelerate
  # for LAPACK (OpenNEC's Makefile BACKEND=accelerate), as the macOS wheel
  cc="${CC:-clang}"
  omp="${LIBOMP_PREFIX:-$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)}"
  cflags+=(-Xpreprocessor -fopenmp -I"$omp/include" -DHAVE_ACCELERATE -DACCELERATE_NEW_LAPACK)
  ldflags=(-L"$omp/lib" -lomp -Wl,-rpath,"$omp/lib" -framework Accelerate)
else
  cc="${CC:-gcc}"
  cflags+=(-fopenmp)
  ldflags=(-fopenmp)
fi
if [[ "$(uname -s)" == Darwin ]]; then
  :
elif pkg-config --exists openblas 2>/dev/null; then
  read -r -a ob_c <<<"$(pkg-config --cflags openblas)"
  read -r -a ob_l <<<"$(pkg-config --libs openblas)"
  cflags+=("${ob_c[@]}" -DHAVE_OPENBLAS)
  ldflags+=("${ob_l[@]}" -llapack -lgfortran)
else
  cflags+=(-DUSE_ORIGINAL_MATRIX)
fi
srcs=("$src"/*.c "$src"/import-export/*.c)
objs=()
pids=()
for s in "${srcs[@]}"; do
  case "$(basename "$s")" in compat_time.c) continue ;; esac
  o="$out/obj/$(basename "${s%.c}").o"
  objs+=("$o")
  extra=()
  # glibc declares input.c's strcasestr only under _GNU_SOURCE (an error from
  # GCC 14 on); that file only, since _GNU_SOURCE also exposes glibc's
  # error_t, which collides with OpenNEC's own where errno.h is included
  if [[ "$(uname -s)" != Darwin && "$(basename "$s")" == input.c ]]; then extra=(-D_GNU_SOURCE); fi
  "$cc" "${cflags[@]}" ${extra[@]+"${extra[@]}"} -c "$s" -o "$o" &
  pids+=($!)
  # four compiles at a time: the laptop this was written on has OOM-killed
  # sessions under heavier parallel load
  if (( ${#pids[@]} >= 4 )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p"; done
"$cc" "${objs[@]}" "${ldflags[@]}" -lm -o "$out/onec"
echo "$out/onec"
