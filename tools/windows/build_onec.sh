#!/usr/bin/env bash
# onec.exe from the opennec/ submodule, in an MSYS2 UCRT64 shell: one file, with
# OpenBLAS, libgomp, winpthreads and libgcc linked in.
#
#   tools/windows/build_onec.sh OUTDIR LIBOPENBLAS generic|avx2  -> OUTDIR/onec.exe
#
# generic  OpenNEC's C at the x86-64 baseline (SSE2), OpenNEC's own -O2.
# avx2     the same with -mavx2 -mfma. Needs a CPU from 2013 or later.
#          -muse-unaligned-vector-move: GCC on Win64 can emit aligned 32-byte
#          moves to a stack it did not align (GCC bug 54412).
# OpenBLAS picks its kernels at run time in both.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${1:?usage: build_onec.sh OUTDIR LIBOPENBLAS generic|avx2}")
blas=$(realpath "${2:?missing LIBOPENBLAS}")
arch=${3:?missing generic|avx2}
src="$here/opennec/src"
case "$arch" in
  generic) archflags=() ;;
  avx2) archflags=(-mavx2 -mfma -Wa,-muse-unaligned-vector-move) ;;
  *) echo "unknown arch $arch" >&2; exit 2 ;;
esac
mkdir -p "$out/obj"
cflags=(-I"$here/opennec" -I"$src" -O2 -fopenmp -DHAVE_OPENBLAS "${archflags[@]}")
objs=()
for s in "$src"/*.c "$src"/import-export/*.c; do
  # compat_time.c is MSVC's clock_gettime; MinGW has its own
  case "$(basename "$s")" in compat_time.c) continue ;; esac
  o="$out/obj/$(basename "${s%.c}").o"
  gcc "${cflags[@]}" -c "$s" -o "$o" &
  objs+=("$o")
  while [ "$(jobs -rp | wc -l)" -ge "$(nproc)" ]; do wait -n; done
done
wait
gcc -O2 -c "$here/tools/windows/onec_exit.c" -o "$out/obj/onec_exit.o"
# 64 MB of stack for the main thread: the Linux default is 8 MB, MinGW's 2 MB
gcc "${objs[@]}" "$out/obj/onec_exit.o" -static -fopenmp "$blas" -lm \
  -Wl,--wrap=main -Wl,--wrap=exit -Wl,--stack,67108864 -o "$out/onec.exe"
strip "$out/onec.exe"
ls -l "$out/onec.exe"
