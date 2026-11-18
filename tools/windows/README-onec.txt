onec.exe: OpenNEC's console engine for 64-bit Windows, as one file.

Built with MinGW-w64 (MSYS2 UCRT64). OpenBLAS, the OpenMP runtime and the GCC
runtime are linked in; the program needs only Windows itself and the Universal
C Runtime (part of Windows 10 and later).

  onec.exe -i antenna.nec -o antenna.out

onec.exe       runs on any x86-64 CPU
onec-avx2.exe  the same engine compiled for AVX2 and FMA (2013 and later CPUs)
Both choose OpenBLAS's kernels for the CPU at run time.

Threads
  set OMP_NUM_THREADS=8        matrix fill and factor (see BUILD-INFO.txt)
  set OPENBLAS_NUM_THREADS=8   the factor alone, when set
Without them every logical processor is used.

See LICENSE.OpenNEC.txt and THIRD-PARTY-NOTICES.txt.
