/* The exit of the Windows onec.exe, linked in with -Wl,--wrap=main,--wrap=exit.
 *
 * A program that links OpenBLAS statically can hang for ever at exit on
 * Windows: the C runtime's exit runs OpenBLAS's shutdown, which waits 50 ms for
 * each pool thread and then TerminateThread()s it, and a thread killed while it
 * holds the loader or heap lock never releases it
 * (OpenMathLib/OpenBLAS#6262). So onec ends the process itself, once its output
 * is on its way, before that shutdown can run. OpenNEC's own sources are not
 * changed.
 *
 * ONEC_EXIT_TRACE=1 says on stderr that this path ran, for the CI gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

int __real_main(int argc, char **argv);
void __wrap_exit(int status);

void __wrap_exit(int status)
{
  fflush(NULL);
  if (getenv("ONEC_EXIT_TRACE")) {
    fprintf(stderr, "onec_exit: TerminateProcess(%d)\n", status);
    fflush(stderr);
  }
  TerminateProcess(GetCurrentProcess(), (UINT)status);
  /* TerminateProcess does not return for the current process */
  for (;;) Sleep(1000);
}

int __wrap_main(int argc, char **argv)
{
  __wrap_exit(__real_main(argc, argv));
  return 0;
}
