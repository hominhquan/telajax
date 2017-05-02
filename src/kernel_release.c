#include "telajax.h"

int
telajax_kernel_release(kernel_t* kernel)
{
	clReleaseKernel(kernel->_kernel);
	clReleaseProgram(kernel->_program);
	return 0;
}
