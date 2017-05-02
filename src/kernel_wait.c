#include "telajax.h"

int
telajax_kernel_wait(kernel_t* kernel)
{
	return clWaitForEvents(1, &(kernel->_event));
}
