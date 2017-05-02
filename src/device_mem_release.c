#include "telajax.h"

int
telajax_device_mem_release(mem_t device_mem)
{
	clReleaseMemObject((cl_mem)device_mem);
	return 0;
}
