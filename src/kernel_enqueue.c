#include "telajax.h"

int
telajax_kernel_enqueue(kernel_t* kernel, device_t* device)
{
	int err = clEnqueueNDRangeKernel(
		device->_queue,
		kernel->_kernel,
		kernel->_work_dim,
		NULL,  // global_offset
		kernel->_globalSize,
		kernel->_localSize,
		0, NULL,
		&(kernel->_event));

	assert(!err);

	return 0;
}
