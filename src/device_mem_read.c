#include "telajax.h"

int
telajax_device_mem_read(device_t* device, mem_t device_mem, void* host_mem, size_t size)
{
	int err = clEnqueueReadBuffer(
		device->_queue,
		(cl_mem)device_mem,
		CL_TRUE, 0, size,
		host_mem, 0, NULL, NULL);

	assert(!err);

	return err;
}


