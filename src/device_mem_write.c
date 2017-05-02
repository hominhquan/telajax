#include "telajax.h"

int
telajax_device_mem_write(device_t* device, mem_t device_mem, void* host_mem, size_t size)
{
	int err = clEnqueueWriteBuffer(
		device->_queue,
		(cl_mem)device_mem,
		CL_TRUE, 0, size,
		host_mem, 0, NULL, NULL);

	assert(!err);

	return err;
}
