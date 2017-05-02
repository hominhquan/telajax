#include "telajax.h"

mem_t
telajax_device_mem_alloc(size_t size, mem_flags_t mem_flags, device_t* device, int* error)
{
	mem_t dev_mem = (mem_t) clCreateBuffer(device->_context, (cl_mem_flags) mem_flags, size, NULL, NULL);
	assert(dev_mem);

	if(error) *error = 0;

	return dev_mem;
}

