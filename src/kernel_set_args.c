#include "telajax.h"

int
telajax_kernel_set_args(int num_args, size_t* args_size, void** args, kernel_t* kernel)
{
	int err = 0;
	for(int i = 0; i < num_args; i++){
		err |= clSetKernelArg(kernel->_kernel, i, args_size[i], args[i]);
	}
	assert(!err);
	return err;
}
