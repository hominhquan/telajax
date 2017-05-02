#include "telajax.h"

int
telajax_kernel_set_dim(
	int work_dim, const size_t* globalSize, const size_t* localSize, kernel_t* kernel)
{
	if(work_dim > 3) return -1;

	kernel->_work_dim = work_dim;
	for(int i = 0; i < work_dim; i++){
		kernel->_globalSize[i] =  globalSize[i];
		kernel->_localSize[i]  =  localSize[i];
	}
	return 0;
}
