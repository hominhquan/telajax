#include "telajax.h"

static int telajax_finalized = 0;

/**
 * Return 1 if telajax is already finalized, 0 otherwise
 */
int
telajax_is_finalized()
{
	return telajax_finalized;
}

/**
 * Wait for termination of all on-flight kernels and finalize telajax
 * @return 0 on success, -1 otherwise
 */
int
telajax_device_finalize(device_t* device)
{
	int finalized = __sync_val_compare_and_swap(&telajax_finalized, 0, 1);

	if(finalized == 0){
		clFinish(device->_queue);

		clReleaseCommandQueue(device->_queue);
		clReleaseContext(device->_context);
		clReleaseDevice(device->_device_id);
	}

	return 0;
}

