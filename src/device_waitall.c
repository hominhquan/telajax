#include "telajax.h"

int
telajax_device_waitall(device_t* device)
{
	clFinish(device->_queue);
	return 0;
}
