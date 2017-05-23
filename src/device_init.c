/* Copyright (c) 2017 Kalray

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "telajax.h"

static int telajax_initialized = 0;

/**
 * Return 1 if telajax is already initialized, 0 otherwise
 */
int
telajax_is_initialized()
{
	return telajax_initialized;
}

static cl_device_type
convert_string_device_type(const char* device_type_string)
{
	if(!strcmp(device_type_string, "CPU")) return CL_DEVICE_TYPE_CPU;
	if(!strcmp(device_type_string, "GPU")) return CL_DEVICE_TYPE_GPU;
	if(!strcmp(device_type_string, "ACCELERATOR")) return CL_DEVICE_TYPE_ACCELERATOR;

	return CL_DEVICE_TYPE_ALL;
}

device_t
telajax_device_init(int argc, char** argv, int* error)
{
	device_t device;
	int err = 0;

	int initialized = __sync_val_compare_and_swap(&telajax_initialized, 0, 1);

	if(initialized == 0){
		// User can set device type by setting env var for example
		// TELAJAX_DEVICE_TYPE=ACCELERATOR
		cl_device_type device_type =  (getenv("TELAJAX_DEVICE_TYPE") == NULL) ?
			CL_DEVICE_TYPE_ACCELERATOR :
			convert_string_device_type(getenv("TELAJAX_DEVICE_TYPE"));

		err = clGetPlatformIDs(1, &device._platform, NULL);
		assert(!err);

		err = clGetDeviceIDs(device._platform, device_type, 1, &device._device_id, NULL);
		assert(!err);

		device._context = clCreateContext(0, 1, &device._device_id, NULL, NULL, &err);
		assert(device._context);

		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

		if(getenv("TELAJAX_OOO") != NULL){
			if(!strcmp(getenv("TELAJAX_OOO"), "1")){
				properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
			}
		}

		device._queue = clCreateCommandQueue(device._context, device._device_id,
			properties, &err);
		assert(!err);
		assert(device._queue);
	}

	if(error) *error = err;

	return device;
}

