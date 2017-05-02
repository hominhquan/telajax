#include "telajax.h"

int
telajax_kernel_set_callback(
	void (CL_CALLBACK *pfn_event_notify)(
		cl_event event,
		cl_int event_command_exec_status,
		void* user_data
	),
	void* user_data, kernel_t* kernel)
{
	int err = 0;

	err = clSetEventCallback(kernel->_event, CL_COMPLETE, pfn_event_notify, user_data);
	assert(!err);

	return (err == CL_SUCCESS) ? 0 : -1;
}
