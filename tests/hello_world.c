#include <stdio.h>
#include <telajax.h>

#define VEC_LENGTH 16

const char* kernel_ocl_wrapper = "\n" \
"void _hello_world();\n" \
"\n" \
"__kernel void hello_world(){\n" \
"	printf(\"Hello world from %d\\n\", get_global_id(0)); \n" \
"	_hello_world(); \n" \
"}\n" \
"\n";


const char* kernel_code = "\n" \
"#include <stdio.h> \n " \
"\n" \
"void _hello_world(){ \n" \
"} \n" ;


void pfn_event_notify(
		cl_event event,
		cl_int event_command_exec_status,
		void* exec_time)
{
	if(exec_time){
		cl_ulong time_start = 0, time_end = 0;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
			sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
			sizeof(time_end), &time_end, NULL);
		*((unsigned long long*)exec_time) = (unsigned long long) (time_end - time_start);
	}
}

int main()
{
	int err = 0;
	unsigned long long exec_time = 0;

	// Initialize device for Telajax
	device_t device = telajax_device_init(0, NULL, &err);
	assert(!err);

	// Build kernel on device
	kernel_t helloworld_kernel = telajax_kernel_build(
		kernel_code,
		" -std=c99 ",       // cflags
		"",                 // lflags
		"hello_world",          // kernel_ocl_name
		kernel_ocl_wrapper,
		&device, &err);
	assert(!err);

	// Enqueue kernel
	telajax_kernel_enqueue(&helloworld_kernel, &device);
	telajax_kernel_set_callback(pfn_event_notify, (void*)&exec_time, &helloworld_kernel);

	// Wait for kernel termination (and callback is executed in backgroud)
	telajax_device_waitall(&device);

	printf("Exec_time = %llu ns\n", exec_time);

	// release kernel
	telajax_kernel_release(&helloworld_kernel);

	// Finalize Telajax
	telajax_device_finalize(&device);

	return 0;
}
