#include "telajax.h"

/*

OpenCL kernel wrapper :

void _vec_add(int n, __global float* x, __global float* y);

__kernel vec_add(int n, __global float* x, __global float* y)
{
	_vec_add(n, x, y);
}


GCC kernel code :

#include <stdio.h>

void _vec_add(int n, float* x, float* y)
{
	for(int i = 0; i < n; i++){
		y[i] += x[i];
	}
}

*/

#define RANDOM_STRING_LENGTH 8
#define FILE_PATH_LENGTH     128
#define COMMAND_LENGTH       1024

kernel_t
telajax_kernel_build(
	const char* kernel_code,
	const char* cflags, const char* lflags,
	const char* kernel_ocl_name,
	const char* kernel_ocl_wrapper,
	device_t* device, int* error)
{
	int err;
	char cmd[COMMAND_LENGTH];
	char rand_string[RANDOM_STRING_LENGTH+1];
	char rand_file_path_src[FILE_PATH_LENGTH];
	char rand_file_path_obj[FILE_PATH_LENGTH];
	kernel_t kernel_res;

	if(kernel_code == NULL || kernel_ocl_name == NULL || kernel_ocl_wrapper == NULL){
		err = -1; goto ERROR;
	}

	// get a random string
	snprintf(cmd, COMMAND_LENGTH,
		"cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w %d | head -n 1",
		RANDOM_STRING_LENGTH);
	FILE* fpopen = popen(cmd, "r");
	if (fpopen == NULL) {
		printf("Failed to generate a random string\n");
		err = -1; goto ERROR;
	}
	fgets(rand_string, RANDOM_STRING_LENGTH+1, fpopen);
	rand_string[RANDOM_STRING_LENGTH] = '\0';
	pclose(fpopen);

	// dump kernel_code to a .c file on disk
	char tmpdir[FILE_PATH_LENGTH];
	snprintf(tmpdir, FILE_PATH_LENGTH, "%s", getenv("HOME"));
	snprintf(tmpdir+strlen(tmpdir), FILE_PATH_LENGTH-strlen(tmpdir), "/.cache/telajax");
	mkdir(tmpdir, S_IRWXU);

	snprintf(rand_file_path_src, FILE_PATH_LENGTH, "%s/%s.c", tmpdir, rand_string);
	snprintf(rand_file_path_obj, FILE_PATH_LENGTH, "%s/%s.o", tmpdir, rand_string);

	FILE* fp;
	fp = fopen(rand_file_path_src, "w");
	if(fp == NULL){
		printf("Failed to create %s on disk - %s\n", rand_file_path_src, strerror(errno));
		err = -1; goto ERROR;
	}
	fwrite(kernel_code, 1, strlen(kernel_code), fp);
	fclose(fp);

	// Compile the C kernel_code to a .o
	if(getenv("K1_TOOLCHAIN_DIR") == NULL){
		printf("K1_TOOLCHAIN_DIR not set, you do not forget something ?\n");
		err = -1; goto ERROR;
	}
	snprintf(cmd, COMMAND_LENGTH,
		"%s"                 // K1_TOOLCHAIN_DIR
		"/bin/k1-elf-gcc "
		" %s "               // cflags
		" -c %s "            // rand_file_path_src
		" -o %s "            // rand_file_path_obj
		" %s "               // lflags
		, getenv("K1_TOOLCHAIN_DIR")
		, cflags
		, rand_file_path_src
		, rand_file_path_obj
		, lflags
		);
	if(system(cmd)){
		printf("%s\n", cmd);
		printf("Failed to compile kernel_code\n");
		err = -1; goto ERROR;
	}

	//  declare two cl_program
	cl_program input_programs[2];

	// build wrapper program
	input_programs[0] = clCreateProgramWithSource(device->_context, 1,
					(const char **) &kernel_ocl_wrapper, NULL, &err);
	assert(input_programs[0]);

	err = clBuildProgram(input_programs[0], 0, NULL, NULL, NULL, NULL);
	assert(!err);

	// create elf program and link
	// read the .o
	fp = fopen(rand_file_path_obj, "r");
	fseek(fp, 0, SEEK_END);
	size_t size_ftell = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *kernel_elf_source = (char*)malloc(size_ftell);
	fread(kernel_elf_source, 1, size_ftell, fp);
	fclose(fp);

	input_programs[1] = clCreateProgramWithBinary(device->_context, 1,
		&(device->_device_id), &size_ftell, (const unsigned char **) &kernel_elf_source,
		NULL, &err);
	assert(input_programs[1]);
	free(kernel_elf_source);

	cl_program program_final = clLinkProgram(device->_context, 1, &(device->_device_id),
		NULL, 2, input_programs, NULL, NULL, &err);
	assert(!err);
	assert(program_final);

	// release input_programs
	clReleaseProgram(input_programs[0]);
	clReleaseProgram(input_programs[1]);

	// build the OpenCL kernel associated with program_final
	cl_kernel kernel_final = clCreateKernel(program_final, kernel_ocl_name, &err);
	assert(kernel_final);
	assert(!err);

	// delete rand_file_path_src and rand_file_path_obj
	remove(rand_file_path_src);
	remove(rand_file_path_obj);

	// setup kernel_res and return
	kernel_res._program       = program_final;
	kernel_res._kernel        = kernel_final;
	kernel_res._work_dim      = 3;
	kernel_res._globalSize[0] = 1;
	kernel_res._globalSize[1] = 1;
	kernel_res._globalSize[2] = 1;
	kernel_res._localSize[0]  = 1;
	kernel_res._localSize[1]  = 1;
	kernel_res._localSize[2]  = 1;

ERROR:
	if(error) *error = err;
	return kernel_res;
}

