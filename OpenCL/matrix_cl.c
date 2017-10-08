#include "matrix_cl.h"


static timestamp_t get_timestamp ()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void printArray(double arr[], int size){
    int i;
    printf("index:\t");
    for(i = 0; i < size; i++){
        printf("%d ", i);
    }
    printf("\n");
    printf("arr:\t");
    for(i = 0; i < size; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}


/****************************************************
 
 ***************************************************/
void matrix_mult_sq(int size, cl_double *matrix1_in,
               cl_double *matrix2_in, cl_double *matrix_out){
    
  int rows, cols;
  int j;
    
    // For cycle added to implement a sequential, matrix by matrix, multiplication that repeats the code given for the vector by matrix multipication for every row of the first operand's matrix
    for(rows=0; rows<size; rows++){
        for(cols=0; cols<size; cols++){
            matrix_out[rows*size + cols] = 0.0;
            for(j=0; j<size; j++)
                matrix_out[ rows*size + cols] += matrix1_in[ rows*size + j] * matrix2_in[j*size + cols];
        }
    }
}

/*****************************************************

 ****************************************************/
void matrix_vector_gen(cl_int size, cl_double *matrix1, cl_double *matrix2){
  int i;
    
  // Changed in order to instead generate two matrixes
  for(i=0; i<size*size; i++)
      matrix1[i] = ((double)rand())/65535.0;
  for(i=0; i<size*size; i++)
      matrix2[i] = ((double)rand())/5307.0;
}


/*****************************************************

 ****************************************************/
int main(int argc, char *argv[]){
    if(argc < 3){
        printf("Usage: %s (matrix/vector_size) (local group size)\n", argv[0]);
        return 0;
    }

    cl_int size = atoi(argv[1]);
    cl_int localSize = atoi(argv[2]);

    if((size == 0) || (localSize == 0)){
        printf("incorrect arguments, make sure both arguments are integers greater than zero\n");
        exit(-1);
    }else if((size % localSize) != 0){
        printf("size should be a multiple of localSize\n");
        exit(-1);
    }

    // Data structures allocated to store the operand and resulting matrixes of size "size*size"
    cl_double *matrix1 = (double *)malloc(sizeof(cl_double)*size*size);
    cl_double *matrix2 = (double *)malloc(sizeof(cl_double)*size*size);
    cl_double *result_sq = (double *)malloc(sizeof(cl_double)*size*size);
    cl_double *result_pl = (double *)malloc(sizeof(cl_double)*size*size);
    matrix_vector_gen(size, matrix1, matrix2);

    double time_sq;
    
    // Variables used to individually calculate the inititalization, copy and compilation times (that form the overhead) and the kernel runtime
    double time_opencl, time_opencl_init, time_opencl_comp, time_opencl_cpy;
    double time1, time2, time3, time4;

    cl_event mulDone;

    time_sq = omp_get_wtime();
    matrix_mult_sq(size, matrix1, matrix2, result_sq);
    time_sq = omp_get_wtime() - time_sq;

    cl_int status;
    time_opencl = omp_get_wtime();


    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    
    // Use clGetPlatformIDs() to retrieve the number of 
    // platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
 
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    if(status != CL_SUCCESS){
        printf("error in step 1\n");
        exit(-1);
    }

    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //-----------------------------------------------------
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    // Use clGetDeviceIDs() to retrieve the number of 
    // devices present
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_GPU,
        0, 
        NULL, 
        &numDevices);
    // Allocate enough space for each device
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    // Fill in devices with clGetDeviceIDs()
    status |= clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_GPU,
        numDevices, 
        devices, 
        NULL);
    // select the device which will be used
    int device_id = 0;
    
    if((status != CL_SUCCESS) || (device_id >= numDevices)){
        printf("error in step 2\n");
        exit(-1);
    }

    //-----------------------------------------------------
    // STEP 3: Create a context
    //-----------------------------------------------------
    cl_context context = NULL;

    // Create a context using clCreateContext() and 
    // associate it with the devices
    context = clCreateContext(
        NULL, 
        1, 
        &devices[device_id], 
        NULL, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 3\n");
        exit(-1);
    }

    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //-----------------------------------------------------
    cl_command_queue cmdQueue;

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute 
    // on
    cmdQueue = clCreateCommandQueue(
        context, 
        devices[device_id], 
        0, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 4\n");
        exit(-1);
    }

    
    //-----------------------------------------------------
    // STEP 5: Create device buffers, images and copy data to buffers
    //-----------------------------------------------------
    cl_mem bufferMatrixIn1, bufferMatrixIn2, bufferMatrixOut;

    
    bufferMatrixIn1 = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        size*size*sizeof(cl_double),
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 5, creating buffer for bufferA\n");
        exit(-1);
    }

    bufferMatrixIn2 = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        size*size*sizeof(cl_double), 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 5, creating buffer for bufferB\n");
        exit(-1);
    }

    bufferMatrixOut = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,                         
        size*size*sizeof(cl_double), 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 5, creating buffer for bufferC\n");
        exit(-1);
    }

    
    // Timer to calculate the initialization time
    time_opencl_init = omp_get_wtime();
    
    
    status = clEnqueueWriteBuffer ( 
        cmdQueue,
        bufferMatrixIn1,
        CL_FALSE,
        0,
        size*size*sizeof(cl_double),
        matrix1,
        0,
        NULL,
        NULL);

    status |= clEnqueueWriteBuffer ( 
        cmdQueue,
        bufferMatrixIn2,
        CL_FALSE,
        0,
        size*size*sizeof(cl_double),
        matrix2,
        0,
        NULL,
        NULL);

    if(status != CL_SUCCESS){
        printf("error in step 5, writing data\n");
        exit(-1);
    }
    
    char *mulFileName, *mulBuffer;
    mulFileName = "vectorMatrixMul.cl";
    FILE *mulFile;
    mulFile = fopen(mulFileName, "r");
    if(mulFile == NULL){
        printf("cannot open .cl file\n");
        printf("current path: %s\n", mulFileName);
        exit(-1);
    }
    fseek(mulFile, 0, SEEK_END);
    size_t mulSize = ftell(mulFile);
    rewind(mulFile);

    // read kernel source into buffer
    mulBuffer = (char*) malloc(mulSize + 1);
    mulBuffer[mulSize] = '\0';
    fread(mulBuffer, sizeof(char), mulSize, mulFile);
    fclose(mulFile);
    
    // Timer to calculate the copytime
    time_opencl_cpy = omp_get_wtime();

    //-----------------------------------------------------
    // STEP 6: Create and compile the program
    //----------------------------------------------------- 
    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**) &mulBuffer,                                 
        &mulSize, 
        &status);
    free(mulBuffer);


    // Build (compile) the program for the devices with
    // clBuildProgram()
    const char options[] = "-cl-std=CL1.2";
    status |= clBuildProgram(
        program, 
        1, 
        &devices[device_id], 
        options, 
        NULL, 
        NULL);

    if(status != CL_SUCCESS){
        printf("error in step 6\n");
        exit(-1);
    }



    //-----------------------------------------------------
    // STEP 7: Create the kernel
    //----------------------------------------------------- 
    cl_kernel mulKernel = NULL;

    // Use clCreateKernel() to create a kernel from the 
    mulKernel = clCreateKernel(program, "mul_kernel", &status);
    if(status != CL_SUCCESS){
        printf("error in step 7\n");
        exit(-1);
    }

    //-----------------------------------------------------
    // STEP 8: Set the kernel arguments
    //----------------------------------------------------- 
    // Associate the input and output buffers with the 
    // kernel 
    // using clSetKernelArg()
    status  = clSetKernelArg(
        mulKernel, 
        0, 
        sizeof(cl_mem), 
        &bufferMatrixIn1);
    status |= clSetKernelArg(
        mulKernel, 
        1, 
        sizeof(cl_mem), 
        &bufferMatrixIn2);
    status |= clSetKernelArg(
        mulKernel, 
        2, 
        sizeof(cl_mem), 
        &bufferMatrixOut);
    status |= clSetKernelArg(
        mulKernel, 
        3, 
        sizeof(cl_int), 
        &size);


    if(status != CL_SUCCESS){
        printf("error in step 8\n");
        exit(-1);
    }

    //-----------------------------------------------------
    // STEP 9: Configure the work-item structure
    //----------------------------------------------------- 
    // Define an index space (global work size) of work 
    // items for 
    // execution. A workgroup size (local work size) is not 
    // required, 
    // but can be used.

    
    size_t globalWorkSize[1];
    globalWorkSize[0] = size*size;

    size_t localWorkSize[1];
    localWorkSize[0] = localSize;
    

    // Timer to calculate the computation time
    time_opencl_comp = omp_get_wtime();
    
    status |= clEnqueueNDRangeKernel(
        cmdQueue, 
        mulKernel, 
        1, 
        NULL, 
        globalWorkSize, 
        localWorkSize, 
        0, 
        NULL, 
        &mulDone);

    if(status != CL_SUCCESS){
         clWaitForEvents (1,&mulDone);

        printf("error in clEnqueueNDRangeKernel\n");
        exit(-1);
    }
    
    clEnqueueReadBuffer(
        cmdQueue, 
        bufferMatrixOut,
        CL_TRUE, 
        0, 
        size*size*sizeof(cl_double),
        result_pl, 
        1, 
        &mulDone, 
        NULL);


    if(status != CL_SUCCESS){
    printf("error in reading data\n");
    exit(-1);
    }

    // Computation time
    time1 = omp_get_wtime() - time_opencl_comp;
    // Inicialization time
    time2 = time_opencl_init - time_opencl;
    // Copytime
    time3 = time_opencl_cpy - time_opencl_init;
    // Compilation time
    time4 = time_opencl_comp - time_opencl_cpy;

    
    printf("SEQUENTIAL EXECUTION: %f (sec)\n", time_sq);
    printf("PARALLEL EXECUTION WITH A LOCAL WORK GROUP SIZE OF %d: %f (sec)\nSplit between OVERHEAD %f (sec) and KERNEL RUNTIME %f (sec).\nOverhead is composed of Inicialization time: %f (sec), Copy time: %f (sec) and Compilation time: %f (sec)\n ", localSize, time1+time2+time3+time4, time2+time3+time4 ,time1, time2, time3, time4 );

    //check
    int i;
    for(i=0; i<size; i++){
        if((int) result_sq[i] != (int) result_pl[i]){
            printf("wrong at position %d\n", i);
            return 0;
        }
    }
     
    
    //-----------------------------------------------------
    // STEP 10: Release OpenCL resources
    //----------------------------------------------------- 
    
    // Free OpenCL resources
    clReleaseKernel(mulKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferMatrixIn1);
    clReleaseMemObject(bufferMatrixIn2);
    clReleaseMemObject(bufferMatrixOut);
    clReleaseContext(context);

    //Free up memory and close files
    free(matrix1);
    free(matrix2);
    free(result_sq);
    free(result_pl);

    return EXIT_SUCCESS;
}

