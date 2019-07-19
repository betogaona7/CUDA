/******* local memory  *******/
__global__ void use_local_gpu_memory(float in){
	float x; // Variable "x" and "in" are in local memory and private to each thread.
	x = in;
}

int main(){
	use_local_gpu_memory<<<128, 1>>>(2.0f);
}



/****** Global memory *******/
__global__ void use_global_gpu_memory(float *array){
	array[threadIdx.x] = 2.0f * (float)threadIdx.x; // "array" is a ponter into global memory on device.
}

int main(){
	float h_arra(128);
	float *d_arr;

	// allocate global memory on the device, place result in "d_arr"
	cudaMalloc((void **) &d_arr, sizeof(float)*128);
	// Copy data from host memory "h_arr" to device memory "d_arr"
	cudaMemcpy((void *)d_arr, (void *) h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);
	// Launch the kernel
	use_global_gpu_memory<<<128, 1>>>(d_arr); // modifes the contents of d_arr
	// copy the modified array back to the host, overriting contents of h_arr
	cudaMemcpy((void*)h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);
}


/******* Shared memory *******/
__global__ void use_shared_gpu_memory(float *array){
	float i, index = threadIdx.x;
	float average, sum = 0.0f;

	// shared variables are visible to all threads in the thread block
	// and have the same lifetime as the thread block. 
	__shared__ float sh_arr[128];

	// copy data from "array" in global memory to "sh_arr" in shared memory.
	sh_arr[index] = array[index];
	_syncthreads(); //enssure all the writes to shared memory have completed.

	// as an example, lets find the average of all previous elements.
	for(i = 0; i < index; i++){ sum += sh_arr[i]; }
	average = sum / (index+1.0f);

	// if array[index] is greater than the average of array[0..index-1] replace with average.
	// since array[] is in global memory, this change will be seen by the host (and potentually
	// other thread block, if any)
	if (array[index] > average){ array[index] = average;}
	_syncthreads();
}

int main(){
	// same as global
}