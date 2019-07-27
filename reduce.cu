__global__ void reduce_kernel(float* d_out, const float *d_in){

	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	// load shared memory from global memory
	sdata[tid] = d_in[myId];
	__syncthreads();

	// Do reduction in shared memory
	for(unsigned int s = blockDim.x/2; s>0; s>>=1){
		if(tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	// only thread 0 writes result for this block back to global mem.
	if (tid == 0){
		d_out[blockIdx.x] = sdata[0];
	}
}


void reduce(float *d_out, float *d_intermediate, float *d_in, int size){
	// assumes that size is not greater than maxThreadsPerBlock^2
	// and that size is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks =size/maxThreadsPerBlock;

	reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
}