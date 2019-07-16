#include <iostream>

using namespace std; 

__global__ void cube(float * d_out, float * d_in){
	int idx = threadIdx.x;
	float val = d_in[idx];
	d_out[idx] = val*val*val;
}

int main(){
	const int ARRAY_SIZE = 96;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];

	// Fill the input array
	for(int i = 0; i < ARRAY_SIZE; ++i){
		h_in[i] = float(i);
	}

	// Declare GPU memory pointers 
	float * d_in;
	float * d_out;

	// Allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// Transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Launch kernel
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// Transfer results to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// Print results
	for(int i = 0; i < ARRAY_SIZE; i++){
		cout << h_out[i] << endl;
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}