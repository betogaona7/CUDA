/** 
	How many different outputs can different run of this program produce?
	Answer: 16! 
**/

#include <iostream>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello(){
	std::cout << "Hello world! I'm a thread in block: " << blockIdx.x << std::endl;
}

int main(){
	// Launch kernel 
	hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

	// force the couts to flush
	cudaDeviceSynchronize();

	std::cout << "That's all!\n";
	return 0;
}
