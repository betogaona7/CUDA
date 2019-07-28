#include "utils.h"

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
								  unsigned char *const greyImage,
								  int numRows, int numCols){
	/*
	Converts an image from color to grayscale taking into account how the eye perceives 
	color. The National Television System Commite (NTSC) defined that eye responds most 
	strongly to green followed by red and then blue. So the formula is:

	I = .299f * R + .587f * G + .114f * B
	*/
	for(int i = 0; i < numRows; ++i){
		for(int j = 0; j < numCols; ++j){
			uchar4 rgba = rgbaImage[i*numCols+j];
			float intensity = (0.299f*rgba.x) + (.587f*rgba.y) + (.114f*rgba.z);
			greyImage[i*numCols+j] = intensity;
		}
	}
}


void convert_to_grayscale(const uchar4 *const h_rgbaImage,
						  const uchar4 *const d_rgbaImage,
						  unsigned char *const d_greyImage,
						  size_t numRows, size_t numCols){
	const int threads = 32;
	const dim3 blockSize(threads, threads, 1);
	const dim3 gridSize(ceil((float)numCols/threads), ceil((float)numRows/threads), 1);

	// launch kernel
	rgba_to_greyscale <<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

	cudaDeviceSynchronize();
}

// !sudo apt-get install nvidia-cuda-toolkit