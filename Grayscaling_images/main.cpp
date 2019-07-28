#include <iostream>
#include <stdio.h>

#include "image_utils.cpp"
#include "timer.h"
#include "utils.h"

// Function definitions 
void convert_to_grayscale(const uchar4 *const h_rgbaImage,
						  const uchar4 *const d_rgbaImage,
						  unsigned char *const d_greyImage,
						  size_t numRows, size_t numCols);

int main(int argc, char *argv[]){

	uchar4 *h_rgbaImage, *d_rgbaImage; 
	unsigned char *h_greyImage, *d_greyImage;

	std::string in_file;
	std::string out_file;

	switch(argc){
		case 2:
			in_file = std::string(argv[1]);
			out_file = "output_image.png";
			break;
		case 3:
			in_file = std::string(argv[1]);
			out_file = std::string(argv[2]);
			break;
		default:
			std::cerr << "Usage: ./imageEditor command input_file [output_filename]" << std::endl;
			return 1;
	}

	// Load image
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, in_file);

	GpuTimer timer; 
	timer.Start();

	// Calling my kernel
	convert_to_grayscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());

	timer.Stop();
	cudaDeviceSynchronize();

	int err = printf("Finished in %f msec.\n", timer.Elapsed());
	if(err < 0){
		std::cerr << "Couldn't print timing information." << std:: endl;
	}

	// Transfer results from GPU to CPU.
	size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	// Save image
	postProcess(out_file, h_greyImage);

	// Clean
	cleanup();

	return 0;
}