#include <iostream>
#include <stdio.h>

#include "image_utils.cpp"
#include "timer.h"
#include "utils.h"

// Function definitions 
void apply_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
						 uchar4* const d_outputImageRGBA,
						 const size_t numRows, const size_t numCols,
						 unsigned char *d_redBlurred,
						 unsigned char *d_greenBlurred,
						 unsigned char *d_blueBlurred,
						 const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth);


int main(int argc, char *argv[]){

	uchar4 *h_inputImage, *d_inputImage; 
	uchar4 *h_outputImage, *d_output_Image;
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

	float *h_filter;
	int filterWidth;

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
	preProcess(&h_inputImage, &h_outputImage, &d_inputImage, &d_output_Image, 
			   &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
			   &h_filter, &filterWidth, in_file);

	allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

	GpuTimer timer; 
	timer.Start();

	// Calling my kernel
	apply_gaussian_blur(h_inputImage, d_inputImage, d_output_Image, numRows(), numCols(),
						d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);



	timer.Stop();
	cudaDeviceSynchronize();

	int err = printf("Finished in %f msec.\n", timer.Elapsed());
	if(err < 0){
		std::cerr << "Couldn't print timing information." << std:: endl;
	}

	// Transfer results from GPU to CPU.
	size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMemcpy(h_outputImage, d_output_Image, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	// Save image
	postProcess(out_file, h_outputImage);

	checkCudaErrors(cudaFree(d_redBlurred));
  	checkCudaErrors(cudaFree(d_greenBlurred));
  	checkCudaErrors(cudaFree(d_blueBlurred));
  	
	// Clean
	cleanup();

	return 0;
}