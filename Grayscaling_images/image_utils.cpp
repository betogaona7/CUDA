#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;


size_t numRows(){
	return imageRGBA.rows;
}

size_t numCols(){
	return imageRGBA.cols;
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage, 
			    uchar4 **d_rgbaImage, unsigned char **d_greyImage,
			    const std::string &filename){

	// Make sure the context initialized okay.
	checkCudaErrors(cudaFree(0));

	// Read input image 
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()){
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	// Change color from BGR to RGBA
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	// Allocate memory for the output 
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();

	// Allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greyImage, 0, sizeof(unsigned char) * numPixels)); // Make sure no memory is left laying around.

	// Copy input array to the GPU 
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string &output_file, unsigned char *data_ptr){
	// Create output image
	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
	// Save it
	cv::imwrite(output_file.c_str(), output);
}

void cleanup(){
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}