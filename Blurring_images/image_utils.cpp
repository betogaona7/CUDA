#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

cv::Mat imageInput;
cv::Mat imageOutput;

uchar4 *d_inputImage__;
uchar4 *d_outputImage__;

float *h_filter__;


size_t numRows(){
	return imageInput.rows;
}

size_t numCols(){
	return imageInput.cols;
}

void preProcess(uchar4 **h_inputImage, uchar4 **h_outputImage, 
			    uchar4 **d_inputImage, uchar4 **d_outputImage,
			    unsigned char **d_redBlurred,
			    unsigned char **d_greenBlurred,
			    unsigned char **d_blueBlurred,
			    float **h_filter, int *filterWidth,
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
	cv::cvtColor(image, imageInput, CV_BGR2RGBA);

	// Allocate memory for the output 
	imageOutput.create(image.rows, image.cols, CV_8UC4);

	*h_inputImage  = (uchar4 *)imageInput.ptr<unsigned char>(0);
	*h_outputImage = (uchar4 *)imageOutput.ptr<unsigned char>(0);
	const size_t numPixels = numRows() * numCols();

	// Allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_inputImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_outputImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(*d_outputImage, 0, sizeof(uchar4) * numPixels)); // Make sure no memory is left laying around.

	// Copy input array to the GPU 
	checkCudaErrors(cudaMemcpy(*d_inputImage, *h_inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImage__ = *d_inputImage;
	d_outputImage__ = *d_outputImage;

	// Create filter 
	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;
	*filterWidth = blurKernelWidth;

	// Fill the filter that we will convolve with
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;
	float filterSum = 0.f; // For normalization

	for(int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r){
		for(int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c){
			float filterValue = expf(-(float)(c*c + r*r) / (2.f * blurKernelSigma * blurKernelSigma));
			(*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for(int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r){
		for(int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c){
			(*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
		}
	}

	//blurred
  	checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string &output_file, uchar4 *data_ptr){
	// Create output image
	cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);
	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
	// Save it
	cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanUp(){
	cudaFree(d_inputImage__);
	cudaFree(d_outputImage__);
	delete[] h_filter__;
}