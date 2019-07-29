#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <algorithm>

// Function definitions 
void preProcess(float **d_luminance, unsigned int **d_cdf,
                size_t *numRows, size_t *numCols, unsigned int *numBins,
                const std::string &filename);

void postProcess(const std::string &output_file, size_t numRows, size_t numCols,
                 float min_logLum, float max_logLum);

void cleanupGlobalMemory(void);

void histogram_and_prefixsum(const float* const d_luminance, unsigned int* const d_cdf,
	                        float &min_logLum, float &max_logLum,
	                        const size_t numRows, const size_t numCols, const size_t numBins);

int main(int argc, char *argv[]){

	float *d_luminance;
	unsigned int *d_cdf;
	size_t numRows, numCols;
	unsigned int numBins;

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

  preProcess(&d_luminance, &d_cdf, &numRows, &numCols, &numBins, in_file);

  GpuTimer timer;
  float min_logLum, max_logLum;
  min_logLum = 0.f;
  max_logLum = 1.f;

  timer.Start();

  histogram_and_prefixsum(d_luminance, d_cdf, min_logLum, max_logLum, numRows, numCols, numBins);

  timer.Stop();
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

  int err = printf("Finished in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  float *h_luminance = (float *) malloc(sizeof(float)*numRows*numCols);
  unsigned int *h_cdf = (unsigned int *) malloc(sizeof(unsigned int)*numBins);

  checkCudaErrors(cudaMemcpy(h_luminance, d_luminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost));

  postProcess(out_file, numRows, numCols, min_logLum, max_logLum);

    cleanupGlobalMemory();

  return 0;
}