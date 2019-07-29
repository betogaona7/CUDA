#include "utils.h"

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                              unsigned char* const outputChannel,
                              int numRows, int numCols,
                              const float* const filter, const int filterWidth){
  
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  float result = 0.0f;
  for(int r = 0; r < filterWidth; ++r){
    for (int c = 0; c < filterWidth; ++c){
      int col = thread_2D_pos.x + c - filterWidth/2;
      int row = thread_2D_pos.y + r - filterWidth/2;

      col = min(max(col, 0), numCols - 1);
      row = min(max(row, 0), numRows - 1);

      result += filter[r * filterWidth + c] * static_cast<float>(inputChannel[row * numCols + col]);
    }
  }
  outputChannel[thread_1D_pos] = result;
}

__global__ void separateChannels(const uchar4* const inputImageRGBA,
                                unsigned char* const redChannel,
                                unsigned char* const greenChannel,
                                unsigned char* const blueChannel,
                                int numRows, int numCols){
  /* 
    Separates the different color channels so that each color is stored contiguously 
    instead of being interleaved. From an AoS to a SoA. 
  */
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

__global__ void recombineChannels(const unsigned char* const redChannel,
                                  const unsigned char* const greenChannel,
                                  const unsigned char* const blueChannel,
                                  uchar4* const outputImageRGBA,
                                  int numRows, int numCols){
  /*
    Combines the different color channels. From a SoA to an AoS. 
  */
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  // alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  // allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  // allocate memory for the filter on the GPU
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

  // copy the filter on the host to the GPU
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}



void apply_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                         uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                         unsigned char *d_redBlurred, 
                         unsigned char *d_greenBlurred, 
                         unsigned char *d_blueBlurred,
                         const int filterWidth){

  const dim3 blockSize(16, 16);
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);

  // Launch a kernel for separating the RGBA images
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, d_red, d_green, d_blue, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Call convolution kernel three times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  

  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Recombine results
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
