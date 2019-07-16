/** 
  Color to Greyscale Conversion

  A common way to represent color images is known as RGBA - the color
  is specified by how much Red, Grean and Blue is in it.
  The 'A' stands for Alpha and is used for transparency, it will be
  ignored in this homework.

  Each channel Red, Blue, Green and Alpha is represented by one byte.
  Since we are using one byte for each color there are 256 different
  possible values for each color.  This means we use 4 bytes per pixel.

  Greyscale images are represented by a single intensity value per pixel
  which is one byte in size.

  To convert an image from color to grayscale one simple method is to
  set the intensity to the average of the RGB channels.  But we will
  use a more sophisticated method that takes into account how the eye 
  perceives color and weights the channels unequally.

  The eye responds most strongly to green followed by red and then blue.
  The NTSC (National Television System Committee) recommends the following
  formula for color to greyscale conversion:

  I = .299f * R + .587f * G + .114f * B
**/


#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
   for(int i = 0; i < numRows; ++i){
    for (int j = 0; j < numCols; ++j){
      uchar4 rgba = rgbaImage[i*numCols+j];
      float intensity = (0.299f*rgba.x) + (.587f*rgba.y) + (.114f*rgba.z);
      greyImage[i*numCols+j] = intensity;
    }
  }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{

  const int threads = 32;
  const dim3 blockSize(threads, threads, 1);
  const dim3 gridSize(ceil((float)numCols/threads), ceil((float)numRows/threads), 1);

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

}
