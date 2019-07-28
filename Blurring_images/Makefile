NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_INCLUDEPATH=/usr/local/cuda-10.0/include
NVCC_OPTS=-O3 -arch=sm_30 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

operations: main.o ops.o Makefile
	$(NVCC) -o imageEditor main.o ops.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h image_utils.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

ops.o: ops.cu utils.h
	nvcc -c ops.cu $(NVCC_OPTS)

clean:
	rm -f *.o *.png hw