NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_INCLUDEPATH=/usr/local/cuda-10.0/include
NVCC_OPTS=-O3 -arch=sm_30 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

student: main.o ops.o image_utils.o loadSaveImage.o Makefile
	$(NVCC) -o imageEditor main.o ops.o image_utils.o loadSaveImage.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h 
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

image_utils.o: image_utils.cu loadSaveImage.h utils.h
	$(NVCC) -c image_utils.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

ops.o: ops.cu utils.h
	$(NVCC) -c ops.cu $(NVCC_OPTS)

clean:
	rm -f *.o hw
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
