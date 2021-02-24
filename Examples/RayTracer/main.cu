#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		//! Make sure we call CUDA device reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

int main(int argc, char* argv[])
{
	int nx = 512, ny = 512;
	int numPixels = nx * ny;
	size_t fb_size = 3 * numPixels * sizeof(float);

	//! Allocates the frame buffer
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));



	return 0;
}