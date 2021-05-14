#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <Common/helper_cuda.h>

//! CUDA Kernel for matrix multiplication
//! __global__ means this is called from the CPU and runs on the GPU
__global__ void matrixMul(const int* __restrict a, const int* __restrict b, int* __restrict c, int N)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if ((col < N) && (row < N))
	{
		int temp = 0;
		for (int k = 0; k < N; ++k)
			temp += a[row * N + k] * b[k * N + col];
		c[row * N + col] = temp;
	}
}

void verify_result(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c, int N)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			int temp = 0;
			for (int k = 0; k < N; ++k)
			{
				temp += a[i * N + k] * b[k * N + j];
			}

			assert(temp == c[i * N + j]);
		}
	}
}

int main()
{
	//! array size of 2^10(1024 elements)
	constexpr int N = 1 << 10;
	constexpr size_t bytes = sizeof(int) * N * N;

	//! vectors for holding the host-side data
	std::vector<int> a, b, c;
	a.reserve(N * N);
	b.reserve(N * N);
	c.reserve(N * N);

	//! Initialize random numbers in each array
	for (size_t i = 0; i < N * N; ++i)
	{
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	//! Allocate memory on the device
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	//! Copy data from the host to the device(CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	//! Threads per CTA
	int THREADS = 32;
	//! Blocks per grid dimension (assume THREADS divides N evenly)
	int BLOCKS = N / THREADS;
	
	//! Use dim3 structs for block and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	//! Launch Kernel
	matrixMul <<<blocks, threads>>> (d_a, d_b, d_c, N);
	//! Copy sum vector from device to host
	//! cudaMemcpy is a synchronous operation, and waits for the prior kernel
	//! launch to complete(both go to the default stream in this case).
	//! Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	//! barrier
	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	//! Check result for errors
	verify_result(a, b, c, N);

	//! Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY" << std::endl;
	return 0;
}