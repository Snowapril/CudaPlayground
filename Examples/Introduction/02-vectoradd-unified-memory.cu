#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <Common/helper_cuda.h>

//! CUDA Kernel for vector addition
//! __global__ means this is called from the CPU and runs on the GPU
//! No change when using CUDA unified memory
__global__ void vectorAdd(const int* __restrict a, const int* __restrict b, int* __restrict c, int N)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) c[tid] = a[tid] + b[tid];
}

void verify_result(int *a, int *b, int *c, int N)
{
	for (size_t i = 0; i < N; ++i)
		assert(c[i] == a[i] + b[i]);
}

int main()
{
	//! array size of 2^16(65536 elements)
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	//! Declare unified memory pointers
	int *a, *b, *c;
	//! Allocation memory for these pointers
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	//! Get the device ID for prefecthing calls
	int id = cudaGetDevice(&id);

	//! Set some hints about the data and do some prefetching
	cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMemPrefetchAsync(c, bytes, id);

	//! Initialize random numbers in each array
	for (size_t i = 0; i < N; ++i)
	{
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	//! Set some hints about the data and do some prefecthing
	cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
	cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);

	//! Threads per CTA (1024 threads per CTA)
	int BLOCK_SIZE = 1 << 10;
	//! CTAS per grid
	int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	//! Launch the kernel on the GPU
	//! Kernel calls are asynchronous (the CPU program continues execution after
	//! call, but no necessarily before the kernel finishes)
	vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

	//! Wait for all previous operations before using values.
	//! We need this because we don't get the implicit synchronization of 
	//! cudaMemcpy like in the original example
	cudaDeviceSynchronize();

	//! Prefetch to the host CPU
	cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
	cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	//! Check result for errors
	verify_result(a, b, c, N);

	//! Free memory on device
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	std::cout << "COMPLETED SUCCESSFULLY" << std::endl;
	return 0;
}