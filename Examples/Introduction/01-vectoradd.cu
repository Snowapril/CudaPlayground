#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

//! CUDA Kernel for vector addition
//! __global__ means this is called from the CPU and runs on the GPU
__global__ void vectorAdd(const int* __restrict a, const int* __restrict b, int* __restrict c, int N)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) c[tid] = a[tid] + b[tid];
}

void verify_result(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c)
{
	for (size_t i = 0; i < a.size(); ++i)
		assert(c[i] == a[i] + b[i]);
}

int main()
{
	//! array size of 2^16(65536 elements)
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	//! vectors for holding the host-side data
	std::vector<int> a, b, c;
	a.reserve(N);
	b.reserve(N);
	c.reserve(N);

	//! Initialize random numbers in each array
	for (size_t i = 0; i < N; ++i)
	{
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	//! Allocate memory on the device
	int* d_a, * d_b, * d_c;
	cudaMalloc(d_a, bytes);
	cudaMalloc(d_b, bytes);
	cudaMalloc(d_c, bytes);

	//! Copy data from the host to the device(CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	//! Threads per CTA(cooperative thread array) (1024)
	int NUM_THREADS = 1 << 10;
	//! CTAs per Grid
	//! We need to launch at LEAST as many threads as we have elements'
	//! This equation pads and extra CTA to the grid if N cannot evenly be divided
	//! by NUM_THREADS(e.g. N=1025, NUM_THREADS = 1024)
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	//! Launch the kernel on the GPU
	//! Kernel calls are asynchronous (the CPU program continues execution after
	//! call, but no necessarily before the kernel finishes)
	vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c);

	//! Copy sum vector from device to host
	//! cudaMemcpy is a synchronous operation, and waits for the prior kernel
	//! launch to complete(both go to the default stream in this case).
	//! Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	//! barrier
	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	//! Check result for errors
	verify_result(a, b, c);

	//! Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY" << std::endl;
	return 0;
}