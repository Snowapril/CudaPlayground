#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <Common/helper_cuda.h>


//! Pull out matrix and shared memory tile size
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;

//! CUDA Kernel for matrix multiplication
//! __global__ means this is called from the CPU and runs on the GPU
__global__ void matrixMul(const int* __restrict a, const int* __restrict b, int* __restrict c)
{
	//! Compute each thread's global row and column index
	int tx = threadIdx.x, ty = threadIdx.y;

	int col = blockDim.x * blockIdx.x + tx;
	int row = blockDim.y * blockIdx.y + ty;

	//! Statically allocated shared memory
	__shared__ int s_a[SHMEM_SIZE];
	__shared__ int s_b[SHMEM_SIZE];

	//! Accumulate in temporary variable
	int temp = 0;

	//! Sweep tile across matrix
	for (int i = 0; i < N / blockDim.x; ++i)
	{
		//! Load in elements for this tile
		s_a[ty * blockDim.x + tx] = a[row * N + i * blockDim.x + tx];
		s_b[ty * blockDim.x + tx] = b[i * blockDim.x * N + ty * N + col];

		//! Wait for both tiles to be loaded in before doing computation
		__syncthreads();

		//! Do matrix multiplication on the small matrix
		for (int j = 0; j < blockDim.x; ++j)
		{
			temp += s_a[ty * blockDim.x + j] * s_b[tx * blockDim.x + j];
		}

		//! Wait for all threads to finish using current tiles before loading in new ones
		__syncthreads();
	}
	c[row * N + col] = temp;
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

void transpose(const std::vector<int>& a, std::vector<int>& a_transposed, int N)
{
	assert(a_transposed.size() == N * N && "a_transposed must be resized "
		"as N^2 before passing this function");

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			a_transposed[j * N + i] = a[i * N + j];
		}
	}
}

int main()
{
	//! matrix size of 2^20 elements
	constexpr size_t bytes = sizeof(int) * N * N;

	//! vectors for holding the host-side data
	std::vector<int> a, b, c, b_transposed;
	a.reserve(N * N);
	b.reserve(N * N);
	c.reserve(N * N);

	//! Initialize random numbers in each array
	for (size_t i = 0; i < N * N; ++i)
	{
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	//! Transpose the B matrix before launching kernel
	b_transposed.resize(N * N);
	transpose(b, b_transposed, N);

	//! Allocate memory on the device
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	//! Copy data from the host to the device(CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b_transposed.data(), bytes, cudaMemcpyHostToDevice);

	//! Threads per CTA
	int THREADS = 32;
	//! Blocks per grid dimension (assume THREADS divides N evenly)
	int BLOCKS = N / THREADS;

	//! Use dim3 structs for block and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	//! Launch Kernel
	matrixMul << <blocks, threads >> > (d_a, d_b, d_c);
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