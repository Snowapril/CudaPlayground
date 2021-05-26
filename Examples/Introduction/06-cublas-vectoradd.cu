#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <Common/helper_cuda.h>

void verify_result(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c, const float scale)
{
	for (size_t i = 0; i < a.size(); ++i)
		assert(c[i] == a[i] * scale + b[i]);
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
	c.resize(N);

	//! Initialize random numbers in each array
	for (size_t i = 0; i < N; ++i)
	{
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	//! Allocate memory on the device
	float* d_a, * d_b;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	//! Create and initialize a new context
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	//! Copy the vectors over to the device
	cublasSetVector(n, sizeof(float), a.data(), 1, d_a, 1);
	cublasSetVector(n, sizeof(float), b.data(), 1, d_b, 1);

	//! Launch simple saxpy kernel (single precision a * x + y)
	const float scale = 1.0f;
	cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

	//! Copy the result vector back out
	cublasGetVector(n, sizeof(float), d_b, 1, c.data(), 1);

	//! Check result for errors
	verify_result(a, b, c, scale);

	//! Clean up the created handle
	cublasDestroy(handle);

	//! Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);

	std::cout << "COMPLETED SUCCESSFULLY" << std::endl;
	return 0;
}