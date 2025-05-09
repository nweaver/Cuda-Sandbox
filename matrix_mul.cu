#include "matrix_mul.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>

Matrix<float> cudamul(Matrix<float> &a, Matrix<float> &b)
{
    return Matrix<float>();
}

__global__ void transpose(float *out, float *in, size_t size, size_t pitch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t x = tid % pitch;
    size_t y = tid / pitch;
    if (x < size && y < size)
    {
        out[x * pitch + y] = in[x + pitch * y];
    }
}

void checkResult(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorName(error) << "\n";

        std::cout << "CUDA error: " << cudaGetErrorString(error) << "\n";
    }
}

float *cudaTransposeInternal(Matrix<float> &bin, size_t &pitch) {
    float *csource;
    float *cdest;
    auto result = cudaMallocPitch(&csource,
                                  &pitch, bin._size * sizeof(float), bin._size);
    checkResult(result);

    result = cudaMemcpy2D(csource, pitch, (void *)bin._data, bin._size * sizeof(float),
                          bin._size * sizeof(float), bin._size, cudaMemcpyHostToDevice);
    checkResult(result);

    assert(result == cudaSuccess);

    result = cudaMallocPitch(&cdest,
                             &pitch, bin._size * sizeof(float), bin._size);
    checkResult(result);

    assert(result == cudaSuccess);

    int block_size = 256;
    int grid_size = ((bin._size * bin._size + block_size) / block_size);

    transpose<<<grid_size, block_size>>>(cdest, csource, bin._size, pitch / sizeof(float));
    cudaFree(csource);
    assert(result == cudaSuccess);
    return cdest;
}


Matrix<float> cudaTranspose(Matrix<float> &bin)
{
    size_t pitch;
    float *cdest = cudaTransposeInternal(bin, pitch);

    Matrix<float> d;
    d._data = (float *)malloc(sizeof(float) * bin._size * bin._size);
    d._size = bin._size;
    auto result = cudaMemcpy2D(d._data, bin._size * sizeof(float), cdest, pitch, bin._size * sizeof(float), bin._size,
                          cudaMemcpyDeviceToHost);
    checkResult(result);

    assert(result == cudaSuccess);

    result = cudaFree(cdest);
    checkResult(result);

    assert(result == cudaSuccess);

    return d;
}