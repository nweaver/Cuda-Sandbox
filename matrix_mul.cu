#include "matrix_mul.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

// Code and concepts "borrowed/liberated" from https://github.com/siboehm/SGEMM_CUDA
// and modified just to be classic matrix multiplication...

long CEIL_DIV(long x, long y)
{
    if (x % y)
        return x / y + 1;
    return x / y;
}

float *cudaTransposeInternal(Matrix<float> &bin, size_t &pitch);

void checkResult(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorName(error) << "\n";

        std::cout << "CUDA error: " << cudaGetErrorString(error) << "\n";
    }
}

// Going to do this as square matrixes, and assume a natural power of 2 size.
__global__ void matmul(float *out, float *a, float *b, size_t size)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;
    for (int i = 0; i < size; ++i)
    {
        tmp += a[x * size + i] * b[i * size + y];
    }
    out[x * size + y] = tmp;
}

__global__ void cachematmul(float *out, float *a, float *b, size_t size)
{
  const int BLOCKSIZE = 32;

  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  a += cRow * BLOCKSIZE * size;                      // row=cRow, col=0
  b += cCol * BLOCKSIZE;                             // row=0, col=cCol
  out += cRow * BLOCKSIZE * size + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < size; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = a[threadRow * size + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = b[threadRow * size + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    a += BLOCKSIZE;
    b += BLOCKSIZE * size;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  out[threadRow * size + threadCol] = tmp;
}


// Going to do this as square matrixes, and assume a natural power of 2 size.
__global__ void coalesce_matmul(float *out, float *a, float *b, size_t size)
{
    const uint BLOCKSIZE = 32; // Number of threads in a warm
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    float tmp = 0.0;
    for (int i = 0; i < size; ++i)
    {
        tmp += a[x * size + i] * b[i * size + y];
    }
    out[x * size + y] = tmp;
}

void cudasetup(Matrix<float> &a, Matrix<float> &b, float *&ain, float *&bin, float *&dout)
{
    auto size = a._size;
    // Cheating, assuming size is 0 mod 64 for square matrixes
    assert(size % 64 == 0);
    assert(a._size == b._size);
    assert(size > 64);

    auto capacity = size * size * sizeof(float);
    auto result = cudaMalloc(&ain, capacity);
    checkResult(result);
    result = cudaMalloc(&bin, capacity);
    checkResult(result);
    result = cudaMalloc(&dout, capacity);
    checkResult(result);
    result = cudaMemcpy(ain, a._data, capacity, cudaMemcpyHostToDevice);
    checkResult(result);
    result = cudaMemcpy(bin, b._data, capacity, cudaMemcpyHostToDevice);
    checkResult(result);
}

void cudateardown(Matrix<float> &d, size_t size, float *&ain, float *&bin, float *&dout)
{

    d._data = (float *)malloc(size * sizeof(float));
    d._size = size;
    auto result = cudaMemcpy(d._data, dout, size * sizeof(float),
                             cudaMemcpyDeviceToHost);
    checkResult(result);

    assert(result == cudaSuccess);
    cudaFree(ain);
    cudaFree(bin);
    result = cudaFree(dout);
    checkResult(result);

    assert(result == cudaSuccess);
}

Matrix<float> cudamul(Matrix<float> &a, Matrix<float> &b)
{
    auto size = a._size;
    Matrix<float> d;
    float *bin;
    float *ain;
    float *dout;
    cudasetup(a, b, ain, bin, dout);
    dim3 gridDim(size / 32, size / 32, 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    matmul<<<gridDim, blockDim>>>(dout, ain, bin, a._size);
    cudateardown(d, size, ain, bin, dout);
    return d;
}

Matrix<float> cudamul_cache(Matrix<float> &a, Matrix<float> &b)
{
    auto size = a._size;
    Matrix<float> d;
    float *bin;
    float *ain;
    float *dout;
    cudasetup(a, b, ain, bin, dout);
    dim3 gridDim(size / 32, size / 32, 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    cachematmul<<<gridDim, blockDim>>>(dout, ain, bin, a._size);
    cudateardown(d, size, ain, bin, dout);
    return d;
}

Matrix<float> cudamul_coalesce(Matrix<float> &a, Matrix<float> &b)
{
    auto size = a._size;
    Matrix<float> d;
    float *bin;
    float *ain;
    float *dout;
    cudasetup(a, b, ain, bin, dout);
    dim3 gridDim(size / 32, size / 32, 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32 * 32, 1, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    coalesce_matmul<<<gridDim, blockDim>>>(dout, ain, bin, a._size);
    cudateardown(d, size, ain, bin, dout);
    return d;
}




Matrix<float> cudamul_smallblock(Matrix<float> &a, Matrix<float> &b)
{
    auto size = a._size;
    Matrix<float> d;
    float *bin;
    float *ain;
    float *dout;
    cudasetup(a, b, ain, bin, dout);
    dim3 gridDim(size / 8, size / 4, 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(8, 4, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    matmul<<<gridDim, blockDim>>>(dout, ain, bin, a._size);
    cudateardown(d, size, ain, bin, dout);
    return d;
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

float *cudaTransposeInternal(Matrix<float> &bin, size_t &pitch)
{
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


void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};
