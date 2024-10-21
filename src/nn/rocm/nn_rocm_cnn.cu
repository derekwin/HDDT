#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <nn_kernel.cuh>
#include <stdio.h>

__global__ void add(int n, float *arrayX, float *arrayY) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    arrayY[index] = arrayX[index] + arrayY[index];
}

__global__ void initArrayKernel(int n, float *arr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] = static_cast<float>(idx) * 0.1f; // 初始化每个元素
  }
}

void KernelAdd(int n, float *arrayX, float *arrayY, int blockSize) {
  int numBlocks = (n + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(n, arrayX, arrayY);
  cudaDeviceSynchronize();
}

void KernelInit(int n, float *arr, int blockSize) {
  int numBlocks = (n + blockSize - 1) / blockSize;
  initArrayKernel<<<numBlocks, blockSize>>>(n, arr);
}