#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <nn_kernel.cuh>
#include <stdio.h>

__global__ void add(int n, float *arrayX, float *arrayY) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    arrayY[index] = arrayX[index] + arrayY[index];
}

__global__ void initArrayKernel(int n, float *arr, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    arr[idx] = value;
  }
}

void KernelAdd(int n, float *arrayX, float *arrayY, int blockSize) {
  int numBlocks = (n + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(n, arrayX, arrayY);
  cudaDeviceSynchronize();
}

void KernelInit(int n, float *arr, int blockSize, float value) {
  int numBlocks = (n + blockSize - 1) / blockSize;
  cudaMalloc((void **)&arr, n * sizeof(float));
  initArrayKernel<<<numBlocks, blockSize>>>(n, arr, value);
  cudaDeviceSynchronize();
}

void KernelMemcpy(int n, float *arrayS, float *arrayD) {
  cudaMemcpy(arrayD, arrayS, n * sizeof(float), cudaMemcpyDeviceToHost);
}

void KernleInitDriver(int index) {
  cudaError_t cudaStatus = cudaSetDevice(index);
  if (cudaStatus != cudaSuccess) {
    printf("Error: cudaSetDevice failed: %s", cudaGetErrorString(cudaStatus));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("Using device: %s\n", deviceProp.name);
}
