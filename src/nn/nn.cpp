#include <nn.hpp>
#include <nn_kernel.cuh>

namespace nn {

void InitDriver(int index) { KernleInitDriver(index); }
void InitArray(int n, float *arr, int blockSize, float value) {
  KernelInit(n, arr, blockSize, value);
}
void AddArray(int n, float *x, float *y, int blockSize) {
  KernelAdd(n, x, y, blockSize);
}
void MemcpyDeviceToHost(int n, float *s, float *d) { KernelMemcpy(n, s, d); }

} // namespace nn
