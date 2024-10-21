#include <iostream>
#include <nn.hpp>

using namespace nn;

int main() {
  int n = 10;
  float *result = new float[n];
  float *d_x;
  float *d_y;
  int block_size = 10;
  float value = 1.0f;

  InitDriver(0);

  InitArray(n, d_x, block_size, value);
  InitArray(n, d_y, block_size, value);

  MemcpyDeviceToHost(n, d_y, result);
  std::cout << "Array y contents before: ";
  for (int i = 0; i < n; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  AddArray(n, d_x, d_y, block_size);

  MemcpyDeviceToHost(n, d_y, result);
  std::cout << "Array y contents after adding: ";
  for (int i = 0; i < n; ++i) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
