#include <include.hpp>

#define SIZE 32

int main() {
  // Initial Allocations
  std::vector<float> a(SIZE);
  std::vector<float> b(SIZE);
  std::vector<float> r(SIZE);
  std::vector<float> r_from_GPU(SIZE);

  for (int i = 0; i < SIZE; i++) {
    a[i] = rand() % (10 - 0 + 1) + 0;
    b[i] = rand() % (10 - 0 + 1) + 0;
  }

  try {
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue queue{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue queue{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue queue{sycl::host_selector()};
#endif

    float *gpu_a = sycl::malloc_device<float>(SIZE, queue);
    float *gpu_b = sycl::malloc_device<float>(SIZE, queue);
    float *gpu_r = sycl::malloc_device<float>(SIZE, queue);

    queue.memcpy(gpu_a, a.data(), sizeof(float) * SIZE);
    queue.memcpy(gpu_b, b.data(), sizeof(float) * SIZE);
    queue.memcpy(gpu_r, r.data(), sizeof(float) * SIZE);
    queue.wait();

    {
      queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range(SIZE), [=](sycl::id<1> idx) {
          gpu_r[idx] = gpu_a[idx] + gpu_b[idx];
        });
      });
      queue.wait();

      queue.memcpy(r_from_GPU.data(), gpu_r, sizeof(float) * SIZE);
      queue.wait();

      sycl::free(gpu_a, queue);
      sycl::free(gpu_b, queue);
      sycl::free(gpu_r, queue);
    }
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  if (SIZE <= 32) {
    // A
    std::cout << "A: \t[ ";
    for (int i = 0; i < SIZE; i++) std::cout << a[i] << " ";
    std::cout << "]" << std::endl;

    // B
    std::cout << "B: \t[ ";
    for (int i = 0; i < SIZE; i++) std::cout << b[i] << " ";
    std::cout << "]" << std::endl;

    // Result
    std::cout << "Result from GPU: [ ";
    for (int i = 0; i < SIZE; i++) std::cout << r_from_GPU[i] << " ";
    std::cout << "]" << std::endl;
  }

  for (int i = 0; i < SIZE; i++) assert(r_from_GPU[i] == a[i] + b[i]);
  std::cout << "TEST PASSED" << std::endl;
}