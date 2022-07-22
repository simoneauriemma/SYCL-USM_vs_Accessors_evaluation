#include <include.hpp>

// #define DEBUG

// Size of the matrices
constexpr size_t N = 4096;
constexpr size_t M = 4096;

int main() {

  float *matrix_A_from_gpu = (float *)malloc(sizeof(float) * N * M);
  float *matrix_B_from_gpu = (float *)malloc(sizeof(float) * N * M);
  float *matrix_from_gpu = (float *)malloc(sizeof(float) * N * M);

  // Timers
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point end;
  std::chrono::duration<double> elapsed_seconds;

  try {
    // Create a queue to work on
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue queue{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue queue{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue queue{sycl::host_selector()};
#endif

    // Gpu Variables delcaration
    float *gpu_matrix_a = sycl::malloc_device<float>(N * M, queue);
    float *gpu_matrix_b = sycl::malloc_device<float>(N * M, queue);
    float *gpu_matrix_c = sycl::malloc_device<float>(N * M, queue);

    // Start timer
    start = std::chrono::steady_clock::now();

    // Initialize a
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Init1>(sycl::nd_range<2>{{N, M}, {1, 1}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_a[r * N + c] = r + c;
      });
    });

    // Initialize b
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Init2>(sycl::nd_range<2>{{N, M}, {1, 1}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_b[r * N + c] = r + c;
      });
    });

    // Compute c
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Compute>(sycl::nd_range<2>{{N, M}, {1, 1}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_c[r * N + c] = gpu_matrix_a[r * N + c] + gpu_matrix_b[r * N + c];
      });
    });

    queue.wait();

    // Get execution time
    end = std::chrono::steady_clock::now();

    queue.memcpy(matrix_A_from_gpu, gpu_matrix_a, sizeof(float) * N * M).wait();
    queue.memcpy(matrix_B_from_gpu, gpu_matrix_b, sizeof(float) * N * M).wait();
    queue.memcpy(matrix_from_gpu, gpu_matrix_c, sizeof(float) * N * M).wait();

    sycl::free(gpu_matrix_a, queue);
    sycl::free(gpu_matrix_b, queue);
    sycl::free(gpu_matrix_c, queue);

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

#ifdef DEBUG
  // Matrice A
  std::cout << std::endl << "Matrice A:" << std::endl;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      std::cout << matrix_A_from_gpu[i * N + j] << " ";

      if ((j + 1) % M == 0) std::cout << std::endl;
    }
  }

  // Matrice B
  std::cout << std::endl << "Matrice B:" << std::endl;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      std::cout << matrix_B_from_gpu[i * N + j] << " ";

      if ((j + 1) % M == 0) std::cout << std::endl;
    }
  }

  // Matrice Result
  std::cout << std::endl << "Result:" << std::endl;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      std::cout << matrix_from_gpu[i * N + j] << " ";

      if ((j + 1) % M == 0) std::cout << std::endl;
    }
  }
#endif

  free(matrix_A_from_gpu);
  free(matrix_B_from_gpu);
  free(matrix_from_gpu);

  return 0;
}
