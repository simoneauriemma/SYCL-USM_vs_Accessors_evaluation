#include <include.hpp>

// #define DEBUG

// Size of the matrices
// constexpr size_t ROWS = 2048;
// constexpr size_t COLUMNS = 2048;
constexpr size_t WORK_GROUP_SIZE = 32;

int main(int argc, char *argv[]) {

  size_t ROWS = size_t(atoi(argv[1]));
  size_t COLUMNS = size_t(atoi(argv[2]));

  // Variables declaration
  float *matrix_A_from_gpu = (float *)malloc(sizeof(float) * ROWS * COLUMNS);
  float *matrix_B_from_gpu = (float *)malloc(sizeof(float) * ROWS * COLUMNS);
  float *matrix_from_gpu = (float *)malloc(sizeof(float) * ROWS * COLUMNS);

  // Timers
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point end;
  std::chrono::duration<double> elapsed_seconds;

  // Start timer
  start = std::chrono::steady_clock::now();

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
    float *gpu_matrix_a = sycl::malloc_device<float>(ROWS * COLUMNS, queue);
    float *gpu_matrix_b = sycl::malloc_device<float>(ROWS * COLUMNS, queue);
    float *gpu_matrix_c = sycl::malloc_device<float>(ROWS * COLUMNS, queue);

    // Initialize a
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Init_A>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_a[r * ROWS + c] = r + c;
      });
    });

    // Initialize b
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Init_B>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_b[r * ROWS + c] = r + c;
      });
    });

    // Compute c = a + b
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class Compute_C>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);
        gpu_matrix_c[r * ROWS + c] = gpu_matrix_a[r * ROWS + c] + gpu_matrix_b[r * ROWS + c];
      });
    });

    queue.wait();

    // Get execution time
    end = std::chrono::steady_clock::now();

    queue.memcpy(matrix_A_from_gpu, gpu_matrix_a, sizeof(float) * ROWS * COLUMNS).wait();
    queue.memcpy(matrix_B_from_gpu, gpu_matrix_b, sizeof(float) * ROWS * COLUMNS).wait();
    queue.memcpy(matrix_from_gpu, gpu_matrix_c, sizeof(float) * ROWS * COLUMNS).wait();

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
  for (size_t i = 0; i < ROWS; i++) {
    for (size_t j = 0; j < COLUMNS; j++) {
      std::cout << matrix_A_from_gpu[i * ROWS + j] << " ";

      if ((j + 1) % COLUMNS == 0) std::cout << std::endl;
    }
  }

  // Matrice B
  std::cout << std::endl << "Matrice B:" << std::endl;
  for (size_t i = 0; i < ROWS; i++) {
    for (size_t j = 0; j < COLUMNS; j++) {
      std::cout << matrix_B_from_gpu[i * ROWS + j] << " ";

      if ((j + 1) % COLUMNS == 0) std::cout << std::endl;
    }
  }

  // Matrice Result
  std::cout << std::endl << "Result:" << std::endl;
  for (size_t i = 0; i < ROWS; i++) {
    for (size_t j = 0; j < COLUMNS; j++) {
      std::cout << matrix_from_gpu[i * ROWS + j] << " ";

      if ((j + 1) % COLUMNS == 0) std::cout << std::endl;
    }
  }
#endif

  free(matrix_A_from_gpu);
  free(matrix_B_from_gpu);
  free(matrix_from_gpu);

  return 0;
}
