#include <include.hpp>

// #define DEBUG

#define SIZE 30

int main() {

  // CPU Variables declaration
  float *variables = (float *)malloc(SIZE * sizeof(float));

  // From GPU Variables declaration
  float *output_fromGpu = (float *)malloc(SIZE * sizeof(float));

  // Init CPU Variables
  for (int i = 0; i < SIZE; i++) {
    variables[i] = i + 1;
  }

  // Timers
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point end;
  std::chrono::duration<float> elapsed_seconds;

  try {
    // Device selector
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue queue{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue queue{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue queue{sycl::host_selector()};
#endif

    // GPU Variables declaration
    float *variables_Gpu = sycl::malloc_device<float>(SIZE, queue);
    float *output_Gpu = sycl::malloc_device<float>(SIZE, queue);

    // Copy data from CPU to GPU
    queue.memcpy(variables_Gpu, variables, SIZE * sizeof(float)).wait();

    // Start timer
    start = std::chrono::steady_clock::now();

    // Queue - FIRST TASK
    queue.submit([&](sycl::handler &cgh) {
      // Execute kernel
      cgh.parallel_for<class Init_out>(sycl::range<1>{SIZE}, [=](sycl::id<1> idx) {
        // Init accessor_out[idx]
        output_Gpu[idx] = variables_Gpu[idx];

        // Compute
        int index = (int(idx) == 0) ? 0 : (int)idx - 1;
        output_Gpu[idx] += (output_Gpu[index] * variables_Gpu[index]) / (int(idx) + 1);
      });
    });

    // Queue - SECOND TASK
    queue.submit([&](sycl::handler &cgh) {
      // Execute kernel
      cgh.parallel_for<class Compute>(sycl::range<1>{SIZE}, [=](sycl::id<1> idx) {
        // Compute
        output_Gpu[idx] = output_Gpu[(idx - 1) % SIZE] + output_Gpu[(idx)] + output_Gpu[(idx + 1) % SIZE];
      });
    });

    queue.memcpy(output_fromGpu, output_Gpu, SIZE * sizeof(float)).wait();
    queue.wait();

    sycl::free(variables_Gpu, queue);
    sycl::free(output_Gpu, queue);

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  // Stampa finale
  for (int i = 0; i < SIZE; i++) {
    std::cout << "Variabile var_" << i << " - valore: " << std::setprecision(10) << output_fromGpu[i] << std::endl;
  }
}