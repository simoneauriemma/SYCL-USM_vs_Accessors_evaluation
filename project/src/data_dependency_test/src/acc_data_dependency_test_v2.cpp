#include <include.hpp>

// #define DEBUG

#define SIZE 30

int main() {

  // CPU Variables declaration
  float *variables = (float *)malloc(SIZE * sizeof(float));
  // float *output = (float *)malloc(SIZE * sizeof(float));

  // Init
  for (int i = 0; i < SIZE; i++) {
    variables[i] = i + 1;
  }

  // Timers
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point end;
  std::chrono::duration<float> elapsed_seconds;

  // Start timer
  start = std::chrono::steady_clock::now();
  sycl::buffer<float, 1> buffer_out{sycl::range<1>{SIZE}};

  try {
    // Device selector
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue queue{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue queue{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue queue{sycl::host_selector()};
#endif

    // Buffers
    sycl::buffer<float, 1> buffer_var(variables, sycl::range{SIZE});
    // sycl::buffer<float, 1> buffer_out(output, sycl::range{SIZE});

    // Queue - FIRST TASK
    queue.submit([&](sycl::handler &cgh) {
      // Accessors
      sycl::accessor accessor_var{buffer_var, cgh, sycl::read_only};
      sycl::accessor accessor_out{buffer_out, cgh, sycl::read_write, sycl::no_init};

      // Execute kernel
      cgh.parallel_for<class Init_out>(sycl::range<1>{SIZE}, [=](sycl::id<1> idx) {
        // Init accessor_out[idx]
        accessor_out[idx] = accessor_var[idx];

        int index = (int(idx) == 0) ? 0 : (int)idx - 1;
        accessor_out[idx] += (accessor_out[index] * accessor_var[index]) / (int(idx) + 1);
      });
    });

    // Queue - SECOND TASK
    queue.submit([&](sycl::handler &cgh) {
      // Accessors
      sycl::accessor accessor_out{buffer_out, cgh, sycl::read_write};

      // Execute kernel
      cgh.parallel_for<class Compute>(sycl::range<1>{SIZE}, [=](sycl::id<1> idx) {
        // Compute
        accessor_out[idx] = accessor_out[(idx - 1) % SIZE] + accessor_out[(idx)] + accessor_out[(idx + 1) % SIZE];
      });
    });

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  sycl::host_accessor host_accessor_output{buffer_out, sycl::read_only};
  // Stampa finale
  for (int i = 0; i < SIZE; i++) {
    std::cout << "Variabile var_" << i << " - valore: " << std::setprecision(10) << host_accessor_output[i] << std::endl;
  }
}