#include <include.hpp>

// #define DEBUG

// Size of the matrices
constexpr size_t N = 4096;
constexpr size_t M = 4096;

int main() {
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

    // Create some 2D buffers of float for our matrices
    sycl::buffer<float, 2> buffer_a{sycl::range<2>{N, M}};
    sycl::buffer<float, 2> buffer_b{sycl::range<2>{N, M}};
    sycl::buffer<float, 2> buffer_c{sycl::range<2>{N, M}};

    // Start timer
    start = std::chrono::steady_clock::now();

    // Initialize a
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor A{buffer_a, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(sycl::range<2>{N, M}, [=](sycl::id<2> index) { A[index] = index[0] * 2 + index[1]; });
    });

    // Launch an asynchronous kernel to initialize b
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor B{buffer_b, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(sycl::range<2>{N, M}, [=](sycl::id<2> index) { B[index] = index[0] * 2014 + index[1] * 42; });
    });

    // Launch an asynchronous kernel to compute matrix addition c = a + b
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor A{buffer_a, cgh, sycl::read_only};
      sycl::accessor B{buffer_b, cgh, sycl::read_only};
      sycl::accessor C{buffer_c, cgh, sycl::write_only, sycl::no_init};

      cgh.parallel_for(sycl::range<2>{N, M}, [=](sycl::id<2> index) { C[index] = A[index] + B[index]; });
    });

    queue.wait();

    // Get execution time
    end = std::chrono::steady_clock::now();

    // Ask for an accessor to read c from application scope. The SYCL runtime
    // waits for c to be ready before returning from the constructor
    sycl::host_accessor A{buffer_a, sycl::read_only};
    sycl::host_accessor B{buffer_b, sycl::read_only};
    sycl::host_accessor C{buffer_c, sycl::read_only};

#ifdef DEBUG
    // Matrice A
    std::cout << std::endl << "Matrice A:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {

        std::cout << A[i][j] << " ";
        if ((j + 1) % M == 0) std::cout << std::endl;
      }
    }

    // Matrice B
    std::cout << std::endl << "Matrice B:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {

        std::cout << B[i][j] << " ";
        if ((j + 1) % M == 0) std::cout << std::endl;
      }
    }

    // Matrice Result
    std::cout << std::endl << "Result:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {

        std::cout << C[i][j] << " ";
        if ((j + 1) % M == 0) std::cout << std::endl;

        // Error check
        if (C[i][j] != i * (2 + 2014) + j * (1 + 42)) {
          std::cout << "Wrong value " << C[i][j] << " on element " << i << " " << j << std::endl;
          exit(-1);
        }
      }
    }
#endif

  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  return 0;
}