#include <include.hpp>

#define DEBUG

// Size of the matrices
constexpr size_t ROWS = 2048;
constexpr size_t COLUMNS = 2048;
constexpr size_t WORK_GROUP_SIZE = 32;

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
    sycl::buffer<float, 2> buffer_a{sycl::range<2>{ROWS, COLUMNS}};
    sycl::buffer<float, 2> buffer_b{sycl::range<2>{ROWS, COLUMNS}};
    sycl::buffer<float, 2> buffer_c{sycl::range<2>{ROWS, COLUMNS}};

    // Start timer
    start = std::chrono::steady_clock::now();

    // Initialize a
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 2> A{buffer_a, cgh, sycl::write_only, sycl::no_init};

      cgh.parallel_for<class Init1>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);

        float sum = 0.f;
        sum = r + c;

        A[r][c] = sum;
      });
    });

    // Launch an asynchronous kernel to initialize b
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 2> B{buffer_b, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for<class Init2>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);

        float sum = 0.f;
        sum = r + c;

        B[r][c] = sum;
      });
    });

    // Launch an asynchronous kernel to compute matrix addition c = a + b
    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor<float, 2> A{buffer_a, cgh, sycl::read_only};
      sycl::accessor<float, 2> B{buffer_b, cgh, sycl::read_only};
      sycl::accessor<float, 2> C{buffer_c, cgh, sycl::write_only, sycl::no_init};

      cgh.parallel_for<class Compute>(sycl::nd_range<2>{{ROWS, COLUMNS}, {WORK_GROUP_SIZE, WORK_GROUP_SIZE}}, [=](sycl::nd_item<2> item) {
        const size_t r = item.get_global_id(0);
        const size_t c = item.get_global_id(1);

        float sum = 0.f;
        sum = A[r][c] + B[r][c];

        C[r][c] = sum;
      });
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
    // // Matrice A
    // std::cout << std::endl << "Matrice A:" << std::endl;
    // for (size_t i = 0; i < ROWS; i++) {
    //   for (size_t j = 0; j < COLUMNS; j++) {

    //     std::cout << A[i][j] << " ";
    //     if ((j + 1) % COLUMNS == 0) std::cout << std::endl;
    //   }
    // }

    // // Matrice B
    // std::cout << std::endl << "Matrice B:" << std::endl;
    // for (size_t i = 0; i < ROWS; i++) {
    //   for (size_t j = 0; j < COLUMNS; j++) {

    //     std::cout << B[i][j] << " ";
    //     if ((j + 1) % COLUMNS == 0) std::cout << std::endl;
    //   }
    // }

    // Matrice Result
    std::cout << std::endl << "Result:" << std::endl;
    for (size_t i = 0; i < ROWS; i++) {
      for (size_t j = 0; j < COLUMNS; j++) {

        // std::cout << C[i][j] << " ";
        // if ((j + 1) % COLUMNS == 0) std::cout << std::endl;

        // Error check
        if (C[i][j] != (i + j) * 2) {
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