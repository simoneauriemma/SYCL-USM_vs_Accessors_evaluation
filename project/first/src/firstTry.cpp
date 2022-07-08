#include <include.hpp>

int main() {
  constexpr uint N = 5;
  constexpr uint B = 32;

  float *matrix_a = (float *)malloc(sizeof(float) * N * N);
  float *matrix_b = (float *)malloc(sizeof(float) * N * N);
  float *matrix_c = (float *)malloc(sizeof(float) * N * N);

  memset(matrix_a, 1, sizeof(float) * N * N);
  memset(matrix_b, 2, sizeof(float) * N * N);
  memset(matrix_c, 0, sizeof(float) * N * N);

  try {
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue q{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue q{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue q{sycl::host_selector()};
#endif

    sycl::buffer<float, 2> buffer_a(matrix_a, sycl::range<2>{N, N});
    sycl::buffer<float, 2> buffer_b(matrix_b, sycl::range<2>{N, N});
    sycl::buffer<float, 2> buffer_r(matrix_c, sycl::range<2>{N, N});

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 2> acce_a{buffer_a, cgh, sycl::read_only};
      sycl::accessor<float, 2> acce_b{buffer_b, cgh, sycl::read_only};
      sycl::accessor<float, 2> acce_r{buffer_r, cgh, sycl::write_only,
                                      sycl::no_init};

      // cgh.parallel_for(sycl::range<2>({N, N}), [=](sycl::id<2> idx) {
      //   const size_t r = idx.get_global_id(0);
      //   const size_t c = idx.get_global_id(1);
      // });

      cgh.parallel_for(sycl::nd_range<2>{{N, N}, {B, 1}},
                       [=](sycl::nd_item<2> idx) {
                         const size_t r = idx.get_global_id(0);
                         const size_t c = idx.get_global_id(1);

                         float sum = 0.f;
                         for (uint k = 0; k < N; k++) {
                           sum += acce_a[r][k] * acce_b[k][c];
                         }

                         acce_r[r][c] = sum;
                       });
    });
    q.wait();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // // A
  // std::cout << "A: [ ";
  // for (int i = 0; i < 5; i++) std::cout << a[i] << " ";
  // std::cout << "]" << std::endl;

  // // B
  // std::cout << "B: [ ";
  // for (int i = 0; i < 5; i++) std::cout << b[i] << " ";
  // std::cout << "]" << std::endl;

  // // Result
  // std::cout << "Result: [ ";
  // for (int i = 0; i < 5; i++) std::cout << r[i] << " ";
  // std::cout << "]" << std::endl;

  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < n; j++) {
  //     std::cout << mat[i][j] << ' ';
  //   }
  //   std::cout << std::endl;
  // }
  // for (int i = 0; i < N; i++) assert(matrix_c[i] == matrix_a[i] *
  // matrix_b[i]); std::cout << "TEST PASSED" << std::endl;

  // for (int i = 0; i < N; i++) assert(matrix_c[i] == matrix_a[i]
  // *matrix_b[i]); std::cout << "TEST PASSED" << std::endl;
  std::cout << "Result: [ ";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << matrix_c[i * N + j] << " ";
      std::cout << "]" << std::endl;
    }
  }

  std::cout << "Result: [ ";
  // for (int i = 0; i < 5; i++) std::cout << matrix_c[i * N + j] << " ";
  std::cout << "]" << std::endl;
}