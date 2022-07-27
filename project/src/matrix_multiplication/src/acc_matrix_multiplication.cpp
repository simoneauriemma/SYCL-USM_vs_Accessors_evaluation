#include <include.hpp>

// #define DEBUG

void init_matrix(int, int, float *);
void print_matrix(int, int, float *, std::string);
void sequential_matrix_multiplication(int, int, float *, float *, float *);
int check_is_correct(int, int, float *, float *);

int main() {
  // Settings
  constexpr uint N = 4096;
  constexpr uint B = 128;

  // Variables declaration
  float *matrix_a = (float *)malloc(sizeof(float) * N * N);
  float *matrix_b = (float *)malloc(sizeof(float) * N * N);
  float *matrix_c = (float *)malloc(sizeof(float) * N * N);
  float *matrix_sequential = (float *)malloc(sizeof(float) * N * N);

  int res = 0;

  // Init
  init_matrix(N, N, matrix_a);
  init_matrix(N, N, matrix_b);
  memset(matrix_c, 0, sizeof(float) * N * N);
  memset(matrix_sequential, 0, sizeof(float) * N * N);

  // Timers
  std::chrono::_V2::steady_clock::time_point start;
  std::chrono::_V2::steady_clock::time_point end;
  std::chrono::duration<double> elapsed_seconds;

  try {
    // Device selector
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue queue{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue queue{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue queue{sycl::host_selector()};
#endif

    // std::cout << "- Execution on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    // Buffers
    sycl::buffer<float, 2> buffer_a(matrix_a, sycl::range<2>{N, N});
    sycl::buffer<float, 2> buffer_b(matrix_b, sycl::range<2>{N, N});
    sycl::buffer<float, 2> buffer_r(matrix_c, sycl::range<2>{N, N});

    // Queue
    queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<float, 2> accessor_a{buffer_a, cgh, sycl::read_only};
          sycl::accessor<float, 2> accessor_b{buffer_b, cgh, sycl::read_only};
          sycl::accessor<float, 2> accessor_r{buffer_r, cgh, sycl::write_only, sycl::no_init};

          // Start timer
          start = std::chrono::steady_clock::now();

          cgh.parallel_for(sycl::nd_range<2>{{N, N}, {B, B}}, [=](sycl::nd_item<2> idx) {
            const size_t r = idx.get_global_id(0);
            const size_t c = idx.get_global_id(1);

            float sum = 0.f;
            for (uint k = 0; k < N; k++) {
              sum += accessor_a[r][k] * accessor_b[k][c];
            }

            accessor_r[r][c] = sum;
          });
        })
        .wait();

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  // // Sequential matrix multiplication (for test)
  // std::cout << std::endl << "- Doing sequential execution..." << std::endl;
  // sequential_matrix_multiplication(N, N, matrix_a, matrix_b, matrix_sequential);
  // std::cout << "- Sequential execution done" << std::endl;

  // // Check if is correct
  // res = check_is_correct(N, N, matrix_c, matrix_sequential);

#ifdef DEBUG
  print_matrix(N, N, matrix_a, "MATRICE A");
  print_matrix(N, N, matrix_b, "MATRICE B");
  print_matrix(N, N, matrix_c, "MATRICE C");
  print_matrix(N, N, matrix_sequential, "MATRICE SEQUENZIALE");
#endif

  // std::cout << std::endl << "RESULT: " << ((res == 1) ? "CORRECT" : "INCORRECT!") << std::endl;

  free(matrix_a);
  free(matrix_b);
  free(matrix_c);
  free(matrix_sequential);
}

void init_matrix(int rows_size, int columns_size, float *matrix) {
  int i;

  for (i = 0; i < rows_size * columns_size; i++) {
    *(matrix + i) = std::rand() % 10 + 1;
  }
}

void print_matrix(int rows_size, int columns_size, float *matrix, std::string matrix_name) {
  int i;

  std::cout << std::endl << matrix_name << std::endl;
  for (i = 0; i < rows_size * columns_size; i++) {
#ifdef DEBUG
    std::cout << "| " << *(matrix + i) << " ";

    if ((i + 1) % columns_size == 0 && i != 0) std::cout << "| " << std::endl;
#endif
  }
}

void sequential_matrix_multiplication(int rows_size, int columns_size, float *matrix_a, float *matrix_b, float *matrix_res) {
  int i, j, k;

  for (i = 0; i < rows_size; i++) {
    for (j = 0; j < columns_size; j++) {
      for (k = 0; k < columns_size; k++) {
        matrix_res[i * rows_size + j] += matrix_a[i * rows_size + k] * matrix_b[k * rows_size + j];
      }
    }
  }
}

int check_is_correct(int rows_size, int columns_size, float *matrix_parallel, float *matrix_sequential) {
  int i;
  int res = 1;

  for (int i = 0; i < rows_size * columns_size; i++)
    if (matrix_parallel[i] != matrix_sequential[i]) res = 0;

  return res;
}