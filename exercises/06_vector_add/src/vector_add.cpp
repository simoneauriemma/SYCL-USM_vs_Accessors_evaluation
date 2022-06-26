#include <include.hpp>

int main() {
  std::vector<int> a(5, 2);
  std::vector<int> b(5, 1);
  std::vector<int> r(5);

  for (int i = 0; i < 5; i++) {
    a[i] = rand() % (10 - 0 + 1) + 0;
    b[i] = rand() % (10 - 0 + 1) + 0;
  }

  try {
#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue q{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue q{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue q{sycl::host_selector()};
#endif

    sycl::buffer<int, 1> buffer_a(a.data(), a.size());
    sycl::buffer<int, 1> buffer_b(b.data(), 5);
    sycl::buffer<int, 1> buffer_r(r.data(), 5);

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acce_a{buffer_a, cgh, sycl::read_only};
      sycl::accessor acce_b{buffer_b, cgh, sycl::read_only};
      sycl::accessor acce_r{buffer_r, cgh, sycl::write_only, sycl::no_init};

      cgh.parallel_for(sycl::range(5), [=](sycl::id<1> idx) {
        acce_r[idx] = acce_a[idx] + acce_b[idx];
      });
    });
    q.wait();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // A
  std::cout << "A: [ ";
  for (int i = 0; i < 5; i++) std::cout << a[i] << " ";
  std::cout << "]" << std::endl;

  // B
  std::cout << "B: [ ";
  for (int i = 0; i < 5; i++) std::cout << b[i] << " ";
  std::cout << "]" << std::endl;

  // Result
  std::cout << "Result: [ ";
  for (int i = 0; i < 5; i++) std::cout << r[i] << " ";
  std::cout << "]" << std::endl;

  for (int i = 0; i < 5; i++) assert(r[i] == a[i] + b[i]);
  std::cout << "TEST PASSED" << std::endl;
}