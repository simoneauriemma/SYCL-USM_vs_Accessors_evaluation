#include <include.hpp>

int main() {
    std::vector<int> a(5, 1);
    std::vector<int> b(5, 1);
    std::vector<int> r(5);

    try {
#if DEVICE_VALUE == CPU_DEVICE
        sycl::queue q{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
        sycl::queue q{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
        sycl::queue q{sycl::host_selector()};
#endif
        {
            sycl::buffer<int, 1> buffer_a(a.data(), a.size());
            sycl::buffer<int, 1> buffer_b(b.data(), 5);
            sycl::buffer<int, 1> buffer_r(r.data(), 5);

            q.submit([&](sycl::handler& cgh) {
        sycl::accessor acce_a{buffer_a, cgh, sycl::read_only};
        sycl::accessor acce_b{buffer_b, cgh, sycl::read_only};
        sycl::accessor acce_r{buffer_r, cgh, sycl::write_only, sycl::no_init};

        cgh.parallel_for(sycl::range(5),
            [=](sycl::id<1>idx)
            { acce_r[idx] = acce_a[idx] + acce_b[idx]; }); }).wait();
        }

    } catch (const sycl::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
    std::cout << "Result: " << r[0] << std::endl;

    for (int i = 1; i < 5; i++)
        std::cout << r[i] << std::endl;
}