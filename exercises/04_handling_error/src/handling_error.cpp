#include <include.hpp>

int main() {
    int a = 18, b = 13, r = 0;

    try {
#if DEVICE_VALUE == CPU_DEVICE
        sycl::queue q{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
        sycl::queue q{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
        sycl::queue q{sycl::host_selector()};
#endif
        {
            sycl::buffer<int, 1> buffer_a(&a, sycl::range{1});
            sycl::buffer<int, 1> buffer_b(&b, sycl::range{1});
            sycl::buffer<int, 1> buffer_r(&r, sycl::range{1});

            q.submit([&](sycl::handler& cgh) {
        sycl::accessor acce_a{buffer_a, cgh, sycl::read_only};
        sycl::accessor acce_b{buffer_b, cgh, sycl::read_only};
        sycl::accessor acce_r{buffer_r, cgh, sycl::write_only};

        cgh.single_task(
            [=]
            { acce_r[0] = acce_a[0] + acce_b[0]; }); });
        }
    } catch (const sycl::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Result: " << r << std::endl;
}