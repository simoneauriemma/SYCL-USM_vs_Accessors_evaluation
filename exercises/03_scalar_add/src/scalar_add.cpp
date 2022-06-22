#include <include.hpp>

int main()
{
    int a = 18, b = 13, r = 0;

#if DEVICE_VALUE == CPU_DEVICE
    sycl::queue q{sycl::cpu_selector()};
#elif DEVICE_VALUE == GPU_DEVICE
    sycl::queue q{sycl::gpu_selector()};
#elif DEVICE_VALUE == HOST_DEVICE
    sycl::queue q{sycl::host_selector()};
#endif

    sycl::buffer<int, 1> buffer_a(&a, sycl::range{1});
    sycl::buffer<int, 1> buffer_b(&b, sycl::range{1});
    sycl::buffer<int, 1> buffer_r(&r, sycl::range{1});

    q.submit([&](sycl::handler &cgh)
             {
        sycl::accessor accessor_a{buffer_a, cgh, sycl::read_only};
        sycl::accessor accessor_b{buffer_b, cgh, sycl::read_only};
        sycl::accessor accessor_r{buffer_r, cgh, sycl::write_only, sycl::no_init};

        cgh.single_task(
            [=]
            { accessor_r[0] = accessor_a[0] + accessor_b[0]; }); })
        .wait();

    std::cout << "Result: " << r << std::endl;
}