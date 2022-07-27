#include <include.hpp>

// #define DEBUG

int main() {

  // Variables declaration
  float var_0 = 1.0;
  float var_1 = 2.0;
  float var_2 = 3.0;
  float var_3 = 4.0;
  float var_4 = 5.0;
  float var_5 = 6.0;
  float var_6 = 7.0;
  float var_7 = 8.0;
  float var_8 = 9.0;
  float var_9 = 10.0;
  float var_10 = 11.0;
  float var_11 = 12.0;
  float var_12 = 13.0;
  float var_13 = 14.0;
  float var_14 = 15.0;
  float var_15 = 16.0;
  float var_16 = 17.0;
  float var_17 = 18.0;
  float var_18 = 19.0;
  float var_19 = 20.0;
  float var_20 = 21.0;
  float var_21 = 22.0;
  float var_22 = 23.0;
  float var_23 = 24.0;
  float var_24 = 25.0;
  float var_25 = 26.0;
  float var_26 = 27.0;
  float var_27 = 28.0;
  float var_28 = 29.0;
  float var_29 = 30.0;
  float var_30 = 31.0;

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

    // std::cout << "- Execution on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    // Buffers
    sycl::buffer<float, 1> buffer_0(&var_0, sycl::range{1});
    sycl::buffer<float, 1> buffer_1(&var_1, sycl::range{1});
    sycl::buffer<float, 1> buffer_2(&var_2, sycl::range{1});
    sycl::buffer<float, 1> buffer_3(&var_3, sycl::range{1});
    sycl::buffer<float, 1> buffer_4(&var_4, sycl::range{1});
    sycl::buffer<float, 1> buffer_5(&var_5, sycl::range{1});
    sycl::buffer<float, 1> buffer_6(&var_6, sycl::range{1});
    sycl::buffer<float, 1> buffer_7(&var_7, sycl::range{1});
    sycl::buffer<float, 1> buffer_8(&var_8, sycl::range{1});
    sycl::buffer<float, 1> buffer_9(&var_9, sycl::range{1});
    sycl::buffer<float, 1> buffer_10(&var_10, sycl::range{1});
    sycl::buffer<float, 1> buffer_11(&var_11, sycl::range{1});
    sycl::buffer<float, 1> buffer_12(&var_12, sycl::range{1});
    sycl::buffer<float, 1> buffer_13(&var_13, sycl::range{1});
    sycl::buffer<float, 1> buffer_14(&var_14, sycl::range{1});
    sycl::buffer<float, 1> buffer_15(&var_15, sycl::range{1});
    sycl::buffer<float, 1> buffer_16(&var_16, sycl::range{1});
    sycl::buffer<float, 1> buffer_17(&var_17, sycl::range{1});
    sycl::buffer<float, 1> buffer_18(&var_18, sycl::range{1});
    sycl::buffer<float, 1> buffer_19(&var_19, sycl::range{1});
    sycl::buffer<float, 1> buffer_20(&var_20, sycl::range{1});
    sycl::buffer<float, 1> buffer_21(&var_21, sycl::range{1});
    sycl::buffer<float, 1> buffer_22(&var_22, sycl::range{1});
    sycl::buffer<float, 1> buffer_23(&var_23, sycl::range{1});
    sycl::buffer<float, 1> buffer_24(&var_24, sycl::range{1});
    sycl::buffer<float, 1> buffer_25(&var_25, sycl::range{1});
    sycl::buffer<float, 1> buffer_26(&var_26, sycl::range{1});
    sycl::buffer<float, 1> buffer_27(&var_27, sycl::range{1});
    sycl::buffer<float, 1> buffer_28(&var_28, sycl::range{1});
    sycl::buffer<float, 1> buffer_29(&var_29, sycl::range{1});
    sycl::buffer<float, 1> buffer_30(&var_30, sycl::range{1});

    // Queue
    queue
        .submit([&](sycl::handler &cgh) {
          // Accessors
          sycl::accessor accessor_0{buffer_0, cgh, sycl::read_write};
          sycl::accessor accessor_1{buffer_1, cgh, sycl::read_write};
          sycl::accessor accessor_2{buffer_2, cgh, sycl::read_write};
          sycl::accessor accessor_3{buffer_3, cgh, sycl::read_write};
          sycl::accessor accessor_4{buffer_4, cgh, sycl::read_write};
          sycl::accessor accessor_5{buffer_5, cgh, sycl::read_write};
          sycl::accessor accessor_6{buffer_6, cgh, sycl::read_write};
          sycl::accessor accessor_7{buffer_7, cgh, sycl::read_write};
          sycl::accessor accessor_8{buffer_8, cgh, sycl::read_write};
          sycl::accessor accessor_9{buffer_9, cgh, sycl::read_write};
          sycl::accessor accessor_10{buffer_10, cgh, sycl::read_write};
          sycl::accessor accessor_11{buffer_11, cgh, sycl::read_write};
          sycl::accessor accessor_12{buffer_12, cgh, sycl::read_write};
          sycl::accessor accessor_13{buffer_13, cgh, sycl::read_write};
          sycl::accessor accessor_14{buffer_14, cgh, sycl::read_write};
          sycl::accessor accessor_15{buffer_15, cgh, sycl::read_write};
          sycl::accessor accessor_16{buffer_16, cgh, sycl::read_write};
          sycl::accessor accessor_17{buffer_17, cgh, sycl::read_write};
          sycl::accessor accessor_18{buffer_18, cgh, sycl::read_write};
          sycl::accessor accessor_19{buffer_19, cgh, sycl::read_write};
          sycl::accessor accessor_20{buffer_20, cgh, sycl::read_write};
          sycl::accessor accessor_21{buffer_21, cgh, sycl::read_write};
          sycl::accessor accessor_22{buffer_22, cgh, sycl::read_write};
          sycl::accessor accessor_23{buffer_23, cgh, sycl::read_write};
          sycl::accessor accessor_24{buffer_24, cgh, sycl::read_write};
          sycl::accessor accessor_25{buffer_25, cgh, sycl::read_write};
          sycl::accessor accessor_26{buffer_26, cgh, sycl::read_write};
          sycl::accessor accessor_27{buffer_27, cgh, sycl::read_write};
          sycl::accessor accessor_28{buffer_28, cgh, sycl::read_write};
          sycl::accessor accessor_29{buffer_29, cgh, sycl::read_write};
          sycl::accessor accessor_30{buffer_30, cgh, sycl::read_write};

          // Start timer
          start = std::chrono::steady_clock::now();

          // Execute kernel
          cgh.parallel_for(sycl::range<1>(10000), [=](sycl::id<1> idx) {
            accessor_0[0] += accessor_30[0] * accessor_0[0];
            accessor_1[0] += accessor_29[0] * accessor_1[0];
            accessor_2[0] += accessor_28[0] * accessor_2[0];
            accessor_3[0] += accessor_27[0] * accessor_3[0];
            accessor_4[0] += accessor_26[0] * accessor_4[0];
            accessor_5[0] += accessor_25[0] * accessor_5[0];
            accessor_6[0] += accessor_24[0] * accessor_6[0];
            accessor_7[0] += accessor_23[0] * accessor_7[0];
            accessor_8[0] += accessor_22[0] * accessor_8[0];
            accessor_9[0] += accessor_21[0] * accessor_9[0];
            accessor_10[0] += accessor_20[0] * accessor_10[0];
            accessor_11[0] += accessor_19[0] * accessor_11[0];
            accessor_12[0] += accessor_18[0] * accessor_12[0];
            accessor_13[0] += accessor_17[0] * accessor_13[0];
            accessor_14[0] += accessor_16[0] * accessor_14[0];
            accessor_15[0] += accessor_15[0] * accessor_15[0];
            accessor_16[0] += accessor_14[0] * accessor_16[0];
            accessor_17[0] += accessor_13[0] * accessor_17[0];
            accessor_18[0] += accessor_12[0] * accessor_18[0];
            accessor_19[0] += accessor_11[0] * accessor_19[0];
            accessor_20[0] += accessor_10[0] * accessor_20[0];
            accessor_21[0] += accessor_9[0] * accessor_21[0];
            accessor_22[0] += accessor_8[0] * accessor_22[0];
            accessor_23[0] += accessor_7[0] * accessor_23[0];
            accessor_24[0] += accessor_6[0] * accessor_24[0];
            accessor_25[0] += accessor_5[0] * accessor_25[0];
            accessor_26[0] += accessor_4[0] * accessor_26[0];
            accessor_27[0] += accessor_3[0] * accessor_27[0];
            accessor_28[0] += accessor_2[0] * accessor_28[0];
            accessor_29[0] += accessor_1[0] * accessor_29[0];
            accessor_30[0] += accessor_0[0] * accessor_30[0];
          });
        })
        .wait();

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Stampa finale
  std::cout << "Variabile var_0 - valore: " << std::setprecision(10) << var_0 << std::endl;
  std::cout << "Variabile var_1 - valore: " << std::setprecision(10) << var_1 << std::endl;
  std::cout << "Variabile var_2 - valore: " << std::setprecision(10) << var_2 << std::endl;
  std::cout << "Variabile var_3 - valore: " << std::setprecision(10) << var_3 << std::endl;
  std::cout << "Variabile var_4 - valore: " << std::setprecision(10) << var_4 << std::endl;
  std::cout << "Variabile var_5 - valore: " << std::setprecision(10) << var_5 << std::endl;
  std::cout << "Variabile var_6 - valore: " << std::setprecision(10) << var_6 << std::endl;
  std::cout << "Variabile var_7 - valore: " << std::setprecision(10) << var_7 << std::endl;
  std::cout << "Variabile var_8 - valore: " << std::setprecision(10) << var_8 << std::endl;
  std::cout << "Variabile var_9 - valore: " << std::setprecision(10) << var_9 << std::endl;
  std::cout << "Variabile var_10 - valore: " << std::setprecision(10) << var_10 << std::endl;
  std::cout << "Variabile var_11 - valore: " << std::setprecision(10) << var_11 << std::endl;
  std::cout << "Variabile var_12 - valore: " << std::setprecision(10) << var_12 << std::endl;
  std::cout << "Variabile var_13 - valore: " << std::setprecision(10) << var_13 << std::endl;
  std::cout << "Variabile var_14 - valore: " << std::setprecision(10) << var_14 << std::endl;
  std::cout << "Variabile var_15 - valore: " << std::setprecision(10) << var_15 << std::endl;
  std::cout << "Variabile var_16 - valore: " << std::setprecision(10) << var_16 << std::endl;
  std::cout << "Variabile var_17 - valore: " << std::setprecision(10) << var_17 << std::endl;
  std::cout << "Variabile var_18 - valore: " << std::setprecision(10) << var_18 << std::endl;
  std::cout << "Variabile var_19 - valore: " << std::setprecision(10) << var_19 << std::endl;
  std::cout << "Variabile var_20 - valore: " << std::setprecision(10) << var_20 << std::endl;
  std::cout << "Variabile var_21 - valore: " << std::setprecision(10) << var_21 << std::endl;
  std::cout << "Variabile var_22 - valore: " << std::setprecision(10) << var_22 << std::endl;
  std::cout << "Variabile var_23 - valore: " << std::setprecision(10) << var_23 << std::endl;
  std::cout << "Variabile var_24 - valore: " << std::setprecision(10) << var_24 << std::endl;
  std::cout << "Variabile var_25 - valore: " << std::setprecision(10) << var_25 << std::endl;
  std::cout << "Variabile var_26 - valore: " << std::setprecision(10) << var_26 << std::endl;
  std::cout << "Variabile var_27 - valore: " << std::setprecision(10) << var_27 << std::endl;
  std::cout << "Variabile var_28 - valore: " << std::setprecision(10) << var_28 << std::endl;
  std::cout << "Variabile var_29 - valore: " << std::setprecision(10) << var_29 << std::endl;
  std::cout << "Variabile var_30 - valore: " << std::setprecision(10) << var_30 << std::endl;

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  // std::cout << std::endl << "RESULT: " << ((res == 1) ? "CORRECT" : "INCORRECT!") << std::endl;
}
// Buffers
