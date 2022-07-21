#include <include.hpp>

// #define DEBUG

int main() {
  // Variables declaration
  double var_1 = 0.0;
  double var_2 = 0.0;
  double var_3 = 0.0;
  double var_4 = 0.0;
  double var_5 = 0.0;
  double var_6 = 0.0;
  double var_7 = 0.0;
  double var_8 = 0.0;
  double var_9 = 0.0;
  double var_10 = 0.0;
  double var_11 = 0.0;
  double var_12 = 0.0;
  double var_13 = 0.0;
  double var_14 = 0.0;
  double var_15 = 0.0;
  double var_16 = 0.0;
  double var_17 = 0.0;
  double var_18 = 0.0;
  double var_19 = 0.0;
  double var_20 = 0.0;
  double var_21 = 0.0;
  double var_22 = 0.0;
  double var_23 = 0.0;
  double var_24 = 0.0;
  double var_25 = 0.0;
  double var_26 = 0.0;
  double var_27 = 0.0;
  double var_28 = 0.0;
  double var_29 = 0.0;
  double var_30 = 0.0;
  double var_31 = 0.0;

  // Init

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
    sycl::buffer<double, 1> buffer_1(&var_1, sycl::range{1});
    sycl::buffer<double, 1> buffer_2(&var_2, sycl::range{1});
    sycl::buffer<double, 1> buffer_3(&var_3, sycl::range{1});
    sycl::buffer<double, 1> buffer_4(&var_4, sycl::range{1});
    sycl::buffer<double, 1> buffer_5(&var_5, sycl::range{1});
    sycl::buffer<double, 1> buffer_6(&var_6, sycl::range{1});
    sycl::buffer<double, 1> buffer_7(&var_7, sycl::range{1});
    sycl::buffer<double, 1> buffer_8(&var_8, sycl::range{1});
    sycl::buffer<double, 1> buffer_9(&var_9, sycl::range{1});
    sycl::buffer<double, 1> buffer_10(&var_10, sycl::range{1});
    sycl::buffer<double, 1> buffer_11(&var_11, sycl::range{1});
    sycl::buffer<double, 1> buffer_12(&var_12, sycl::range{1});
    sycl::buffer<double, 1> buffer_13(&var_13, sycl::range{1});
    sycl::buffer<double, 1> buffer_14(&var_14, sycl::range{1});
    sycl::buffer<double, 1> buffer_15(&var_15, sycl::range{1});
    sycl::buffer<double, 1> buffer_16(&var_16, sycl::range{1});
    sycl::buffer<double, 1> buffer_17(&var_17, sycl::range{1});
    sycl::buffer<double, 1> buffer_18(&var_18, sycl::range{1});
    sycl::buffer<double, 1> buffer_19(&var_19, sycl::range{1});
    sycl::buffer<double, 1> buffer_20(&var_20, sycl::range{1});
    sycl::buffer<double, 1> buffer_21(&var_21, sycl::range{1});
    sycl::buffer<double, 1> buffer_22(&var_22, sycl::range{1});
    sycl::buffer<double, 1> buffer_23(&var_23, sycl::range{1});
    sycl::buffer<double, 1> buffer_24(&var_24, sycl::range{1});
    sycl::buffer<double, 1> buffer_25(&var_25, sycl::range{1});
    sycl::buffer<double, 1> buffer_26(&var_26, sycl::range{1});
    sycl::buffer<double, 1> buffer_27(&var_27, sycl::range{1});
    sycl::buffer<double, 1> buffer_28(&var_28, sycl::range{1});
    sycl::buffer<double, 1> buffer_29(&var_29, sycl::range{1});
    sycl::buffer<double, 1> buffer_30(&var_30, sycl::range{1});
    sycl::buffer<double, 1> buffer_31(&var_31, sycl::range{1});

    // Queue
    queue
        .submit([&](sycl::handler &cgh) {
          sycl::stream out(1024, 256, cgh);

          // Accessors
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
          sycl::accessor accessor_31{buffer_31, cgh, sycl::read_write};

          // Start timer
          start = std::chrono::steady_clock::now();

          cgh.parallel_for(sycl::range<1>(31), [=](sycl::id<1> idx) {
            accessor_1[0] = 10;
            accessor_2[0] = 10;
            accessor_3[0] = 10;
            accessor_4[0] = 10;
            accessor_5[0] = 10;
            accessor_6[0] = 10;
            accessor_7[0] = 10;
            accessor_8[0] = 10;
            accessor_9[0] = 10;
            accessor_10[0] = 10;
            accessor_11[0] = 10;
            accessor_12[0] = 10;
            accessor_13[0] = 10;
            accessor_14[0] = 10;
            accessor_15[0] = 10;
            accessor_16[0] = 10;
            accessor_17[0] = 10;
            accessor_18[0] = 10;
            accessor_19[0] = 10;
            accessor_20[0] = 10;
            accessor_21[0] = 10;
            accessor_22[0] = 10;
            accessor_23[0] = 10;
            accessor_24[0] = 10;
            accessor_25[0] = 10;
            accessor_26[0] = 10;
            accessor_27[0] = 10;
            accessor_28[0] = 10;
            accessor_29[0] = 10;
            accessor_30[0] = 10;
            accessor_31[0] = 10;

            accessor_1[0] += accessor_31[0] * accessor_1[0];
            accessor_2[0] += accessor_30[0] * accessor_2[0];
            accessor_3[0] += accessor_29[0] * accessor_3[0];
            accessor_4[0] += accessor_28[0] * accessor_4[0];
            accessor_5[0] += accessor_27[0] * accessor_5[0];
            accessor_6[0] += accessor_26[0] * accessor_6[0];
            accessor_7[0] += accessor_25[0] * accessor_7[0];
            accessor_8[0] += accessor_24[0] * accessor_8[0];
            accessor_9[0] += accessor_23[0] * accessor_9[0];
            accessor_10[0] += accessor_22[0] * accessor_10[0];
            accessor_11[0] += accessor_21[0] * accessor_11[0];
            accessor_12[0] += accessor_20[0] * accessor_12[0];
            accessor_13[0] += accessor_19[0] * accessor_13[0];
            accessor_14[0] += accessor_18[0] * accessor_14[0];
            accessor_15[0] += accessor_17[0] * accessor_15[0];
            accessor_16[0] += accessor_16[0] * accessor_16[0];
            accessor_17[0] += accessor_15[0] * accessor_17[0];
            accessor_18[0] += accessor_14[0] * accessor_18[0];
            accessor_19[0] += accessor_13[0] * accessor_19[0];
            accessor_20[0] += accessor_12[0] * accessor_20[0];
            accessor_21[0] += accessor_11[0] * accessor_21[0];
            accessor_22[0] += accessor_10[0] * accessor_22[0];
            accessor_23[0] += accessor_9[0] * accessor_23[0];
            accessor_24[0] += accessor_8[0] * accessor_24[0];
            accessor_25[0] += accessor_7[0] * accessor_25[0];
            accessor_26[0] += accessor_6[0] * accessor_26[0];
            accessor_27[0] += accessor_5[0] * accessor_27[0];
            accessor_28[0] += accessor_4[0] * accessor_28[0];
            accessor_29[0] += accessor_3[0] * accessor_29[0];
            accessor_30[0] += accessor_2[0] * accessor_30[0];
            accessor_31[0] += accessor_1[0] * accessor_31[0];

            // out << "Hello stream! " << idx << sycl::flush;
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

  // std::cout << std::endl << "RESULT: " << ((res == 1) ? "CORRECT" : "INCORRECT!") << std::endl;
}
// Buffers
