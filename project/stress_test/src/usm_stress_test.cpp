#include <include.hpp>

// #define DEBUG

int main() {
  // Variables declarationlaration.0t *)malloc(sizeof(float));
  float var1 = 1.0;
  float var2 = 2.0;
  float var3 = 3.0;
  float var4 = 4.0;
  float var5 = 5.0;
  float var6 = 6.0;
  float var7 = 7.0;
  float var8 = 8.0;
  float var9 = 9.0;
  float var10 = 10.0;
  float var11 = 11.0;
  float var12 = 12.0;
  float var13 = 13.0;
  float var14 = 14.0;
  float var15 = 15.0;
  float var16 = 16.0;
  float var17 = 17.0;
  float var18 = 18.0;
  float var19 = 19.0;
  float var20 = 20.0;
  float var21 = 21.0;
  float var22 = 22.0;
  float var23 = 23.0;
  float var24 = 24.0;
  float var25 = 25.0;
  float var26 = 26.0;
  float var27 = 27.0;
  float var28 = 28.0;
  float var29 = 29.0;
  float var30 = 30.0;
  float var31 = 31.0;

  float var1_fromGpu;
  float var2_fromGpu;
  float var3_fromGpu;
  float var4_fromGpu;
  float var5_fromGpu;
  float var6_fromGpu;
  float var7_fromGpu;
  float var8_fromGpu;
  float var9_fromGpu;
  float var10_fromGpu;
  float var11_fromGpu;
  float var12_fromGpu;
  float var13_fromGpu;
  float var14_fromGpu;
  float var15_fromGpu;
  float var16_fromGpu;
  float var17_fromGpu;
  float var18_fromGpu;
  float var19_fromGpu;
  float var20_fromGpu;
  float var21_fromGpu;
  float var22_fromGpu;
  float var23_fromGpu;
  float var24_fromGpu;
  float var25_fromGpu;
  float var26_fromGpu;
  float var27_fromGpu;
  float var28_fromGpu;
  float var29_fromGpu;
  float var30_fromGpu;
  float var31_fromGpu;

  float var1_Gpu;
  float var2_Gpu;
  float var3_Gpu;
  float var4_Gpu;
  float var5_Gpu;
  float var6_Gpu;
  float var7_Gpu;
  float var8_Gpu;
  float var9_Gpu;
  float var10_Gpu;
  float var11_Gpu;
  float var12_Gpu;
  float var13_Gpu;
  float var14_Gpu;
  float var15_Gpu;
  float var16_Gpu;
  float var17_Gpu;
  float var18_Gpu;
  float var19_Gpu;
  float var20_Gpu;
  float var21_Gpu;
  float var22_Gpu;
  float var23_Gpu;
  float var24_Gpu;
  float var25_Gpu;
  float var26_Gpu;
  float var27_Gpu;
  float var28_Gpu;
  float var29_Gpu;
  float var30_Gpu;
  float var31_Gpu;

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

    queue.memcpy(&var1, &var1_Gpu, sizeof(float));
    queue.memcpy(&var2, &var2_Gpu, sizeof(float));
    queue.memcpy(&var3, &var3_Gpu, sizeof(float));
    queue.memcpy(&var4, &var4_Gpu, sizeof(float));
    queue.memcpy(&var5, &var5_Gpu, sizeof(float));
    queue.memcpy(&var6, &var6_Gpu, sizeof(float));
    queue.memcpy(&var7, &var7_Gpu, sizeof(float));
    queue.memcpy(&var8, &var8_Gpu, sizeof(float));
    queue.memcpy(&var9, &var9_Gpu, sizeof(float));
    queue.memcpy(&var10, &var10_Gpu, sizeof(float));
    queue.memcpy(&var11, &var11_Gpu, sizeof(float));
    queue.memcpy(&var12, &var12_Gpu, sizeof(float));
    queue.memcpy(&var13, &var13_Gpu, sizeof(float));
    queue.memcpy(&var14, &var14_Gpu, sizeof(float));
    queue.memcpy(&var15, &var15_Gpu, sizeof(float));
    queue.memcpy(&var16, &var16_Gpu, sizeof(float));
    queue.memcpy(&var17, &var17_Gpu, sizeof(float));
    queue.memcpy(&var18, &var18_Gpu, sizeof(float));
    queue.memcpy(&var19, &var19_Gpu, sizeof(float));
    queue.memcpy(&var20, &var20_Gpu, sizeof(float));
    queue.memcpy(&var21, &var21_Gpu, sizeof(float));
    queue.memcpy(&var22, &var22_Gpu, sizeof(float));
    queue.memcpy(&var23, &var23_Gpu, sizeof(float));
    queue.memcpy(&var24, &var24_Gpu, sizeof(float));
    queue.memcpy(&var25, &var25_Gpu, sizeof(float));
    queue.memcpy(&var26, &var26_Gpu, sizeof(float));
    queue.memcpy(&var27, &var27_Gpu, sizeof(float));
    queue.memcpy(&var28, &var28_Gpu, sizeof(float));
    queue.memcpy(&var29, &var29_Gpu, sizeof(float));
    queue.memcpy(&var30, &var30_Gpu, sizeof(float));
    queue.memcpy(&var31, &var31_Gpu, sizeof(float));
    queue.wait();
    // Queue
    queue
        .submit([&](sycl::handler &cgh) {
          // Start timer
          start = std::chrono::steady_clock::now();

          var1_Gpu += var31_Gpu * var1_Gpu;
          var2_Gpu += var30_Gpu * var2_Gpu;
          var3_Gpu += var29_Gpu * var3_Gpu;
          var4_Gpu += var28_Gpu * var4_Gpu;
          var5_Gpu += var27_Gpu * var5_Gpu;
          var6_Gpu += var26_Gpu * var6_Gpu;
          var7_Gpu += var25_Gpu * var7_Gpu;
          var8_Gpu += var24_Gpu * var8_Gpu;
          var9_Gpu += var23_Gpu * var9_Gpu;
          var10_Gpu += var22_Gpu * var10_Gpu;
          var11_Gpu += var21_Gpu * var11_Gpu;
          var12_Gpu += var20_Gpu * var12_Gpu;
          var13_Gpu += var19_Gpu * var13_Gpu;
          var14_Gpu += var18_Gpu * var14_Gpu;
          var15_Gpu += var17_Gpu * var15_Gpu;
          var16_Gpu += var16_Gpu * var16_Gpu;
          var17_Gpu += var15_Gpu * var17_Gpu;
          var18_Gpu += var14_Gpu * var18_Gpu;
          var19_Gpu += var13_Gpu * var19_Gpu;
          var20_Gpu += var12_Gpu * var20_Gpu;
          var21_Gpu += var11_Gpu * var21_Gpu;
          var22_Gpu += var10_Gpu * var22_Gpu;
          var23_Gpu += var9_Gpu * var23_Gpu;
          var24_Gpu += var8_Gpu * var24_Gpu;
          var25_Gpu += var7_Gpu * var25_Gpu;
          var26_Gpu += var6_Gpu * var26_Gpu;
          var27_Gpu += var5_Gpu * var27_Gpu;
          var28_Gpu += var4_Gpu * var28_Gpu;
          var29_Gpu += var3_Gpu * var29_Gpu;
          var30_Gpu += var2_Gpu * var30_Gpu;
          var31_Gpu += var1_Gpu * var31_Gpu;
        })
        .wait();

    queue.memcpy(&var1_fromGpu, &var1_Gpu, sizeof(float)).wait();
    queue.memcpy(&var2_fromGpu, &var2_Gpu, sizeof(float)).wait();
    queue.memcpy(&var3_fromGpu, &var3_Gpu, sizeof(float)).wait();
    queue.memcpy(&var4_fromGpu, &var4_Gpu, sizeof(float)).wait();
    queue.memcpy(&var5_fromGpu, &var5_Gpu, sizeof(float)).wait();
    queue.memcpy(&var6_fromGpu, &var6_Gpu, sizeof(float)).wait();
    queue.memcpy(&var7_fromGpu, &var7_Gpu, sizeof(float)).wait();
    queue.memcpy(&var8_fromGpu, &var8_Gpu, sizeof(float)).wait();
    queue.memcpy(&var9_fromGpu, &var9_Gpu, sizeof(float)).wait();
    queue.memcpy(&var10_fromGpu, &var10_Gpu, sizeof(float)).wait();
    queue.memcpy(&var11_fromGpu, &var11_Gpu, sizeof(float)).wait();
    queue.memcpy(&var12_fromGpu, &var12_Gpu, sizeof(float)).wait();
    queue.memcpy(&var13_fromGpu, &var13_Gpu, sizeof(float)).wait();
    queue.memcpy(&var14_fromGpu, &var14_Gpu, sizeof(float)).wait();
    queue.memcpy(&var15_fromGpu, &var15_Gpu, sizeof(float)).wait();
    queue.memcpy(&var16_fromGpu, &var16_Gpu, sizeof(float)).wait();
    queue.memcpy(&var17_fromGpu, &var17_Gpu, sizeof(float)).wait();
    queue.memcpy(&var18_fromGpu, &var18_Gpu, sizeof(float)).wait();
    queue.memcpy(&var19_fromGpu, &var19_Gpu, sizeof(float)).wait();
    queue.memcpy(&var20_fromGpu, &var20_Gpu, sizeof(float)).wait();
    queue.memcpy(&var21_fromGpu, &var21_Gpu, sizeof(float)).wait();
    queue.memcpy(&var22_fromGpu, &var22_Gpu, sizeof(float)).wait();
    queue.memcpy(&var23_fromGpu, &var23_Gpu, sizeof(float)).wait();
    queue.memcpy(&var24_fromGpu, &var24_Gpu, sizeof(float)).wait();
    queue.memcpy(&var25_fromGpu, &var25_Gpu, sizeof(float)).wait();
    queue.memcpy(&var26_fromGpu, &var26_Gpu, sizeof(float)).wait();
    queue.memcpy(&var27_fromGpu, &var27_Gpu, sizeof(float)).wait();
    queue.memcpy(&var28_fromGpu, &var28_Gpu, sizeof(float)).wait();
    queue.memcpy(&var29_fromGpu, &var29_Gpu, sizeof(float)).wait();
    queue.memcpy(&var30_fromGpu, &var30_Gpu, sizeof(float)).wait();
    queue.memcpy(&var31_fromGpu, &var31_Gpu, sizeof(float)).wait();

  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  // std::cout << std::endl << "RESULT: " << ((res == 1) ? "CORRECT" : "INCORRECT!") << std::endl;
}