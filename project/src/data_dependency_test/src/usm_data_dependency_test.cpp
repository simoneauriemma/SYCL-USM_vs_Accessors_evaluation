#include <include.hpp>

// #define DEBUG

int main() {

  // Variables declaration
  float var0 = 1.0;
  float var1 = 2.0;
  float var2 = 3.0;
  float var3 = 4.0;
  float var4 = 5.0;
  float var5 = 6.0;
  float var6 = 7.0;
  float var7 = 8.0;
  float var8 = 9.0;
  float var9 = 10.0;
  float var10 = 11.0;
  float var11 = 12.0;
  float var12 = 13.0;
  float var13 = 14.0;
  float var14 = 15.0;
  float var15 = 16.0;
  float var16 = 17.0;
  float var17 = 18.0;
  float var18 = 19.0;
  float var19 = 20.0;
  float var20 = 21.0;
  float var21 = 22.0;
  float var22 = 23.0;
  float var23 = 24.0;
  float var24 = 25.0;
  float var25 = 26.0;
  float var26 = 27.0;
  float var27 = 28.0;
  float var28 = 29.0;
  float var29 = 30.0;
  float var30 = 31.0;

  // Variables that will be returned from the GPU
  float var0_fromGpu;
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

    // Allocate space for variables on the GPU
    float* var0_Gpu = sycl::malloc_device<float>(1, queue);
    float* var1_Gpu = sycl::malloc_device<float>(1, queue);
    float* var2_Gpu = sycl::malloc_device<float>(1, queue);
    float* var3_Gpu = sycl::malloc_device<float>(1, queue);
    float* var4_Gpu = sycl::malloc_device<float>(1, queue);
    float* var5_Gpu = sycl::malloc_device<float>(1, queue);
    float* var6_Gpu = sycl::malloc_device<float>(1, queue);
    float* var7_Gpu = sycl::malloc_device<float>(1, queue);
    float* var8_Gpu = sycl::malloc_device<float>(1, queue);
    float* var9_Gpu = sycl::malloc_device<float>(1, queue);
    float* var10_Gpu = sycl::malloc_device<float>(1, queue);
    float* var11_Gpu = sycl::malloc_device<float>(1, queue);
    float* var12_Gpu = sycl::malloc_device<float>(1, queue);
    float* var13_Gpu = sycl::malloc_device<float>(1, queue);
    float* var14_Gpu = sycl::malloc_device<float>(1, queue);
    float* var15_Gpu = sycl::malloc_device<float>(1, queue);
    float* var16_Gpu = sycl::malloc_device<float>(1, queue);
    float* var17_Gpu = sycl::malloc_device<float>(1, queue);
    float* var18_Gpu = sycl::malloc_device<float>(1, queue);
    float* var19_Gpu = sycl::malloc_device<float>(1, queue);
    float* var20_Gpu = sycl::malloc_device<float>(1, queue);
    float* var21_Gpu = sycl::malloc_device<float>(1, queue);
    float* var22_Gpu = sycl::malloc_device<float>(1, queue);
    float* var23_Gpu = sycl::malloc_device<float>(1, queue);
    float* var24_Gpu = sycl::malloc_device<float>(1, queue);
    float* var25_Gpu = sycl::malloc_device<float>(1, queue);
    float* var26_Gpu = sycl::malloc_device<float>(1, queue);
    float* var27_Gpu = sycl::malloc_device<float>(1, queue);
    float* var28_Gpu = sycl::malloc_device<float>(1, queue);
    float* var29_Gpu = sycl::malloc_device<float>(1, queue);
    float* var30_Gpu = sycl::malloc_device<float>(1, queue);

    // Copy the data from the host to the GPU
    queue.memcpy(var0_Gpu, &var0, sizeof(float));
    queue.memcpy(var1_Gpu, &var1, sizeof(float));
    queue.memcpy(var2_Gpu, &var2, sizeof(float));
    queue.memcpy(var3_Gpu, &var3, sizeof(float));
    queue.memcpy(var4_Gpu, &var4, sizeof(float));
    queue.memcpy(var5_Gpu, &var5, sizeof(float));
    queue.memcpy(var6_Gpu, &var6, sizeof(float));
    queue.memcpy(var7_Gpu, &var7, sizeof(float));
    queue.memcpy(var8_Gpu, &var8, sizeof(float));
    queue.memcpy(var9_Gpu, &var9, sizeof(float));
    queue.memcpy(var10_Gpu, &var10, sizeof(float));
    queue.memcpy(var11_Gpu, &var11, sizeof(float));
    queue.memcpy(var12_Gpu, &var12, sizeof(float));
    queue.memcpy(var13_Gpu, &var13, sizeof(float));
    queue.memcpy(var14_Gpu, &var14, sizeof(float));
    queue.memcpy(var15_Gpu, &var15, sizeof(float));
    queue.memcpy(var16_Gpu, &var16, sizeof(float));
    queue.memcpy(var17_Gpu, &var17, sizeof(float));
    queue.memcpy(var18_Gpu, &var18, sizeof(float));
    queue.memcpy(var19_Gpu, &var19, sizeof(float));
    queue.memcpy(var20_Gpu, &var20, sizeof(float));
    queue.memcpy(var21_Gpu, &var21, sizeof(float));
    queue.memcpy(var22_Gpu, &var22, sizeof(float));
    queue.memcpy(var23_Gpu, &var23, sizeof(float));
    queue.memcpy(var24_Gpu, &var24, sizeof(float));
    queue.memcpy(var25_Gpu, &var25, sizeof(float));
    queue.memcpy(var26_Gpu, &var26, sizeof(float));
    queue.memcpy(var27_Gpu, &var27, sizeof(float));
    queue.memcpy(var28_Gpu, &var28, sizeof(float));
    queue.memcpy(var29_Gpu, &var29, sizeof(float));
    queue.memcpy(var30_Gpu, &var30, sizeof(float));
    queue.wait();

    // Queue
    queue
        .submit([&](sycl::handler& cgh) {
          // Start timer
          start = std::chrono::steady_clock::now();

          // Execute kernel
          cgh.parallel_for(sycl::range<1>(31), [=](sycl::id<1> idx) {
            *var0_Gpu += *var30_Gpu * *var0_Gpu;
            *var1_Gpu += *var29_Gpu * *var1_Gpu;
            *var2_Gpu += *var28_Gpu * *var2_Gpu;
            *var3_Gpu += *var27_Gpu * *var3_Gpu;
            *var4_Gpu += *var26_Gpu * *var4_Gpu;
            *var5_Gpu += *var25_Gpu * *var5_Gpu;
            *var6_Gpu += *var24_Gpu * *var6_Gpu;
            *var7_Gpu += *var23_Gpu * *var7_Gpu;
            *var8_Gpu += *var22_Gpu * *var8_Gpu;
            *var9_Gpu += *var21_Gpu * *var9_Gpu;
            *var10_Gpu += *var20_Gpu * *var10_Gpu;
            *var11_Gpu += *var19_Gpu * *var11_Gpu;
            *var12_Gpu += *var18_Gpu * *var12_Gpu;
            *var13_Gpu += *var17_Gpu * *var13_Gpu;
            *var14_Gpu += *var16_Gpu * *var14_Gpu;
            *var15_Gpu += *var15_Gpu * *var15_Gpu;
            *var16_Gpu += *var14_Gpu * *var16_Gpu;
            *var17_Gpu += *var13_Gpu * *var17_Gpu;
            *var18_Gpu += *var12_Gpu * *var18_Gpu;
            *var19_Gpu += *var11_Gpu * *var19_Gpu;
            *var20_Gpu += *var10_Gpu * *var20_Gpu;
            *var21_Gpu += *var9_Gpu * *var21_Gpu;
            *var22_Gpu += *var8_Gpu * *var22_Gpu;
            *var23_Gpu += *var7_Gpu * *var23_Gpu;
            *var24_Gpu += *var6_Gpu * *var24_Gpu;
            *var25_Gpu += *var5_Gpu * *var25_Gpu;
            *var26_Gpu += *var4_Gpu * *var26_Gpu;
            *var27_Gpu += *var3_Gpu * *var27_Gpu;
            *var28_Gpu += *var2_Gpu * *var28_Gpu;
            *var29_Gpu += *var1_Gpu * *var29_Gpu;
            *var30_Gpu += *var0_Gpu * *var30_Gpu;
          });
        })
        .wait();

    // Return the data to the host
    queue.memcpy(&var0_fromGpu, var0_Gpu, sizeof(float));
    queue.memcpy(&var1_fromGpu, var1_Gpu, sizeof(float));
    queue.memcpy(&var2_fromGpu, var2_Gpu, sizeof(float));
    queue.memcpy(&var3_fromGpu, var3_Gpu, sizeof(float));
    queue.memcpy(&var4_fromGpu, var4_Gpu, sizeof(float));
    queue.memcpy(&var5_fromGpu, var5_Gpu, sizeof(float));
    queue.memcpy(&var6_fromGpu, var6_Gpu, sizeof(float));
    queue.memcpy(&var7_fromGpu, var7_Gpu, sizeof(float));
    queue.memcpy(&var8_fromGpu, var8_Gpu, sizeof(float));
    queue.memcpy(&var9_fromGpu, var9_Gpu, sizeof(float));
    queue.memcpy(&var10_fromGpu, var10_Gpu, sizeof(float));
    queue.memcpy(&var11_fromGpu, var11_Gpu, sizeof(float));
    queue.memcpy(&var12_fromGpu, var12_Gpu, sizeof(float));
    queue.memcpy(&var13_fromGpu, var13_Gpu, sizeof(float));
    queue.memcpy(&var14_fromGpu, var14_Gpu, sizeof(float));
    queue.memcpy(&var15_fromGpu, var15_Gpu, sizeof(float));
    queue.memcpy(&var16_fromGpu, var16_Gpu, sizeof(float));
    queue.memcpy(&var17_fromGpu, var17_Gpu, sizeof(float));
    queue.memcpy(&var18_fromGpu, var18_Gpu, sizeof(float));
    queue.memcpy(&var19_fromGpu, var19_Gpu, sizeof(float));
    queue.memcpy(&var20_fromGpu, var20_Gpu, sizeof(float));
    queue.memcpy(&var21_fromGpu, var21_Gpu, sizeof(float));
    queue.memcpy(&var22_fromGpu, var22_Gpu, sizeof(float));
    queue.memcpy(&var23_fromGpu, var23_Gpu, sizeof(float));
    queue.memcpy(&var24_fromGpu, var24_Gpu, sizeof(float));
    queue.memcpy(&var25_fromGpu, var25_Gpu, sizeof(float));
    queue.memcpy(&var26_fromGpu, var26_Gpu, sizeof(float));
    queue.memcpy(&var27_fromGpu, var27_Gpu, sizeof(float));
    queue.memcpy(&var28_fromGpu, var28_Gpu, sizeof(float));
    queue.memcpy(&var29_fromGpu, var29_Gpu, sizeof(float));
    queue.memcpy(&var30_fromGpu, var30_Gpu, sizeof(float));
    queue.wait();

    // Deallocate GPU variables
    sycl::free(var0_Gpu, queue);
    sycl::free(var1_Gpu, queue);
    sycl::free(var2_Gpu, queue);
    sycl::free(var3_Gpu, queue);
    sycl::free(var4_Gpu, queue);
    sycl::free(var5_Gpu, queue);
    sycl::free(var6_Gpu, queue);
    sycl::free(var7_Gpu, queue);
    sycl::free(var8_Gpu, queue);
    sycl::free(var9_Gpu, queue);
    sycl::free(var10_Gpu, queue);
    sycl::free(var11_Gpu, queue);
    sycl::free(var12_Gpu, queue);
    sycl::free(var13_Gpu, queue);
    sycl::free(var14_Gpu, queue);
    sycl::free(var15_Gpu, queue);
    sycl::free(var16_Gpu, queue);
    sycl::free(var17_Gpu, queue);
    sycl::free(var18_Gpu, queue);
    sycl::free(var19_Gpu, queue);
    sycl::free(var20_Gpu, queue);
    sycl::free(var21_Gpu, queue);
    sycl::free(var22_Gpu, queue);
    sycl::free(var23_Gpu, queue);
    sycl::free(var24_Gpu, queue);
    sycl::free(var25_Gpu, queue);
    sycl::free(var26_Gpu, queue);
    sycl::free(var27_Gpu, queue);
    sycl::free(var28_Gpu, queue);
    sycl::free(var29_Gpu, queue);
    sycl::free(var30_Gpu, queue);

  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  // Stampa finale
  std::cout << "Variabile var0_fromGpu - valore: " << std::setprecision(10) << var0_fromGpu << std::endl;
  std::cout << "Variabile var1_fromGpu - valore: " << std::setprecision(10) << var1_fromGpu << std::endl;
  std::cout << "Variabile var2_fromGpu - valore: " << std::setprecision(10) << var2_fromGpu << std::endl;
  std::cout << "Variabile var3_fromGpu - valore: " << std::setprecision(10) << var3_fromGpu << std::endl;
  std::cout << "Variabile var4_fromGpu - valore: " << std::setprecision(10) << var4_fromGpu << std::endl;
  std::cout << "Variabile var5_fromGpu - valore: " << std::setprecision(10) << var5_fromGpu << std::endl;
  std::cout << "Variabile var6_fromGpu - valore: " << std::setprecision(10) << var6_fromGpu << std::endl;
  std::cout << "Variabile var7_fromGpu - valore: " << std::setprecision(10) << var7_fromGpu << std::endl;
  std::cout << "Variabile var8_fromGpu - valore: " << std::setprecision(10) << var8_fromGpu << std::endl;
  std::cout << "Variabile var9_fromGpu - valore: " << std::setprecision(10) << var9_fromGpu << std::endl;
  std::cout << "Variabile var10_fromGpu - valore: " << std::setprecision(10) << var10_fromGpu << std::endl;
  std::cout << "Variabile var11_fromGpu - valore: " << std::setprecision(10) << var11_fromGpu << std::endl;
  std::cout << "Variabile var12_fromGpu - valore: " << std::setprecision(10) << var12_fromGpu << std::endl;
  std::cout << "Variabile var13_fromGpu - valore: " << std::setprecision(10) << var13_fromGpu << std::endl;
  std::cout << "Variabile var14_fromGpu - valore: " << std::setprecision(10) << var14_fromGpu << std::endl;
  std::cout << "Variabile var15_fromGpu - valore: " << std::setprecision(10) << var15_fromGpu << std::endl;
  std::cout << "Variabile var16_fromGpu - valore: " << std::setprecision(10) << var16_fromGpu << std::endl;
  std::cout << "Variabile var17_fromGpu - valore: " << std::setprecision(10) << var17_fromGpu << std::endl;
  std::cout << "Variabile var18_fromGpu - valore: " << std::setprecision(10) << var18_fromGpu << std::endl;
  std::cout << "Variabile var19_fromGpu - valore: " << std::setprecision(10) << var19_fromGpu << std::endl;
  std::cout << "Variabile var20_fromGpu - valore: " << std::setprecision(10) << var20_fromGpu << std::endl;
  std::cout << "Variabile var21_fromGpu - valore: " << std::setprecision(10) << var21_fromGpu << std::endl;
  std::cout << "Variabile var22_fromGpu - valore: " << std::setprecision(10) << var22_fromGpu << std::endl;
  std::cout << "Variabile var23_fromGpu - valore: " << std::setprecision(10) << var23_fromGpu << std::endl;
  std::cout << "Variabile var24_fromGpu - valore: " << std::setprecision(10) << var24_fromGpu << std::endl;
  std::cout << "Variabile var25_fromGpu - valore: " << std::setprecision(10) << var25_fromGpu << std::endl;
  std::cout << "Variabile var26_fromGpu - valore: " << std::setprecision(10) << var26_fromGpu << std::endl;
  std::cout << "Variabile var27_fromGpu - valore: " << std::setprecision(10) << var27_fromGpu << std::endl;
  std::cout << "Variabile var28_fromGpu - valore: " << std::setprecision(10) << var28_fromGpu << std::endl;
  std::cout << "Variabile var29_fromGpu - valore: " << std::setprecision(10) << var29_fromGpu << std::endl;
  std::cout << "Variabile var30_fromGpu - valore: " << std::setprecision(10) << var30_fromGpu << std::endl;

  // Get execution time
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "- Parallel time in seconds: " << elapsed_seconds.count() << std::endl;

  // std::cout << std::endl << "RESULT: " << ((res == 1) ? "CORRECT" : "INCORRECT!") << std::endl;
}