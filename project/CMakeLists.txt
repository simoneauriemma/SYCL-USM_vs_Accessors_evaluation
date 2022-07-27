project(project_example)
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Include macros
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
include(exec_helper)

option(SYCL_BACKEND "Enable selected SYCL backend" "")
option(SYCL_DEVICE "Select target SYCL device" GPU)

if(SYCL_BACKEND STREQUAL "hipSYCL")
	find_package(hipSYCL REQUIRED)
endif()

# Handle multiple devices
message(STATUS "SYCL_BACKEND: ${SYCL_BACKEND}")
message(STATUS "SYCL_DEVICE: ${SYCL_DEVICE}")

# Handle multiple devices
set(CPU_DEVICE 0)
set(GPU_DEVICE 1)
set(HOST_DEVICE 2)

string(TOUPPER ${SYCL_DEVICE} SYCL_DEVICE)
set(device_value -1)
if(SYCL_DEVICE STREQUAL CPU)
        set(device_value ${CPU_DEVICE})
elseif(SYCL_DEVICE STREQUAL GPU)
        set(device_value ${GPU_DEVICE})
elseif(SYCL_DEVICE STREQUAL HOST)
        set(device_value ${HOST_DEVICE})
else()
        message(FATAL_ERROR "Unknown device selected: ${SYCL_DEVICE}" )
endif()

add_subdirectory(src/matrix_multiplication)
add_subdirectory(src/matrix_addition)
add_subdirectory(src/stress_test)
