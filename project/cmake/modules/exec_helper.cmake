macro(add_sycl_executable target source_file)
	add_executable(${target})
	
	target_include_directories(${target}
       	PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
	)

target_sources(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${source_file}
)
set_target_properties(${target} PROPERTIES
        PUBLIC_HEADER
        ${CMAKE_CURRENT_SOURCE_DIR}/include/include.hpp
)

add_compile_definitions(${target}
        CPU_DEVICE=${CPU_DEVICE}
        GPU_DEVICE=${GPU_DEVICE}
        HOST_DEVICE=${HOST_DEVICE}
        DEVICE_VALUE=${device_value}
)

if(SYCL_BACKEND STREQUAL hipSYCL)
        add_sycl_to_target(TARGET ${target})
endif()

endmacro()
