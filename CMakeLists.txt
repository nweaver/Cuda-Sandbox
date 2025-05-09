cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)

project(CudaTest CUDA CXX)


set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

set(DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

add_executable(CudaTest main.cu)

set_target_properties(CudaTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set (CMAKE_CXX_STANDARD 20)
# This is "Turn on ALL the hardening stuff.  There is a 
# minor performance impact but hey, better Safe than Sorry"
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g \
-O2 \
-Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
-Werror=format-security \
-D_GLIBCXX_ASSERTIONS \
-fopenmp ")

set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra --coverage -D_GLIBCXX_ASSERTIONS")



# This is geting gunit so you don't have to...
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
	
enable_testing()


add_executable(testbinary basetest.cpp vector_add.cu parallel_utils.cpp matrix.cpp matrix_mul.cu) 
target_link_libraries(
  testbinary
  GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(testbinary)