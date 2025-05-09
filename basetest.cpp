#include <gtest/gtest.h>
#include <string>

#include "vector_add.h"
#include "parallel_utils.hpp"
#include "matrix.hpp"
#include "matrix_mul.h"

// Demonstrate some basic assertions.
TEST(SimpleTest, CanCallCudaCode)
{
  EXPECT_TRUE(true);
  EXPECT_TRUE(cuda_call_test());
  test_vector_add();
}

TEST(Performance, OpenMP) 
{
  size_t size = 2048;
  size = 32;
        Matrix<float> a(size, true);
        Matrix<float> b(size, true);
        Matrix<float> seqd;
        Matrix<float> paralleld;
        parallelNuke();
        auto sequential = GetTiming([&]() {
            seqd = a * b;
        });
        parallelNuke();
        auto parallel = GetTiming([&]() {
            paralleld = parallel_multiply4(a, b);
        });
        std::cout << "Speedup = " << (((float) sequential) / ((float) parallel)) << "\n";
        EXPECT_TRUE(seqd == paralleld);
}

TEST(Cuda, CudaTranspose){
  Matrix<float> in(4096, true);
  auto ref = in.transpose();
  auto cuda = cudaTranspose(in);
  EXPECT_TRUE(ref == cuda);
}