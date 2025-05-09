#include <gtest/gtest.h>
#include <string>

#include "vector_add.h"
#include "parallel_utils.hpp"
#include "matrix.hpp"
#include "matrix_mul.h"
#include <ranges>

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
  auto sequential = GetTiming([&]()
                              { seqd = a * b; });
  parallelNuke();
  auto parallel = GetTiming([&]()
                            { paralleld = parallel_multiply4(a, b); });
  std::cout << "Speedup = " << (((float)sequential) / ((float)parallel)) << "\n";
  EXPECT_TRUE(seqd == paralleld);
}

TEST(Simple, Cuda)
{
  Matrix<float> a(2);
  Matrix<float> b(2);
  a(0, 0) = 4.0;
  a(0, 1) = 2.0;
  a(1, 0) = 0.0;
  a(1, 1) = 0.0;

  b(0, 0) = 1.0;
  b(0, 1) = 3.0;
  b(1, 0) = 0.0;
  b(1, 1) = 0.0;

  std::cout << a(0, 0) << " " << a(0, 1) << "\n";
  std::cout << a(1, 0) << " " << a(1, 1) << "\n\n";

  std::cout << b(0, 0) << " " << b(0, 1) << "\n";
  std::cout << b(1, 0) << " " << b(1, 1) << "\n\n";

  auto d = a * b;
  std::cout << d(0, 0) << " " << d(0, 1) << "\n";
  std::cout << d(1, 0) << " " << d(1, 1) << "\n\n";
  auto f = cudamul(a, b);
  std::cout << f(0, 0) << " " << f(0, 1) << "\n";
  std::cout << f(1, 0) << " " << f32addf128(1, 1) << "\n";
}

TEST(Performance, Cuda)
{
  size_t size = 4096;
  Matrix<float> a(size, true);
  Matrix<float> b(size, true);
  Matrix<float> paralleld;
  Matrix<float> cudad;
  parallelNuke();
  auto parallel = GetTiming([&]()
                            { paralleld = parallel_multiply4(a, b); });
  auto cuda = GetTiming([&]()
                        { cudad = cudamul(a, b); });
  std::cout << "Speedup = " << (((float)parallel) / ((float)cuda)) << "\n";

  EXPECT_TRUE(cudad == paralleld);
}

TEST(Cuda, CudaTranspose)
{
  auto tmp = 2;
  for (auto i = 0; i < 10; ++i)
  {
    Matrix<float> in(tmp, true);
    auto ref = in.transpose();
    auto cuda = cudaTranspose(in);
    std::cout << tmp << "\n";
    EXPECT_TRUE(ref == cuda);
    tmp = tmp * 2;
  }
}