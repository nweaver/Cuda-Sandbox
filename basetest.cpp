#include <gtest/gtest.h>
#include <string>

#include "vector_add.h"
#include "parallel_utils.hpp"
#include "matrix.hpp"
#include "matrix_mul.h"
#include <ranges>
#include <omp.h>


// Demonstrate some basic assertions.
TEST(SimpleTest, CanCallCudaCode)
{
  EXPECT_TRUE(true);
  EXPECT_TRUE(cuda_call_test());
  test_vector_add();
  CudaDeviceInfo();
}

TEST(Performance, OpenMP)
{
  size_t size = 64;
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

TEST(Performance, Cuda)
{
  size_t size = 4096;
  Matrix<float> a(size, true);
  Matrix<float> b(size, true);
  Matrix<float> paralleld;
  Matrix<float> cudad;
  Matrix<float> cudasmallblock;
  parallelNuke();
  auto parallel = GetTiming([&]()
                            { paralleld = parallel_multiply4(a, b); });
  auto cuda = GetTiming([&]()
                        { cudad = cudamul(a, b); });
  auto cudasmall = GetTiming([&]()
                             { cudad = cudamul_smallblock(a, b); });
  auto cudacoalesce = GetTiming([&]()
                                { cudad = cudamul_coalesce(a, b); });
  auto cudacache = GetTiming([&]()
                             { cudad = cudamul_cache(a, b); });

  std::cout << "Speedup naive      = " << (((float)parallel) / ((float)cuda)) << "\n";
  std::cout << "Speedup smallblock = " << (((float)parallel) / ((float)cudasmall)) << "\n";
  std::cout << "Speedup coalesce   = " << (((float)parallel) / ((float)cudacoalesce)) << "\n";
  std::cout << "Speedup cache      = " << (((float)parallel) / ((float)cudacache)) << "\n";

  EXPECT_TRUE(cudad == paralleld);
}

// Note, this IS failing on small matrixes...
TEST(Cuda, CudaTranspose)
{
  auto tmp = 256;
  for (auto i = 0; i < 5; ++i)
  {
    Matrix<float> in(tmp, true);
    auto ref = in.transpose();
    auto cuda = cudaTranspose(in);
    EXPECT_TRUE(ref == cuda);
    tmp = tmp * 2;
  }
}

TEST(Performance, Transpose)
{
  size_t size = 4096 * 4;
  Matrix<float> a(size, true);
  Matrix<float> b, c, d;
  parallelNuke();
  auto sequential = GetTiming([&]()
                              { b = a.transpose(); });
  parallelNuke();
  auto parallel = GetTiming([&]()
                            { d = a.ptranspose(); });
  parallelNuke();
  auto cuda = GetTiming([&]()
                        { c = cudaTranspose(a); });
  std::cout << "Sequential = " << sequential << "\n";
  std::cout << "Parallel   = " << parallel << "\n";
  std::cout << "CUDA       = " << cuda << "\n";
  EXPECT_TRUE(b == c);
  EXPECT_TRUE(b == d);
}