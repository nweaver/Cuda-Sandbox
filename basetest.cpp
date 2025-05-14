#include <gtest/gtest.h>
#include <string>

#include "vector_add.h"
#include "parallel_utils.hpp"
#include "matrix.hpp"
#include "matrix_mul.h"
#include <ranges>
#include <omp.h>

// The STB libraries do introduce a couple of compiler warnings
// in my preferred paranoid compiler settings, so...
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wconversion"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma GCC diagnostic pop

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
  size_t size = 128; // 4096;
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
  size_t size = 64; // 4096 * 4;
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

TEST(Image, Simple)
{
  int x, y, n;
  unsigned char *data = stbi_load("../Lena_2048.png", &x, &y, &n, 0);
  if (data)
  {
    std::cout << "Size x: " << x << " y: " << y << " n: " << n << "\n";
    unsigned char * data2 = (unsigned char *) malloc(sizeof(unsigned char) * 64 * 64 * 3);
    for (auto xi = 0; xi < 64; ++xi)
    {
      for (auto yi = 0; yi < 64; ++yi)
      {
        data2[xi * 3 + yi * 3 * 64 ] = data[xi * 32 * 3 + yi * 2048 * 3 * 32];
        data2[xi * 3 + yi * 3 * 64 + 1] = data[xi * 32 * 3 + yi * 2048 * 3 * 32 + 1];
        data2[xi * 3 + yi * 3 * 64 + 2] = data[xi * 32 * 3 + yi * 2048 * 3 * 32 + 2];
      }
    }
      // Stride is separating between rows to keep things aligned.

    auto ret = stbi_write_png("./foo.png", 64, 64, 3, data2, 64 * 3);
    if (!ret)
    {
      std::cerr << "Failed to write test png\n";
      EXPECT_TRUE(false);
    }

    stbi_image_free(data);
    free(data2);
  }
  else
  {
    std::cerr << "Unable to load file!, reason: " << stbi_failure_reason() << "\n";
    EXPECT_TRUE(false);
  }

}
