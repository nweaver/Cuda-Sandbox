#include <gtest/gtest.h>
#include <string>

#include "vector_add.h"

// Demonstrate some basic assertions.
TEST(SimpleTest, CanCallCudaCode)
{
  EXPECT_TRUE(true);
  EXPECT_TRUE(cuda_call_test());
}
