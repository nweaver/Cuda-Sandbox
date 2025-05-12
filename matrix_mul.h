#ifndef MATRIX_MUL
#define MATRIX_MUL
#include "matrix.hpp"


Matrix<float> cudamul(Matrix<float> &a, Matrix<float> &b);

Matrix<float> cudaTranspose(Matrix<float> &in);
Matrix<float> cudamul_smallblock(Matrix<float> &a, Matrix<float> &b);
Matrix<float> cudamul_coalesce(Matrix<float> &a, Matrix<float> &b);
Matrix<float> cudamul_cache(Matrix<float> &a, Matrix<float> &b);

void CudaDeviceInfo();


#endif