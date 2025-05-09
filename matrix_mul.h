#ifndef MATRIX_MUL
#define MATRIX_MUL
#include "matrix.hpp"


Matrix<float> cudamul(Matrix<float> &a, Matrix<float> &b);

Matrix<float> cudaTranspose(Matrix<float> &in);

#endif