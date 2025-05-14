#ifndef _MATRIX_H
#define _MATRIX_H

#include <random>
#include <omp.h>
#include <cassert>



// Declaration before use
template <class T> class Matrix;
template <class T>
Matrix<T> parallel_multiply(Matrix<T> &a, Matrix<T> &b);


// This is a container for single precision square matrices in row-major form
// We implement access as a () operator rather than a [] operator because
// we can't have , in bracket operator and we want to be able to change the 
// representation in variants.
template <class T>
class Matrix {
public:

    Matrix(){
        _size = 0;
        _data = nullptr;
    }

    Matrix(size_t size, bool random = false) {
        _data = (T* ) malloc(sizeof(T) * size * size);
        _size = size;
        if(random) {randomize();}
    }

    Matrix(const Matrix &m){
        _size = m._size;
        _data = new T[_size * _size];
        for(size_t i = 0; i < (_size * _size); ++i){
            _data[i] = m._data[i];
        }
    }

    Matrix(Matrix &&m){
        _size = m._size;
        _data = m._data;
        m._data = nullptr;
        m._size = 0;
    }

    Matrix<T> &operator=(const Matrix &m){
        if (&m == this) return *this;
        if (_data) delete _data;
        _data = new T[_size * _size];
        for(size_t i = 0; i < (_size * _size); ++i){
            _data[i] = m._data[i];
        }
        return *this;
    }

    Matrix<T> &operator=(Matrix &&m){
        _size = m._size;
        _data = m._data;
        m._data = nullptr;
        m._size = 0;
        return *this;
    }

    bool operator==(const Matrix &b) const{
        if(_size != b._size) return false;
        for(size_t i = 0; i < (_size * _size); ++i){
            auto diff = b._data[i] - _data[i];
            if(diff > 0.000001 || diff < -0.00001) return false;

        }
        return true;
    }

    ~Matrix(){
        if(_data) free(_data);
    }

    void randomize() {
        //cppreference stolen from
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(1.0, 2.0);
        for(size_t i = 0; i < (_size * _size); ++i){
            _data[i] = (T) dis(gen);
        }
    
    }

    // Matrix access.   (x, y) to get x, y
    T& operator() (size_t row, size_t column) noexcept {
        assert(row < _size);
        assert(column < _size);
        return (_data[row * _size + column]);
    }

    Matrix<T> transpose() {
        Matrix<T> ret(_size);
        for(size_t i = 0; i < _size; ++i) {
            for(size_t j = 0; j < _size; ++j){
                ret(i,j) = (*this)(j, i);
            }
        }
        return ret;
    }

    Matrix<T> ptranspose() {
        Matrix<T> ret(_size);
#pragma omp parallel for
        for(size_t i = 0; i < _size; ++i) {
            for(size_t j = 0; j < _size; ++j){
                ret(i,j) = (*this)(j, i);
            }
        }
        return ret;
    }

    Matrix<T> operator*(Matrix &b) noexcept {
        assert(_size == b._size);
        Matrix<T> dest(_size);
        for(size_t i = 0; i < _size; ++i){
            for(size_t j = 0; j < _size; ++j){
                T tmp = 0;
                for(size_t k = 0; k < _size; ++k){
                    tmp += (*this)(i,k) * b(k,j);
                }
                dest(i,j) = tmp;
            }
        }
        return dest;
    }

    T* _data;
    size_t _size;
};

template <class T>
Matrix<T> parallel_multiply(Matrix<T> &a, Matrix<T> &b){
    assert(a._size == b._size);
    Matrix<T> dest(a._size);
    auto size = a._size;
#pragma omp parallel for
    for(size_t i = 0; i < size; ++i){
        for(size_t j = 0; j < size; ++j){
            T tmp = 0;
            for(size_t k = 0; k < size; ++k){
                tmp += a(i,k) * b(k,j);
            }
            dest(i,j) = tmp;
        }
    }
    return dest;
}

template <class T>
Matrix<T> parallel_multiply2(Matrix<T> &a, Matrix<T> &b){
    assert(a._size == b._size);
    Matrix<T> dest(a._size);
    auto size = a._size;
    auto dataa = a._data;
    auto datab = b._data;
    auto datad = dest._data;
#pragma omp parallel for
    for(size_t i = 0; i < size; ++i){
        auto isize = i * size;
        for(size_t j = 0; j < size; ++j){
            T tmp = 0;
            for(size_t k = 0; k < size; ++k){
                tmp += dataa[isize + k] * datab[k * size + j];
            }
            datad[isize + j] = tmp;
        }
    }
    return dest;
}


template <class T>
Matrix<T> parallel_multiply3(Matrix<T> &a, Matrix<T> &b){
    assert(a._size == b._size);
    Matrix<T> dest(a._size);
    auto size = a._size;
    auto dataa = a._data;
    auto datab = b._data;
    auto datad = dest._data;
#pragma omp parallel for schedule(guided, 1)
    for(size_t i = 0; i < size; ++i){
        auto isize = i * size;
        for(size_t j = 0; j < size; ++j){
            T tmp = 0;
            for(size_t k = 0; k < size; ++k){
                tmp += dataa[isize + k] * datab[k * size + j];
            }
            datad[isize + j] = tmp;
        }
    }
    return dest;
}

template <class T>
Matrix<T> parallel_multiply4(Matrix<T> &a, Matrix<T> &b){
    assert(a._size == b._size);
    Matrix<T> dest(a._size);
    auto size = a._size;
    auto dataa = a._data;
    auto tmp = b.transpose();
    auto datab = tmp._data;
    auto datad = dest._data;
#pragma omp parallel for schedule(guided, 1)
    for(size_t i = 0; i < size; ++i){
        auto isize = i * size;
        for(size_t j = 0; j < size; ++j){
            T tmp = 0;
            for(size_t k = 0; k < size; ++k){
                tmp += dataa[isize + k] * datab[j * size + k];
            }
            datad[isize + j] = tmp;
        }
    }
    return dest;
}

 

#endif