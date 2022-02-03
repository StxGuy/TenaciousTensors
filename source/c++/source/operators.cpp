#include <iostream>
#include <cstring>
#include <cblas.h>

#include "tenten.hpp"

using namespace std;

/****************************************************************************************
 *                                   OPERATOR ()                                        *
 ****************************************************************************************/

// Matrix - rank 2
float &tensor::operator()(const int &i, const int &j) {
    if (i < this->l1i || i > this->l1s || j < this->l2i || j > this->l2s) {
        cout << "WARNING: Trying to access element out of bounds..." << endl;
        return this->garbage[0];
    }
    
    int pos = (j-this->l2i)*this->n1 + (i-this->l1i);
    
    return this->mem[pos];
}

// Cube - rank 3
float &tensor::operator()(const int &i, const int &j, const int &k) {
    if (i < this->l1i || i > this->l1s || j < this->l2i || j > this->l2s ||
        k < this->l3i || k > this->l3s)
    {
        cout << "WARNING: Trying to access element out of bounds..." << endl;
        return this->garbage[0];
    }
    
    int pos = (k-this->l3i)*this->n1*this->n2 + (j-this->l2i)*this->n1 + (i-this->l1i);
    
    return this->mem[pos];
}   

// Tesseract - rank 4
float &tensor::operator()(const int &i, const int &j, const int &k, const int &l) {
    if (i < this->l1i || i > this->l1s || j < this->l2i || j > this->l2s ||
        k < this->l3i || k > this->l3s || l < this->l4i || l > this->l4s)
    {
        cout << "WARNING: Trying to access element out of bounds..." << endl;
        return this->garbage[0];
    }    
    int pos = (l-this->l4i)*this->n1*this->n2*this->n3 + (k-this->l3i)*this->n1*this->n2 + (j-this->l2i)*this->n1 + (i-this->l1i);
    
    return this->mem[pos];
}   

/****************************************************************************************
 *                                   OPERATOR =                                         *
 ****************************************************************************************/
tensor &tensor::operator=(const tensor &other) { 
    if (this == &other)
        return *this;
    
    this->rank = other.rank;
    
    this->n1 = other.n1;
    this->n2 = other.n2;
    this->n3 = other.n3;
    this->n4 = other.n4;
    
    this->l1s = other.l1s;
    this->l1i = other.l1i;
    
    this->l2s = other.l2s;
    this->l2i = other.l2i;
    
    this->l3s = other.l3s;
    this->l3i = other.l3i;
    
    this->l4s = other.l4s;
    this->l4i = other.l4i;
    
    this->mem_size = other.mem_size;
    if (this->mem)
        delete[] this->mem;
    this->mem = new float[this->mem_size];
    
    memcpy(this->mem,other.mem,this->mem_size*sizeof(float));
    
    if (!this->garbage)
        this->garbage = new float;
    this->garbage[0] = 0;
    
    return *this;
}    

/****************************************************************************************
 *                                 OPERATOR +,+=                                        *
 ****************************************************************************************/
tensor tensor::operator+(const tensor &other) {
    tensor res;
    
    res.copyshape(other);
        
    for(unsigned int i = 0; i < this->mem_size; i ++)
        res.mem[i] = this->mem[i] + other.mem[i];
    
    return res;
}

tensor tensor::operator+(const float &a) {
    tensor res;
    
    res.copyshape(*this);
    
    for(unsigned int i = 0; i < this->mem_size; i ++)
        res.mem[i] = this->mem[i] + a;
    
    return res;
}

tensor &tensor::operator+=(const tensor &other) {
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] += other.mem[i];
    
    return *this;
}

tensor &tensor::operator+=(const float &a) {
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] += a;
    
    return *this;
}

/****************************************************************************************
 *                                 OPERATOR -,-=                                        *
 ****************************************************************************************/
tensor tensor::operator-(const tensor &other) {
    tensor res;
    
    res.copyshape(other);
    
    for(unsigned int i = 0; i < this->mem_size; i ++)
        res.mem[i] = this->mem[i] - other.mem[i];
    
    return res;
} 

tensor tensor::operator-(const float &a) {
    tensor res;
    
    res.copyshape(*this);
    
    for(unsigned int i = 0; i < this->mem_size; i ++)
        res.mem[i] = this->mem[i] - a;
    
    return res;
}

tensor &tensor::operator-=(const tensor &other) {
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] -= other.mem[i];
    
    return *this;
} 

tensor &tensor::operator-=(const float &a) {
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] -= a;
    
    return *this;
}

/****************************************************************************************
 *                                 OPERATOR *,*=                                        *
 ****************************************************************************************/
tensor tensor::operator*(const float &a) {
    tensor res;
    
    res.copyshape(*this);
    
    for(unsigned int i = 0; i < this->mem_size; i ++)
        res.mem[i] = this->mem[i] * a;
    
    return res;
}

tensor tensor::operator*(const tensor &other) {
    tensor res;
    
    res.set(this->n1,other.n2);
    
    if (other.n2 == 1) {
        // alpha.Ax + beta.y
        cblas_sgemv(CblasColMajor,  // Layout
                    CblasNoTrans,   // trans
                    this->n1,       // number of rows
                    this->n2,       // number of columns
                    1.0,            // alpha
                    this->mem,      // A
                    this->n1,       // lda
                    other.mem,      // x
                    1,              // incx
                    0,              // beta
                    res.mem,        // y
                    1);             // incy
    }
    else {
        // alpha.AB + beta.C
        cblas_sgemm(CblasColMajor,  // Layout
                    CblasNoTrans,   // transa
                    CblasNoTrans,   // transb
                    this->n1,       // number of rows of A
                    other.n2,       // number of columns of B
                    this->n2,       // number of columns of A or rows of B
                    1.0,            // alpha
                    this->mem,      // A
                    this->n1,       // lda
                    other.mem,      // B
                    other.n1,       // ldb
                    0,              // beta
                    res.mem,        // C
                    this->n1);      // ldc
    }    
    
    return res;
}

tensor &tensor::operator*=(const float &a) {
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] *= a;
    
    return *this;
}


tensor &tensor::operator*=(const tensor &other) {
        
    if (other.n2 == 1) {
        // alpha.Ax + beta.y
        cblas_sgemv(CblasColMajor,  // Layout
                    CblasNoTrans,   // trans
                    this->n1,       // number of rows
                    this->n2,       // number of columns
                    1.0,            // alpha
                    this->mem,      // A
                    this->n1,       // lda
                    other.mem,      // x
                    1,              // incx
                    0,              // beta
                    this->mem,      // y
                    1);             // incy
    }
    else {
        float mem[this->n1*other.n2];
        
        // alpha.AB + beta.C
        cblas_sgemm(CblasColMajor,  // Layout
                    CblasNoTrans,   // transa
                    CblasNoTrans,   // transb
                    this->n1,       // number of rows of A
                    other.n2,       // number of columns of B
                    this->n2,       // number of columns of A or rows of B
                    1.0,            // alpha
                    this->mem,      // A
                    this->n1,       // lda
                    other.mem,      // B
                    other.n1,       // ldb
                    0,              // beta
                    mem,            // C
                    this->n1);      // ldc
    
        memcpy(this->mem,mem,this->mem_size*sizeof(float));
    }    
    
    return *this;
}
