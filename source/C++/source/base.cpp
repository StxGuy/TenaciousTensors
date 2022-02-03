#include <iostream>
#include <cstring>
#include <random>

#include "tenten.hpp"

using namespace std;


/****************************************************************************************
 *                                    CONSTRUCTORS                                      *
 ****************************************************************************************/

// Empty
tensor::tensor(void) {
    this->rank = 0;
    this->mem_size = 0;
    this->mem = nullptr;
    this->garbage = nullptr;
    this->l1i = 0;
    this->l1s = 0;
    this->l2i = 0;
    this->l2s = 0;
    this->l3i = 0;
    this->l3s = 0;
    this->l4i = 0;
    this->l4s = 0;
    this->n1 = 0;
    this->n2 = 0;
    this->n3 = 0;
    this->n4 = 0;
}

// Matrix - rank 2
void tensor::set(unsigned int e1, unsigned int e2) {
    this->rank = 2;
    
    this->n1 = e1;
    this->n2 = e2;
    this->n3 = 0;
    this->n4 = 0;
    
    this->l1s = e1 >> 1;
    this->l1i = this->l1s - e1 + 1;
    
    this->l2s = e2 >> 1;
    this->l2i = this->l2s - e2 + 1;
    
    this->l3s = 0;
    this->l3i = 0;
    this->l4s = 0;
    this->l4i = 0;
    
    this->mem_size = e1*e2;
    if (this->mem)
        delete[] this->mem;
    this->mem = new float[mem_size];
    if (!this->garbage)
        this->garbage = new float;
    this->garbage[0] = 0;
}

// Cube - rank 3
void tensor::set(unsigned int e1, unsigned int e2, unsigned int e3) {
    this->rank = 3;
    
    this->n1 = e1;
    this->n2 = e2;
    this->n3 = e3;
    this->n4 = 0;
    
    this->l1s = e1 >> 1;
    this->l1i = this->l1s - e1 + 1;
    
    this->l2s = e2 >> 1;
    this->l2i = this->l2s - e2 + 1;
    
    this->l3s = e3 >> 1;
    this->l3i = this->l3s - e3 + 1;
    
    this->l4s = 0;
    this->l4i = 0;
    
    this->mem_size = e1*e2*e3;
    if (this->mem)
        delete[] this->mem;
    this->mem = new float[mem_size];
    if (!this->garbage)
        this->garbage = new float;
    this->garbage[0] = 0;
}

// Tesseract - rank 4
void tensor::set(unsigned int e1, unsigned int e2, unsigned int e3, unsigned int e4) {
    this->rank = 4;
    
    this->n1 = e1;
    this->n2 = e2;
    this->n3 = e3;
    this->n4 = e4;
    
    this->l1s = e1 >> 1;
    this->l1i = this->l1s - e1 + 1;
    
    this->l2s = e2 >> 1;
    this->l2i = this->l2s - e2 + 1;
    
    this->l3s = e3 >> 1;
    this->l3i = this->l3s - e3 + 1;
    
    this->l4s = e4 >> 1;
    this->l4i = this->l4s - e4 + 1;
    
    this->mem_size = e1*e2*e3*e4;
    if (this->mem)
        delete[] this->mem;
    this->mem = new float[mem_size];
    if (!this->garbage)
        this->garbage = new float;
    this->garbage[0] = 0;
}

// Constructor for matrix
tensor::tensor(unsigned int e1, unsigned int e2) {
    this->set(e1,e2);
}

// Constructor for cube
tensor::tensor(unsigned int e1, unsigned int e2, unsigned int e3) {
    this->set(e1,e2,e3);
}

// Constructor for tesseract
tensor::tensor(unsigned int e1, unsigned int e2, unsigned int e3, unsigned int e4) {
    this->set(e1,e2,e3,e4);
}

// Destructor
tensor::~tensor(void) {
    if(this->mem) {
        delete[] this->mem;
    }
    if (this->garbage) {
        delete this->garbage;
    }
}

/****************************************************************************************
 *                                   COPY SHAPE                                         *
 ****************************************************************************************/
void tensor::copyshape(const tensor &T) {
    this->rank = T.rank;
    
    this->n1 = T.n1;
    this->n2 = T.n2;
    this->n3 = T.n3;
    this->n4 = T.n4;
    
    this->l1s = T.l1s;
    this->l1i = T.l1i;
    
    this->l2s = T.l2s;
    this->l2i = T.l2i;
    
    this->l3s = T.l3s;
    this->l3i = T.l3i;
    
    this->l4s = T.l4s;
    this->l4i = T.l4i;
    
    this->mem_size = T.mem_size;
    if (this->mem)
        delete[] this->mem;
    this->mem = new float[mem_size];
    if (!this->garbage)
        this->garbage = new float;
    this->garbage[0] = 0;
}    

/****************************************************************************************
 *                                      FILLING                                         *
 ****************************************************************************************/

// zero a tensor
void tensor::zeros(void) {
    memset(this->mem,0,this->mem_size*sizeof(float));
}
// fill a tensor with 1
void tensor::ones(void) {
    fill(this->mem,this->mem+this->mem_size,1.0);
}

// fill a tensor with random numbers
void tensor::rand(void) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution <> dis(0,1);
    for(unsigned int i = 0; i < this->mem_size; i ++)
        this->mem[i] = dis(gen);
}

/****************************************************************************************
 *                                   MISCELLANEOUS                                      *
 ****************************************************************************************/

// X = X - eta.transpose(dX)
void tensor::grad_update(const tensor &grad, float eta) {
    unsigned int t0,t1,t2;
    unsigned int tos,pos;
    
    switch(this->rank) {
        case 4: // Tesseract
            for(unsigned int l = 0; l < this->n4; l ++) {
                t0 = l*this->n1*this->n2*this->n3;
                for(unsigned int k = 0; k < this->n3; k ++) {
                    t1 = t0 + k*this->n1*this->n2;
                    for(unsigned int j = 0; j < this->n2; j ++) {
                        t2 = t1 + j*this->n1;
                        for(unsigned int i = 0; i < this->n1; i ++) {
                            pos = i + t2;
                            tos = j + i*grad.n1 + t1;
                            this->mem[pos] -= eta*grad.mem[tos];
                        }
                    }
                }
            }
            break;
        case 3: // Cube
            for(unsigned int k = 0; k < this->n3; k ++) {
                t1 = k*this->n1*this->n2;
                for(unsigned int j = 0; j < this->n2; j ++) {
                    t2 = t1 + j*this->n1;
                    for(unsigned int i = 0; i < this->n1; i ++) {
                        pos = i + t2;
                        tos = j + i*grad.n1 + t1;
                        this->mem[pos] -= eta*grad.mem[tos];
                    }
                }
            }
            break;
        case 2: // Matrix
            for(unsigned int j = 0; j < this->n2; j ++) {
                    t2 = j*this->n1;
                    for(unsigned int i = 0; i < this->n1; i ++) {
                        pos = i + t2;
                        tos = j + i*grad.n1;
                        this->mem[pos] -= eta*grad.mem[tos];
                }
            }
            break;
    }
}

// print
void tensor::print(void) {
   // if (this->rank > 2) return;
    
    for(unsigned int i = 0; i < this->n1; i ++) {
        for(unsigned int j = 0; j < this->n2; j ++) {
            unsigned int pos = j*this->n1 + i;
            cout << this->mem[pos] << " ";
        }
        cout << endl;
    }
    cout << "--------------------------" << endl;
}
