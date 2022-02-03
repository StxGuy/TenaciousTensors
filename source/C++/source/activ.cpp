#include <cmath>

#include "tenten.hpp"


tensor tensor::activate(string activation) {
    tensor Z;
    
    Z.copyshape(*this);
        
    for(unsigned int i = 0; i < this->mem_size; i ++) {
        if (activation == "relu") {
            if (this->mem[i] > 0) {
                Z.mem[i] = this->mem[i];
            }
            else
                Z.mem[i] = 0;
        }
        if (activation == "sigmoid") {
            if (this->mem[i] > 100) 
                Z.mem[i] = 1.0/(1.0 + exp(-100));
            else if (this->mem[i] < -100) 
                Z.mem[i] = 1.0/(1.0 + exp(100));
            else
                Z.mem[i] = 1.0/(1.0 + exp(-this->mem[i]));
        }
        if (activation == "tanh") {
            if (this->mem[i] > 100)
                Z.mem[i] = tanh(100);
            else if (this->mem[i] < -100)
                Z.mem[i] = tanh(-100);
            else
                Z.mem[i] = tanh(this->mem[i]);
        }
    }
    
    return Z;
}

tensor tensor::deactivate(string activation) {
    tensor Y;
    
    Y.copyshape(*this);

    
    for(unsigned int i = 0; i < this->mem_size; i ++) {
        if (activation == "relu") {
            if (this->mem[i] > 0)
                Y.mem[i] = 1;
            else
                Y.mem[i] = 0;
        }
        if (activation == "sigmoid") 
            Y.mem[i] = this->mem[i]*(1-this->mem[i]);
        if (activation == "tanh")
            Y.mem[i] = 1 - this->mem[i]*this->mem[i];
    }
    
    return Y;
}
    
    
