#ifndef TENACIOUS_TENSOR
#define TENACIOUS_TENSOR

class tensor {
    private:
        unsigned int mem_size;
        float *mem;
        float *garbage;
        unsigned int rank;
    
    public:
        unsigned int n1,n2,n3,n4;
        
        int l1i, l2i, l3i, l4i;
        int l1s, l2s, l3s, l4s;
        
        // Constructors and destructors
        tensor(void);
        ~tensor(void);
        tensor(unsigned int, unsigned int);
        tensor(unsigned int, unsigned int, unsigned int);
        tensor(unsigned int, unsigned int, unsigned int, unsigned int);
        
        // Operators
        float &operator()(const int &, const int &);
        float &operator()(const int &, const int &, const int &);
        float &operator()(const int &, const int &, const int &, const int &);
        
        tensor &operator=(const tensor &);
        
        tensor operator+(const tensor &);
        tensor operator+(const float &);
        tensor &operator+=(const tensor &);
        tensor &operator+=(const float &);
        
        tensor operator-(const tensor &);
        tensor operator-(const float &);
        tensor &operator-=(const tensor &);
        tensor &operator-=(const float &);
        
        tensor operator*(const float &);
        tensor operator*(const tensor &);
        tensor &operator*=(const float &);
        tensor &operator*=(const tensor &);
        
        // Initializing functions
        void set(unsigned int, unsigned int);
        void set(unsigned int, unsigned int, unsigned int);
        void set(unsigned int, unsigned int, unsigned int, unsigned int);
        void copyshape(const tensor &);
                
        // Methods
        void print(void);
        
        void zeros(void);
        void ones(void);
        void rand(void);
        
        void conv(const tensor &, const tensor &, const tensor &);
        void dF(const tensor &, const tensor &, const tensor &);
        void dX(const tensor &, const tensor &, const tensor &);
        
        void grad_update(const tensor &, float);
};

#endif
