#include "tenten.hpp"

/****************************************************************************************
 *                                    CONVOLUTION                                       *
 ****************************************************************************************/
void tensor::conv(const tensor &X, const tensor &F, const tensor &B) {
    unsigned int pos0, pos1, pos2;
    unsigned int fos0, fos1, fos2, fos3;
    unsigned int xos0, xos1, xos2;
    unsigned int delta1,delta2,delta3;
    int t1, t2;
    
    if (this->n1 != X.n1 || this->n2 != X.n2 || this->n3 != F.n4) 
        this->set(X.n1,X.n2,F.n4);
           
    delta1 = this->n1*this->n2;
    delta2 = F.n1*F.n2;
    delta3 = delta2*F.n3;
    
        
    // Y <- this can be parallelized
    for(int k = this->l3i; k <= this->l3s; k ++) {
        pos0 = (k - this->l3i)*delta1;
        fos0 = (k - F.l4i)*delta3;
        for(int j = this->l2i; j <= this->l2s; j ++) {
            pos1 = pos0 + (j-this->l2i)*this->n1;
            for(int i = this->l1i; i <= this->l1s; i ++) {
                pos2 = pos1 + i-this->l1i; 
                this->mem[pos2] = B.mem[pos2]; 
                              
                // Filter
                for(int p = F.l3i; p <= F.l3s; p ++) {
                    xos0 = (p - F.l3i)*delta2;
                    fos1 = fos0 + xos0;
                    for(int m = F.l2i; m <= F.l2s; m ++) {
                        t1 = j-m;
                        if (t1 >= X.l2i && t1 <= X.l2s) {
                            fos2 = fos1 + (m - F.l2i)*F.n1;
                            xos1 = xos0 + (t1 - X.l2i)*X.n1;
                            for(int n = F.l1i; n <= F.l1s; n ++) {
                                t2 = i-n;
                                if (t2 >= X.l1i && t2 <= X.l1s) {
                                    fos3 = fos2 + n - F.l1i;
                                    xos2 = xos1 + t2 - X.l1i;
                                    this->mem[pos2] += F.mem[fos3]*X.mem[xos2];
                                }
                            }
                        }
                    }
                } 
            }
        }
    } 
}

/****************************************************************************************/
/*                                       dLdF                                           */
/****************************************************************************************/
void tensor::dF(const tensor &dL, const tensor &X, const tensor &F) {
    unsigned int fos0, fos1, fos2, fos3;
    unsigned int yos0, yos1, yos2;
    unsigned int xos0, xos1, xos2;
    unsigned int delta1, delta2, delta3;
    int t1,t2;
    
    if (this->n1 != F.n2 || this->n2 != F.n1 || this->n3 != X.n3 || this->n4 != dL.n3)
        this->set(F.n2,F.n1,X.n3,dL.n3);
    
    delta3 = this->n1*this->n2;
    delta1 = delta3*this->n3;
    delta2 = dL.n1*dL.n2;
        
    for(int d = this->l4i; d <= this->l4s; d ++) {
        fos0 = (d - this->l4i)*delta1;
        yos0 = (d - dL.l3i)*delta2;
        for(int c = this->l3i; c <= this->l3s; c ++) {
            fos1 = fos0 + (c - this->l3i)*delta3;
            xos0 = (c-X.l3i)*X.n1*X.n2;
            for(int b = this->l2i; b <= this->l2s; b ++) {
                fos2 = fos1 + (b - this->l2i)*this->n1;
                for(int a = this->l1i; a <= this->l1s; a ++) {
                    fos3 = fos2 + (a - this->l1i);
                    this->mem[fos3] = 0;
                    
                    // X
                    for(int j = dL.l2i; j <= dL.l2s; j ++) {
                        yos1 = yos0 + (j-dL.l2i)*dL.n1;
                        t1 = j-b;
                        if (t1 >= X.l1i && t1 <= X.l1s) {
                            xos1 = xos0 + (t1-X.l1i);
                            for(int i = dL.l1i; i <= dL.l1s; i ++) {
                                yos2 = yos1 + (i-dL.l1i);
                                t2 = i-a;
                                if (t2 >= X.l2i && t2 <= X.l2s) {
                                    xos2 = xos1 + (t2-X.l2i)*X.n1;
                                    this->mem[fos3] += dL.mem[yos2]*X.mem[xos2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/****************************************************************************************/
/*                                       dLdX                                           */
/****************************************************************************************/
void tensor::dX(const tensor &dL, const tensor &F, const tensor &X) {
    unsigned int fos0, fos1, fos2, fos3;
    unsigned int xos0, xos1, xos2;
    unsigned int yos0, yos1, yos2;
    unsigned int delta1, delta2, delta3;
    int t1,t2;
    
    delta1 = this->n1*this->n2;
    delta2 = F.n1*F.n2;
    delta3 = F.n1*F.n2*F.n3;
    
    if (this->n1 != X.n2 || this->n2 != X.n1 || this->n3 != F.n3)
        this->set(X.n2,X.n1,F.n3);
    
    for(int c = this->l3i; c <= this->l3s; c++) {
        xos0 = (c-this->l3i)*delta1;
        fos0 = (c-this->l3i)*delta2;
        for(int b = this->l2i; b <= this->l2s; b++) {
            xos1 = xos0 + (b-this->l2i)*this->n1;
            for(int a = this->l1i; a <= this->l1s; a++) {
                xos2 = xos1 + (a-this->l1i);
                this->mem[xos2] = 0;
                
                // F
                for(int k = dL.l3i; k <= dL.l3s; k ++) {
                    yos0 = (k-dL.l3i)*dL.n1*dL.n2;
                    fos1 = fos0 + (k-dL.l3i)*delta3;
                    for(int j = dL.l2i; j <= dL.l2s; j ++) {
                        yos1 = yos0 + (j-dL.l2i)*dL.n1;
                        t1 = j-b;
                        if (t1 >= F.l1i && t1 <= F.l1s) {
                            fos2 = fos1 + (t1-dL.l1i)*F.n1;
                            for(int i = dL.l1i; i <= dL.l1s; i ++) {
                                yos2 = yos1 + (i-dL.l1i);
                                t2 = i-a;
                                if (t2 >= F.l2i && t2 <= F.l2s) {
                                    fos3 = fos2 + (t2-dL.l2i);
                                    this->mem[xos2] += dL.mem[yos2]*F.mem[fos3];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
