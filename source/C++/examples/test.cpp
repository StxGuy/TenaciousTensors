#include <tenten>

int main(void) {
    tensor A,B;
    
    A.set(2,2);
    B.set(2,2);
    
    A.ones();
    B.ones();
    
    B += A;
    
    B.print();
}
    
