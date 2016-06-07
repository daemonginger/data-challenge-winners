#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

// Some tests with sparse matrices.
// Compile with g++ arma2.cpp -o prog -O2 -std=c++11 -Wall -Wextra -larmadillo

typedef unsigned char uchar;

int main(void)
{
	umat A(2,3);
	A << 0 << 1 << 2 << endr
	  << 0 << 2 << 2 << endr;
	
	Col<uchar> v(3);
	v << 1 << 1 << 1;
	SpMat<uchar> B(A,v);
	cout << B << endl;
	return 0;
}

