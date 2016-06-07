// -*- c++ -*- //
/***************************************************************************
 -   begin                : 2003-09-05
 -   copyright            : (C) 2003 by Gunter Winkler
 -   email                : guwi17@gmx.de
 -   This program is free. Use it at your own risk                
 ***************************************************************************/

// Just an example of how to use sparse matrices in boost

#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas ;

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  const size_t size = 10;
  // define a row-major sparse matrix
  compressed_matrix<double>  A(size, size);

  A.push_back(0,8,10.);
  A.push_back(0,9,10.);
  for (size_t i=1; i<A.size1(); ++i)	{
	if (i>=1) { 
	  A.push_back(i,i-1,-1.0); 
	}
	A.push_back(i,i,4);
	if (i+1<A.size2()) { 
	  A.push_back(i,i+1,-1.0); 
	}
  }
  for(int i=0;i<size;i++)
  {
	  for(int j=0;j<size;j++)
		  cout << A(i,j) << " ";
	  cout << endl;
  }

  return EXIT_SUCCESS;
};
