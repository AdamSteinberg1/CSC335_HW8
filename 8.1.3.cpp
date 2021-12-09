#include <iostream>
#include <armadillo>
#include <cassert>
#include "Polynomial.h"

using namespace arma;
using namespace std;

Polynomial leastSquares(vec x, vec y, int n)
{
  assert(x.size()==y.size());
  assert(n > 0);

  int m = x.size();
  mat A(n+1,n+1); //A is a matrix that represents n+1 normal equations
  for(int i=0; i<n+1; i++)
  {
    for(int j=0; j<n+1; j++)
    {
      double sum = 0;
      for(const double& x_i : x)
      {
        sum += pow(x_i, i+j);
      }
      A(i,j) = sum;
    }
  }

  vec b(n+1);
  for(int i = 0; i<=n; i++)
  {
    double sum = 0;
    for(int j=0; j<m; j++)
    {
      sum += y(j)*pow(x(j), i);
    }
    b(i) = sum;
  }

  return Polynomial(conv_to<std::vector<double>>::from(solve(A,b)));
}

double leastSquaresError(Polynomial p, vec x, vec y)
{
  assert(x.size()==y.size());

  double error = 0;
  int n = x.size();
  for(int i=0; i<n; i++)
  {
    error += pow(y(i) - p.evaluate(x(i)), 2);
  }
  return error;
}

//returns a string representing a polynomial in gnuplot's format
string gnuPrint(Polynomial p)
{
  stringstream result;
  result.precision(numeric_limits<double>::max_digits10);
  int n = p.getDegree();
  for (int i =0; i <= n; i++)
  {
    if(p[i] == 0)
      continue;

    result << p[i];
    if (i != 0)
      result << "*x";

    if(i > 1)
      result << "**" << i;

    if(i < n)
      result << " + ";
  }
  return result.str();
}

void graph(vec x, vec y, Polynomial linear, Polynomial quadratic, Polynomial cubic)
{
  ofstream script("tmp.plt");

  script  << "set terminal pngcairo" << endl
          << "set output '8.1.3.png'" << endl
          << "set xrange [0:2.2]" << endl
          << "set yrange [0:4]" << endl
          << "set samples 200" << endl
          << "p1(x) = " << gnuPrint(linear) << endl
          << "p2(x) = " << gnuPrint(quadratic) << endl
          << "p3(x) = " << gnuPrint(cubic) << endl;

  script  << "plot p1(x) title 'linear', p2(x) title 'quadratic', p3(x) title 'cubic', '-' notitle" <<  endl;

  for(int i =0 ; i<x.size(); i++)
  {
    script << x(i) << '\t' << y(i) << endl;
  }

  script.close();

  system("gnuplot tmp.plt");
  system("rm tmp.plt");
}

int main()
{
  vec x = {1.0, 1.1, 1.3, 1.5, 1.9, 2.1};
  vec y = {1.84, 1.96, 2.21, 2.45, 2.94, 3.18};

  cout.precision(7);
  Polynomial linear = leastSquares(x,y,1);
  double linearError= leastSquaresError(linear, x, y);
  Polynomial quadratic = leastSquares(x,y,2);
  double quadraticError= leastSquaresError(quadratic, x, y);
  Polynomial cubic = leastSquares(x,y,3);
  double cubicError = leastSquaresError(cubic, x, y);

  cout << linear << " with an error of " << linearError << endl;
  cout << quadratic << " with an error of " << quadraticError << endl;
  cout << cubic << " with an error of " << cubicError << endl;

  graph(x, y, linear, quadratic, cubic);
}
