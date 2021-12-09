#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace arma;
using namespace std;

#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6

mat jacobian(vec x, vector<function<vec(vec)>> partialDerivatives)
{
  int n = x.size();
  mat J(n,n);
  for(int i = 0; i < n; i++)
  {
    J.col(i) = partialDerivatives[i](x);
  }
  return J;
}

vec newtonsMethod(vec x, function<vec(vec)> F, vector<function<vec(vec)>> partialDerivatives)
{
  for(int k=0; k<MAX_ITERATIONS; k++)
  {
    mat J = jacobian(x, partialDerivatives);
    vec y = solve(J, -F(x));
    x += y;
    if(norm(y, "inf") < TOLERANCE)
    {
      return x;
    }
  }
  cout << "Max Iterations Exceeded" << endl;
  return x;
}

int main()
{
  auto F = [](vec x){
    return vec({log(x(0)*x(0) + x(1)*x(1)) - sin(x(0)*x(1)) - M_LN2 - log(M_PI),
                exp(x(0)-x(1)) + cos(x(0)*x(1))});
  };

  //partial derivative of F with respect to x1
  function<vec(vec)> dFdx1 = [](vec x){
    return vec({2*x(0)/(x(0)*x(0) + x(1)*x(1)) - x(1)*cos(x(0)*x(1)),
                exp(x(0)-x(1)) - x(1)*sin(x(0)*x(1))});
  };

  //partial derivative of F with respect to x2
  function<vec(vec)> dFdx2 = [](vec x){
    return vec({2*x(1)/(x(0)*x(0) + x(1)*x(1)) - x(0)*cos(x(0)*x(1)),
                -exp(x(0)-x(1)) - x(0)*sin(x(0)*x(1))});
  };

  vector<function<vec(vec)>> partialDerivatives = {dFdx1, dFdx2};

  //intital approximation
  vec x0 = {2, 2};

  cout.precision(7);
  vec solution = newtonsMethod(x0, F, partialDerivatives);
  cout << "x =\n";
  solution.raw_print();
}
