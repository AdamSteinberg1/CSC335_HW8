#include <iostream>
#include <functional>
#include <armadillo>
using namespace std;
using namespace arma;

#define TOLERANCE 0.005
#define MAX_ITERATIONS 1000

vec steepestDescent(vec x, function<double(vec)> g, function<vec(vec)> g_gradient)
{
  for(int k=0; k<MAX_ITERATIONS; k++)
  {
    double g1 = g(x);
    vec z = g_gradient(x);
    double z0 = norm(z, 2);
    if(z0 == 0)
    {
      cout << "Zero gradient\n";
      return x;
    }
    z = z/z0;
    double a1 = 0;
    double a3 = 1;
    double g3 = g(x-a3*z);
    while(g3 >= g1)
    {
      a3 /= 2;
      g3 = g(x-a3*z);
      if(a3 < TOLERANCE/2)
      {
        cout << "No likely improvement\n";
        return x;
      }
    }
    double a2 = a3/2;
    double g2 = g(x-a2*z);
    double h1= (g2-g1)/a2;
    double h2 = (g3-g2)/(a3-a2);
    double h3 = (h2-h1)/a3;
    double a0 = 0.5*(a2-h1/h3);
    double g0 = g(x-a0*z);
    double g, a;
    tie(g,a) = min({make_pair(g0, a0), make_pair(g1, a1), make_pair(g2, a2), make_pair(g3,a3)});
    x -= a*z;
    if(fabs(g-g1) < TOLERANCE)
    {
      return x;
    }
  }
  cout << "Maximum iterations exceeded\n";
  return x;
}


int main()
{
  auto g = [](vec x){
    return x(0)*x(0)+2*x(1)*x(1)+x(2)*x(2)-2*x(0)*x(1)+2*x(0)-2.5*x(1)-x(2)+2;
  };

  auto g_gradient = [](vec x){
    return vec({2*x(0)-2*x(1)+2, -2*x(0)+4*x(1)-2.5, 2*x(2)-1});
  };

  vec guess = {0,0,0};

  vec x_min = steepestDescent(guess, g, g_gradient); //location of the minimum
  double minimum = g(x_min);
  cout.precision(8);
  cout << "Minimum of " << minimum << " at x = (" << x_min(0) << ", " << x_min(1) << ", " << x_min(2) << ")\n";
}
