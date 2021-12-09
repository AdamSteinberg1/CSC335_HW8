#include <functional>
#include <armadillo>
#include <cmath>
using namespace arma;
using namespace std;

#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-5

vec fixedPointIteration(vec x0, function<vec(vec)> G)
{
  for (int i=0; i<MAX_ITERATIONS; i++)
  {
    vec x = G(x0);
    if(norm(x - x0, "inf") < TOLERANCE)
    {
      return x;
    }
    x0 = x;
  }
  cout << "Max Iterations exceeded" << endl;
  return x0;
}

int main()
{
  //fixed point equation
  auto G = [](vec x){
    if(x(0) > -0.5)
      return vec({(-1+sqrt(8*x(1)-71))/2, sqrt(25-pow(x(0)-1,2))+6});
    else
      return vec({(-1-sqrt(8*x(1)-71))/2, sqrt(25-pow(x(0)-1,2))+6});
  };

  //initial guesses
  vec guess1 = {-3, 11};
  vec guess2 = {1.5, 11};

  vec solution1 = fixedPointIteration(guess1, G);
  vec solution2 = fixedPointIteration(guess2, G);

  cout.precision(7);
  cout << "First solution =\n";
  solution1.raw_print();
  cout << endl;
  cout << "Second solution =\n";
  solution2.raw_print();
}
