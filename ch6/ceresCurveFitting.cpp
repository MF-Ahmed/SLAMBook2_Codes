//
// Created by xiang on 18-11-19.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// calculation model of cost function
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  // calculation of residuals
  template<typename T>
  // we define a functor which will be used later by the solver 
  bool operator()(
    const T *const abc, // model parameter, has 3 dimensions
    T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y- exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y; // x,y data
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0; // actual parameter value
  double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameter values
  int N = 100; // data point
  double w_sigma = 1.0; // Noise Sigma value
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng; // OpenCV random number generator

  vector<double> x_data, y_data; // data
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  double abc[3] = {ae, be, ce}; // initial guess


  // Construct the least squares problem
  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
    problem.AddResidualBlock( // Add an error term to the problem
      // Use automatic derivation, template parameters: error type, output dimension, input dimension, the dimension should be consistent with the previous struct
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
        new CURVE_FITTING_COST(x_data[i], y_data[i])
      ),
      nullptr, // kernel function, not used here, empty
      abc // parameters to be estimated
    );
  }

  // configure the solver
  ceres::Solver::Options options; // There are many configuration options to fill in
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // how the incremental equation is solved
  options.minimizer_progress_to_stdout = true; // output to cout

  ceres::Solver::Summary summary; // Optimization information
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary); // start optimization
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // output result
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a:abc) cout << a << " ";
  cout << endl;
  return 0;
}
