#pragma once
#include <cmath>
#include <vector>

class LogisticRegression {
private:
  std::vector<double> weights_;
  double bias_ = 0.0;
  // Activation function
  double sigmoid(double z) const;

public:
  LogisticRegression(int numFeatures);
  double predict(const std::vector<double> &x) const;
  void train(const std::vector<std::vector<double>> &x,
             const std::vector<double> &y, double lr, int epochs);

  std::vector<double> getWeights() const;
  double getBias() const;
  int getnumFeatures() const;
  int predictClass(const std::vector<double> &x) const;
  void evaluate(const std::vector<std::vector<double>> &x,
                const std::vector<double> &y) const;
  
  double computeAUC(const std::vector<std::vector<double>> &x, const std::vector<double> &y) const;



};
