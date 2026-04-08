#pragma once
#include <cmath>
#include <vector>

enum class RegularizationType {
    NONE,
    L1,
    L2
};

class LogisticRegression {
private:
  std::vector<double> weights_;
  double bias_ = 0.0;
  RegularizationType reg_type_ = RegularizationType::NONE;
  double lambda_ = 0.0;  // Regularization strength
  
  // Activation function
  double sigmoid(double z) const;
  // Sign function for L1 regularization
  int sign(double w) const;

public:
  LogisticRegression(int numFeatures);
  
  // Set regularization parameters
  void setRegularization(RegularizationType type, double lambda);
  
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
  double computeLoss(const std::vector<std::vector<double>> &x, const std::vector<double> &y) const;

};
