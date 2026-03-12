#include "logistic_regression.hpp"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <algorithm>


struct EvalPair {
   double prob;
   int label;
};


bool pairComparator(const EvalPair& a, EvalPair& b) {
  return a.prob < b.prob;
}

LogisticRegression::LogisticRegression(int numFeatures) {
  // input validation
  if (numFeatures <= 0) {
    throw std::runtime_error("Invalid number of features");
  }

  weights_.resize(numFeatures);
}

double LogisticRegression::predict(const std::vector<double> &x) const {

  double prediction = 0.0;
  // Compute the dot product between input features and weights
  for (size_t i = 0; i < x.size(); i++) {
    prediction += x[i] * weights_[i];
  }

  // Add the bias term to complete linear equation: y = w*x + b
  prediction += bias_;
  return sigmoid(prediction);
}

void LogisticRegression::train(const std::vector<std::vector<double>> &x,
                               const std::vector<double> &y, double lr,
                               int epochs) {

  if (x.size() != y.size()) {
    throw std::runtime_error("Invalid input dimenstions");
  }
  // Run the training loop for the specified number of epochs
  for (int epoch = 0; epoch < epochs; epoch++) {
    // Initialize the gradients for weights and bias to 0
    std::vector<double> grad_w(weights_.size(), 0.0);
    double grad_b = 0.0;

    // Loop through all training examples to compute the gradients
    for (size_t i = 0; i < x.size(); i++) {
      // Calculate the residual (error) for the current prediction
      double error = y[i] - predict(x[i]);

      // Accumulate the gradients for each weight based on error and input
      for (size_t j = 0; j < weights_.size(); j++) {
        grad_w[j] += x[i][j] * error;
      }
      // Accumulate the gradient for the bias
      grad_b += error;
    }

    // Apply gradient descent step for weights (average gradient across
    // examples)
    for (size_t j = 0; j < weights_.size(); j++) {
      weights_[j] += lr * grad_w[j] / static_cast<double>(x.size());
    }
    // Apply gradient descent step for bias
    bias_ += lr * grad_b / static_cast<double>(x.size());

    // std::cout << "Epoch: " << epoch << "    Loss: " << computeLoss(x, y) <<
    // std::endl;
  }
}

int LogisticRegression::predictClass(const std::vector<double> &x) const {
  return predict(x) >= 0.5 ? 1 : 0;
}

void LogisticRegression::evaluate(const std::vector<std::vector<double>> &x,
                                  const std::vector<double> &y) const {
  if (x.size() != y.size()) {
    throw std::runtime_error("Invalid input dimenstions");
  }
  size_t size = x.size();

  int true_pos = 0;
  int true_neg = 0;
  int false_pos = 0;
  int false_neg = 0;

  for (size_t i = 0; i < size; i++) {
    int prediction = predictClass(x[i]);
    if (y[i] == 1 && prediction == 1) {
      true_pos++;
    } else  if (y[i] == 1 && prediction != 1) {
      false_neg++;
    } else if (y[i] == 0 && prediction == 0) {
      true_neg++;
    } else {
      false_pos++;
    } 
  }

  double accuracy = static_cast<double>(true_pos + true_neg) /
                     static_cast<double>(true_pos + true_neg + false_neg + false_pos);
  double precision = static_cast<double>(true_pos) / static_cast<double>(true_pos + false_pos);
  double recall = static_cast<double>(true_pos) / static_cast<double>(true_pos + false_neg);


  std::cout << "Accuracy: " << accuracy << '\n' << "Precision: " << 
            precision << '\n' << "Recall: " << recall << std::endl;
}

double LogisticRegression::computeAUC(const std::vector<std::vector<double>> &x, const std::vector<double> &y) const {
  // declare a local struct to ease computation

  std::vector<EvalPair> vect;

  for (size_t i = 0; i < x.size(); i++) {    
    vect.push_back({predict(x[i]), static_cast<int>(y[i])});
  }

  std::sort(vect.begin(), vect.end(), pairComparator);

  int total_ones = 0;
  int total_zeros = 0;
  int total_rankings = 0;

  for (auto ele: vect) {
    if (ele.label == 0) {
      total_zeros++;
    }

    if (ele.label == 1) {
      total_rankings += total_zeros;
      total_ones++;
    }
  }

  double auc = static_cast<double>(total_rankings) / static_cast<double>(total_ones * total_zeros);

  return auc;
}

std::vector<double> LogisticRegression::getWeights() const { return weights_; }

double LogisticRegression::getBias() const { return bias_; }

int LogisticRegression::getnumFeatures() const { return weights_.size(); }

double LogisticRegression::sigmoid(double z) const {
  return 1.0 / (1.0 + std::exp(-z));
}