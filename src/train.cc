#include <iostream>
#include <iomanip>
#include "data_loader.h"
#include "matrix.hpp"
#include "normalizer.hpp"
#include "logistic_regression.hpp"

int main() {
    std::cout << "--- Credit Risk Baseline Experiment ---\n\n";

    // 1. Load Data
    std::cout << "Loading dataset...\n";
    DataLoader loader("data/credit_risk_dataset.csv");
    auto [X_raw, y] = loader.loadData(true);
    
    std::cout << "Loaded " << X_raw.getRows() << " companies with " << X_raw.getCols() << " features.\n\n";

    // 2. Normalize Features (Z-Score)
    std::cout << "Normalizing features...\n";
    Normalizer scaler;
    scaler.fit(X_raw);
    Matrix X_scaled = scaler.transform(X_raw);

    // 3. Initialize Model
    LogisticRegression model(X_scaled.getCols());

    // 4. Train Model (Baseline, no L2 regularization yet)
    std::cout << "Training Standard Logistic Regression...\n";
    double learning_rate = 0.1;
    int epochs = 1000;
    model.train(X_scaled.getData(), y, learning_rate, epochs);

    // 5. Evaluate
    std::cout << "\n--- Baseline Results ---\n";
    model.evaluate(X_scaled.getData(), y);
    
    double auc = model.computeAUC(X_scaled.getData(), y);
    std::cout << "AUC: " << auc << "\n\n";

    // Print out the learned weights
    std::cout << "Learned Feature Weights:\n";
    std::vector<double> weights = model.getWeights();
    std::cout << "  DebtRatio:        " << std::fixed << std::setprecision(4) << weights[0] << "\n";
    std::cout << "  ProfitMargin:     " << weights[1] << "\n";
    std::cout << "  CurrentRatio:     " << weights[2] << "\n";
    std::cout << "  RetainedEarnings: " << weights[3] << "\n";
    std::cout << "  Bias:             " << model.getBias() << "\n";

    return 0;
}
