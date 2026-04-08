#include <iostream>
#include <iomanip>
#include "data_loader.h"
#include "matrix.hpp"
#include "normalizer.hpp"
#include "logistic_regression.hpp"

int main() {
    std::cout << "=== Credit Risk Regularization Experiment ===\n\n";
    
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

    // Split data into train (80%) and test (20%)
    size_t train_size = static_cast<size_t>(0.8 * X_scaled.getRows());
    
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    
    auto all_data = X_scaled.getData();
    for (size_t i = 0; i < train_size; i++) {
        X_train.push_back(all_data[i]);
        y_train.push_back(y[i]);
    }
    for (size_t i = train_size; i < X_scaled.getRows(); i++) {
        X_test.push_back(all_data[i]);
        y_test.push_back(y[i]);
    }
    
    std::cout << "Train set: " << X_train.size() << " samples\n";
    std::cout << "Test set: " << X_test.size() << " samples\n\n";

    double learning_rate = 0.1;
    int epochs = 1000;
    
    // =====================
    // Experiment 1: No Regularization (Baseline)
    // =====================
    std::cout << "--- Experiment 1: No Regularization (Baseline) ---\n";
    LogisticRegression model_baseline(X_scaled.getCols());
    model_baseline.train(X_train, y_train, learning_rate, epochs);
    
    std::cout << "\nTraining Loss: " << model_baseline.computeLoss(X_train, y_train) << "\n";
    std::cout << "Test Loss: " << model_baseline.computeLoss(X_test, y_test) << "\n";
    std::cout << "\nTrain Metrics:\n";
    model_baseline.evaluate(X_train, y_train);
    std::cout << "Train AUC: " << model_baseline.computeAUC(X_train, y_train) << "\n";
    std::cout << "\nTest Metrics:\n";
    model_baseline.evaluate(X_test, y_test);
    std::cout << "Test AUC: " << model_baseline.computeAUC(X_test, y_test) << "\n";
    
    std::cout << "\nFeature Weights:\n";
    std::vector<double> weights = model_baseline.getWeights();
    std::cout << "  DebtRatio:        " << std::fixed << std::setprecision(4) << weights[0] << "\n";
    std::cout << "  ProfitMargin:     " << weights[1] << "\n";
    std::cout << "  CurrentRatio:     " << weights[2] << "\n";
    std::cout << "  RetainedEarnings: " << weights[3] << "\n";
    std::cout << "  Bias:             " << model_baseline.getBias() << "\n";

    // =====================
    // Experiment 2: L2 Regularization (Ridge)
    // =====================
    std::cout << "\n\n--- Experiment 2: L2 Regularization (Ridge) ---\n";
    std::vector<double> lambda_values = {0.01, 0.1, 1.0};
    
    for (double lambda : lambda_values) {
        std::cout << "\n[Lambda = " << lambda << "]\n";
        LogisticRegression model_l2(X_scaled.getCols());
        model_l2.setRegularization(RegularizationType::L2, lambda);
        model_l2.train(X_train, y_train, learning_rate, epochs);
        
        std::cout << "Training Loss: " << model_l2.computeLoss(X_train, y_train) << "\n";
        std::cout << "Test Loss: " << model_l2.computeLoss(X_test, y_test) << "\n";
        std::cout << "Test AUC: " << model_l2.computeAUC(X_test, y_test) << "\n";
        
        weights = model_l2.getWeights();
        std::cout << "Weights: [";
        for (size_t i = 0; i < weights.size(); i++) {
            std::cout << std::setprecision(4) << weights[i];
            if (i < weights.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "Weight Magnitude (L2 norm): " << std::setprecision(4) 
                  << std::sqrt(weights[0]*weights[0] + weights[1]*weights[1] + 
                               weights[2]*weights[2] + weights[3]*weights[3]) << "\n";
    }

    // =====================
    // Experiment 3: L1 Regularization (Lasso)
    // =====================
    std::cout << "\n\n--- Experiment 3: L1 Regularization (Lasso) ---\n";
    
    for (double lambda : lambda_values) {
        std::cout << "\n[Lambda = " << lambda << "]\n";
        LogisticRegression model_l1(X_scaled.getCols());
        model_l1.setRegularization(RegularizationType::L1, lambda);
        model_l1.train(X_train, y_train, learning_rate, epochs);
        
        std::cout << "Training Loss: " << model_l1.computeLoss(X_train, y_train) << "\n";
        std::cout << "Test Loss: " << model_l1.computeLoss(X_test, y_test) << "\n";
        std::cout << "Test AUC: " << model_l1.computeAUC(X_test, y_test) << "\n";
        
        weights = model_l1.getWeights();
        std::cout << "Weights: [";
        for (size_t i = 0; i < weights.size(); i++) {
            std::cout << std::setprecision(4) << weights[i];
            if (i < weights.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        // Count zero weights (feature selection effect)
        int zero_count = 0;
        for (double w : weights) {
            if (std::abs(w) < 1e-6) zero_count++;
        }
        std::cout << "Zero weights (feature selection): " << zero_count << " / " << weights.size() << "\n";
    }

    // =====================
    // Summary Comparison
    // =====================
    std::cout << "\n\n=== Summary Comparison ===\n";
    std::cout << "Comparing best models from each regularization type:\n\n";
    
    // Find best lambda for L2
    double best_l2_lambda = 0.01;
    double best_l2_auc = 0.0;
    for (double lambda : lambda_values) {
        LogisticRegression model(X_scaled.getCols());
        model.setRegularization(RegularizationType::L2, lambda);
        model.train(X_train, y_train, learning_rate, epochs);
        double auc = model.computeAUC(X_test, y_test);
        if (auc > best_l2_auc) {
            best_l2_auc = auc;
            best_l2_lambda = lambda;
        }
    }
    
    // Find best lambda for L1
    double best_l1_lambda = 0.01;
    double best_l1_auc = 0.0;
    for (double lambda : lambda_values) {
        LogisticRegression model(X_scaled.getCols());
        model.setRegularization(RegularizationType::L1, lambda);
        model.train(X_train, y_train, learning_rate, epochs);
        double auc = model.computeAUC(X_test, y_test);
        if (auc > best_l1_auc) {
            best_l1_auc = auc;
            best_l1_lambda = lambda;
        }
    }
    
    double baseline_auc = model_baseline.computeAUC(X_test, y_test);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Baseline (No Reg):     AUC = " << baseline_auc << "\n";
    std::cout << "Best L2 (λ=" << best_l2_lambda << "):   AUC = " << best_l2_auc << "\n";
    std::cout << "Best L1 (λ=" << best_l1_lambda << "):   AUC = " << best_l1_auc << "\n";
    
    if (best_l2_auc > baseline_auc && best_l2_auc > best_l1_auc) {
        std::cout << "\n✓ L2 regularization provides the best generalization!\n";
    } else if (best_l1_auc > baseline_auc && best_l1_auc > best_l2_auc) {
        std::cout << "\n✓ L1 regularization provides the best generalization!\n";
        std::cout << "  Bonus: L1 also performs feature selection by driving some weights to zero.\n";
    } else {
        std::cout << "\n✓ Baseline model performs competitively. Regularization may not be needed for this dataset.\n";
    }

    return 0;
}
