# Credit Risk & Corporate Default Prediction - Capstone Project

## Project Overview
This project is a C++ implementation of a logistic regression model trained from scratch to predict the Probability of Default (PD) for public companies based on their financial ratios. This serves as a quantitative finance capstone project demonstrating statistical modeling, C++ engineering, and data processing skills.

## Core Objectives
1.  **C++ Logistic Regression Engine:** Utilize a custom-built logistic regression and linear algebra (Matrix) engine.
2.  **C++ Data Engineering:** Build a robust `DataLoader` class in C++ using `std::ifstream`, `std::stringstream`, etc. to parse raw, messy CSV financial data and handle missing values (`NaN`).
3.  **Handling Extreme Imbalance:** Financial default datasets are heavily imbalanced (e.g., 98% survival / 2% default). The project will evaluate model effectiveness using appropriate metrics like AUC (Area Under the Curve) and Precision/Recall, rather than just strict accuracy.
4.  **Economic Interpretation:** Implement analysis of learned coefficients (e.g., impact of Debt-to-Equity ratio on default probability).
5.  **Model Enhancements:**
    *   Implement L2 Regularization (Ridge Regression) to prevent overfitting.
    *   Implement k-fold cross-validation for robust training.
6.  **Final Deliverable:** A command-line application (`driver.cc`) that accepts a company ticker/data, processes its ratios through the trained weights, and outputs a Credit Risk Profile with the PD and primary risk factors.

## Implementation Guidelines
*   **No Code Generation Replacements:** The user (Nilay) is an undergrad studying math at UIUC. The goal is *guided learning*. Agents should discuss math, intuition, architecture, and debugging strategies *without* just dropping the final block of code.
*   **Debugging Approach:** When encountering issues, agents should point out the logic flaw or request diagnostic information rather than fixing the code directly.
*   **Focus Areas:** Memory management, C++ best practices, linear algebra optimization, and statistical soundness.

## Proposed Directory Structure
```
credit-risk-project/
    src/
        logistic_regression.cpp
        data_loader.cpp
        metrics.cpp
        driver.cpp
    include/
        logistic_regression.h
        data_loader.h
        metrics.h
    data/
    experiments/
    report/
        methodology.md
        results.md
    README.md
```

## Immediate Next Steps (as of Initialization)
1. Discuss the math and architecture for L2 Regularization or the `DataLoader`.
2. Begin implementation of the chosen first phase.
