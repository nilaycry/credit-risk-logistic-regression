# Regularized Logistic Regression for Corporate Default Prediction

Building a statistical model from scratch in C++ to estimate **corporate probability of default (PD)** using financial ratios.

## Research Question
*How do different regularization methods (L1 vs L2) affect the ability of logistic regression to predict corporate default risk, especially under extreme class imbalance?*

## Experimental Setup
*   **Model**: Logistic Regression Engine (built from scratch in C++)
*   **Techniques**: L1 (Lasso) and L2 (Ridge) Regularization, Threshold Optimization
*   **Metrics**: ROC-AUC, Precision, Recall
*   **Validation**: Time-Based Rolling Window Backtesting

## Repository Structure
*   `src/`: C++ Engine implementation (Data Loaders, Math Matrix, Logistic Regression, Normalizer, Metrics)
*   `scripts/`: Python ETL scripts for synthetic data generation and visualization
*   `data/`: Corporate financial ratios datasets
*   `experiments/`: Logs of model runs with different regularization strengths
*   `report/`: Analytical write-ups on methodology, economic interpretations, and real-world system design

## Current Progress
*   **Phase 1 Completed**: Core C++ Logistic Regression Engine, Matrix Algebra, Custom DataLoader, feature standardization, and Baseline Experiment orchestration.
*   **Phase 2 In-Progress**: Implementing L1/L2 Regularization calculations.
