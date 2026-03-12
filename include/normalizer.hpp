#pragma once
#include "matrix.hpp"
#include <vector>


/**
 * @brief A utility class for normalizing data (Z-score normalization).
 * Ensures features (columns) have zero mean and unit variance.
 */
class Normalizer {
    private: 
        std::vector<double> means_; // Vector storing the computed mean for each feature column
        std::vector<double> stds_;  // Vector storing the computed std dev for each feature column
        
    public:
        Normalizer();
        
        /**
         * @brief Computes the mean and standard deviation for each column (feature) in the input matrix.
         * @param x A 2D Matrix where rows are samples and columns are features.
         */
        void fit(const Matrix& x);
        
        /**
         * @brief Normalizes a 2D Matrix using the previously fitted means and standard deviations.
         * @param x The 2D Matrix to transform.
         * @return A new Matrix identical in dimensions to x, but with normalized features.
         */
        Matrix transform(const Matrix& x) const;
        
        /**
         * @brief Normalizes a single 1D vector of features (e.g., a single company's data).
         * @param x The single data point (vector of features) to transform.
         * @return The normalized data vector.
         */
        std::vector<double> transformSingle(const std::vector<double>& x) const;
        
        std::vector<double> getMean() const; // Returns the fitted mean
        std::vector<double> getStd() const;  // Returns the fitted standard deviation
};