#include "normalizer.hpp"
#include "matrix.hpp"
#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>


Normalizer::Normalizer(){

}

std::vector<double> Normalizer::getMean() const{
    return means_;
}

std::vector<double> Normalizer::getStd() const{
    return stds_;
}

void Normalizer::fit(const Matrix& x) {

    if (x.getRows() == 0 || x.getCols() == 0) {
        throw std::runtime_error("Cannot fit without data");
    }

    // Step 1: Compute the Mean for each Feature (Column)
    // Outer loop iterates through each column (i.e. 'Current Ratio', 'Debt', etc.)
    for (int i = 0; i < x.getCols(); i++) {
        double curr_mean = 0.0;

        // Inner loop iterates down the rows to sum every company's value for this specific column
        for (int j = 0; j < x.getRows(); j++) {
            curr_mean += x.get(j, i); // x.get(row, col)
        }

        // Divide sum by total rows (companies) to get the Feature Mean
        curr_mean = curr_mean / static_cast<double>(x.getRows());
        means_.push_back(curr_mean);
    }

    // Step 2: Compute the Standard Deviation for each Feature (Column)
    for (int i = 0; i < x.getCols(); i++) {
        double curr_std = 0.0;

        // Inner loop to sum the squared differences from the column's mean
        for (int j = 0; j < x.getRows(); j++) {
            curr_std += std::pow(x.get(j, i) - means_[i], 2);
        }

        // Divide by N (rows) to get the Variance, then take sqrt for Standard Deviation
        curr_std = curr_std / static_cast<double>(x.getRows());
        curr_std = std::sqrt(curr_std);
        stds_.push_back(curr_std);
    }


}

Matrix Normalizer::transform(const Matrix& x) const {

    // Prevent division by zero if any feature has exactly 0 variance (all values are identical)
    for (size_t i = 0; i < stds_.size(); i++) {
        if (stds_[i] == 0) {
            throw std::runtime_error("Invalid division by 0: A feature has zero standard deviation.");
        }
    }

    // Initialize the output matrix with the exact same dimensions as the input
    Matrix norm(x.getRows(), x.getCols());

    // Iterate over every element in the matrix
    for (int i = 0; i < norm.getRows(); i++) {
        for (int j = 0; j < norm.getCols(); j++) {
            // Apply Z-score formula using the corresponding column's mean and std dev
            norm.set(i, j, (x.get(i, j) - means_[j]) / stds_[j]);
        }
    }

    return norm;
}

std::vector<double> Normalizer::transformSingle(const std::vector<double>& x) const {
    std::vector<double> norm;

    // Apply Z-score formula to a single 1D vector (e.g. one company's features)
    // We iterate through each feature i, and use the pre-computed means_[i] and stds_[i]
    for (size_t i = 0; i < x.size(); i++) {
        norm.push_back((x[i] - means_[i]) / stds_[i]);
    }
    
    return norm;
}