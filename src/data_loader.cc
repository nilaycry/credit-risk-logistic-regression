#include "data_loader.h"
#include "matrix.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>



DataLoader::DataLoader(const std::string& filename): filename_(filename) {};

std::tuple<Matrix, std::vector<double>> DataLoader::loadData(bool has_header) const {

    std::ifstream ifs{filename_};

    if (!ifs.is_open()) {
       throw std::runtime_error("Could not open file");
    }

    if (has_header) {
        std::string discard;
        std::getline(ifs, discard);
    }

    std::string row("");
    std::vector<double> y_labels;
    std::vector<std::vector<double>> temp_matrix;
    

    size_t expected_cols = 0; // Tracks the expected number of tokens per row

    while (std::getline(ifs, row)) {
        std::vector<double> current_row;
        std::string token;
        std::stringstream ss(row);
        std::vector<std::string> temp_string;
        
        // Extract all tokens separated by commas
        while (std::getline(ss, token, ',')) {
            temp_string.push_back(token);            
        }

        // --- Robust Parsing Features --- 
        // 1. Establish the "correct" number of columns based on the very first row
        if (expected_cols == 0) {
            expected_cols = temp_string.size();
        } 
        // 2. If a subsequent row has an irregular number of commas, skip it entirely
        else if (temp_string.size() != expected_cols) {
            continue; 
        }

        // 3. Explicitly scan for "NaN" or empty values before C++ converts them
        bool contains_nan = false;
        for (const std::string& str : temp_string) {
            if (str == "" || str == "NaN" || str == "nan" || str == "NA" || str == " NaN") {
                contains_nan = true;
                break;
            }
        }
        if (contains_nan) {
            continue; // Skip the row entirely if any feature is missing
        }

        // 4. Prevent std::stod() crashes from other bad text (e.g. "corrupted_data")
        try {
            // Extract the Features (X), starting at index 1 to skip CompanyID
            for (size_t i = 1; i < temp_string.size() - 1; i++) {
                current_row.push_back(std::stod(temp_string[i]));
            }

            // Extract the Label (y)
            y_labels.push_back(std::stod(temp_string.back()));

            // Only push the row if stod() didn't fail
            temp_matrix.push_back(current_row);
            
        } catch (const std::exception& e) {
            // We caught a bad value! Skip this row and continue reading the CSV
            continue;
        }
    }   

    Matrix X(temp_matrix);

    std::tuple<Matrix, std::vector<double>> data_ld = {X, y_labels};

    return data_ld;

}

/**
 * @brief Splits a dataset into training and testing sets.
 * 
 * Uses the provided ratio to slice the feature Matrix (X) and label vector (y) 
 * into two distinct partitions to prevent data leakage during model evaluation.
 * 
 * @param X The full feature matrix
 * @param y The full label vector
 * @param train_ratio The proportion of data to use for training (e.g., 0.8)
 * @return A tuple containing {X_train, X_test, y_train, y_test}
 */
std::tuple<Matrix, Matrix, std::vector<double>, std::vector<double>> 
        DataLoader::train_test_split(const Matrix& X, const std::vector<double>& y, double train_ratio) {

    std::vector<int> indices(X.getRows());

    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<double>> x_test;
    std::vector<std::vector<double>> x_train;
    std::vector<double> y_train;
    std::vector<double> y_test;

    int cutoff = static_cast<int>( train_ratio * X.getRows());

    for (int i = 0; i < cutoff; i++) {
        x_train.push_back(X.getRow(indices[i]));
        y_train.push_back(y[indices[i]]);
    }

    for(int j = cutoff; j < X.getRows(); j++) {
        x_test.push_back(X.getRow(indices[j]));
        y_test.push_back(y[indices[j]]); 
    }

    Matrix X_test(x_test);
    Matrix X_train(x_train);

    return {X_train, X_test, y_train, y_test};
}