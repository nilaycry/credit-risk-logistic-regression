#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>
#include <tuple>

/**
 * @brief Utility class for parsing financial CSV datasets.
 * 
 * Handles reading messy real-world CSV files, extracting numeric features 
 * from strings, and converting them into the 2D Matrix format required by 
 * the logistic regression engine.
 */
class DataLoader {
    private:
        std::string filename_; // Path to the CSV file

    public:
        /**
         * @brief Constructs a DataLoader to read from the specified file.
         * @param filename The path to the CSV data file.
         */
        DataLoader(const std::string& filename);

        /**
         * @brief Reads the CSV and parses it into Features (X) and Labels (y).
         * 
         * Expects a CSV where the first row acts as a header. It will drop 
         * non-numeric columns like "Company_ID" and pull all ratio columns 
         * into a Matrix, storing the final column as the target labels.
         * 
         * @param has_header If true, skips the first line of the file.
         * @return A std::tuple containing: 
         *         - The X feature Matrix
         *         - The y label vector (1 for Default, 0 for Survive)
         */
        std::tuple<Matrix, std::vector<double>> loadData(bool has_header = true) const;

        // In data_loader.h
        static std::tuple<Matrix, Matrix, std::vector<double>, std::vector<double>> 
            train_test_split(const Matrix& X, const std::vector<double>& y, double train_ratio);

};
