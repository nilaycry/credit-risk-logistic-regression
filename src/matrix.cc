#include "matrix.hpp"
#include <iostream>
#include <stdexcept>

Matrix::Matrix(int rows, int cols) {

  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("Invalid matrix dimensions");
  }

  this->rows_ = rows;
  this->cols_ = cols;
  this->data_.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("Cannot initialize Matrix with empty 2D vector");
    }

    this->rows_ = data.size();
    this->cols_ = data[0].size();
    this->data_ = data; // Directly copy the vector data
}

std::vector<double> Matrix::getRow(int row) const {

  if (row > rows_ || row < 0) {
    throw std::runtime_error("Row index out of bound");
  }
  
  return data_[row];
}

int Matrix::getRows() const { return this->rows_; }

int Matrix::getCols() const { return this->cols_; }

const std::vector<std::vector<double>>& Matrix::getData() const { return this->data_; }

double Matrix::get(int row, int col) const {

  if (row < 0 || col < 0) {
    throw std::runtime_error("Matrix index out of bound");
  }

  return this->data_[row][col];
}

void Matrix::set(int row, int col, double value) {

  if (row < 0 || col < 0) {
    throw std::runtime_error("Matrix index out of bound");
  }

  this->data_[row][col] = value;
}

void Matrix::print() const {

  for (size_t i = 0; i < data_.size(); i++) {

    for (size_t j = 0; j < data_[0].size(); j++) {
      std::cout << data_[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::add(const Matrix &other) const {
  // Matrix addition requires both matrices to have identical dimensions
  if (getRows() != other.getRows() || getCols() != other.getCols()) {
    throw std::runtime_error("Invalid operation");
  }

  Matrix output(getRows(), getCols());

  // Iterate over all elements and sum corresponding components
  for (size_t i = 0; i < data_.size(); i++) {
    for (size_t j = 0; j < data_[0].size(); j++) {
      output.set(i, j, get(i, j) + other.get(i, j));
    }
  }

  return output;
}

Matrix Matrix::scalarMultiply(double scalar) const {

  Matrix output(getRows(), getCols());

  // Multiply every element in the matrix by the scalar value
  for (size_t i = 0; i < data_.size(); i++) {
    for (size_t j = 0; j < data_[0].size(); j++) {
      output.set(i, j, scalar * get(i, j));
    }
  }
  return output;
}

Matrix Matrix::transpose() const {

  // Swaps dimensions: cols become rows, rows become cols
  Matrix output = Matrix(getCols(), getRows());

  // Copy element at (i, j) to (j, i)
  for (size_t i = 0; i < data_.size(); i++) {

    for (size_t j = 0; j < data_[0].size(); j++) {
      output.set(j, i, get(i, j));
    }
  }

  return output;
}

Matrix Matrix::multiply(const Matrix &other) const {

  // Matrix multiplication requires inner dimensions to match (Cols of A == Rows
  // of B)
  if (getCols() != other.getRows()) {
    throw std::runtime_error("Invalid matrix operation");
  }

  Matrix result(getRows(), other.getCols());

  // Iterate through rows of the first matrix (A)
  for (int i = 0; i < getRows(); i++) {

    // Iterate through columns of the second matrix (B)
    for (int j = 0; j < other.getCols(); j++) {

      double value = 0;

      // Compute the dot product of row i from A and column j from B
      for (int k = 0; k < getCols(); k++) {

        value += get(i, k) * other.get(k, j);
      }

      // Store the computed dot product in the result matrix
      result.set(i, j, value);
    }
  }

  return result;
}

Matrix Matrix::inverse() const {
  // Gauss-Jordan elimination requires a square matrix
  if (getCols() != getRows()) {
    throw std::runtime_error("Inverse does not exist");
  }

  // Step 1: Create an augmented matrix [A | I]
  Matrix augmented(getRows(), 2 * getCols());
  int size = getRows();

  // fill in the original matrix A on the left half
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      augmented.set(i, j, get(i, j));
    }
  }
  // set the diagonlas to 1.0 to add the identity matrix I on the right half
  for (int i = 0; i < size; i++) {
    augmented.set(i, size + i, 1.0);
  }

  // Step 2: Main Gauss-Jordan Elimination loop
  for (int i = 0; i < size; i++) {
    double curr_diagonal = augmented.get(i, i);

    // Step 2a: Pivoting - Swap rows if the current diagonal is zero
    if (curr_diagonal == 0.0) {
      // check to confirm if swap was made
      bool swap = false;
      for (int k = i + 1; k < size; k++) {
        if (augmented.get(k, i) != 0.0) {
          swap = true;
          // swap rows k and i
          for (int j = 0; j < 2 * size; j++) {
            double temp = augmented.get(k, j);
            augmented.set(k, j, augmented.get(i, j));
            augmented.set(i, j, temp);
          }
          break;
        }
      }
      // check if swap occurred, if not then throw an error
      if (!swap) {
        throw std::runtime_error(
            "Matrix is singular and the Inverse does not exist");
      }
    }

    curr_diagonal = augmented.get(i, i);

    // Step 2b: Normalize the pivot row so the diagonal becomes 1.0
    for (int j = 0; j < 2 * size; j++) {
      augmented.set(i, j, augmented.get(i, j) / curr_diagonal);
    }

    // Step 2c: Eliminate all other rows in the current column
    for (int k = 0; k < size; k++) {
      // dont change the pivot row itself
      if (k == i) {
        continue;
      }
      //

      double factor = augmented.get(k, i);
      //
      for (int j = 0; j < 2 * size; j++) {
        augmented.set(k, j, augmented.get(k, j) - factor * augmented.get(i, j));
      }
    }
  }

  // extracting and returning the final matrix
  Matrix inv(size, size);
  // copying the right half of the matrix
  for (int row = 0; row < size; row++) {
    for (int col = 0; col < size; col++) {
      inv.set(row, col, augmented.get(row, size + col));
    }
  }

  return inv;
}