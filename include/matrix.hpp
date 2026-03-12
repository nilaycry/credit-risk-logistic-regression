#pragma once
#include <vector>

/**
 * @brief A simple 2D Matrix class for linear algebra operations.
 */
class Matrix {

private:
  int rows_; // Number of rows in the matrix
  int cols_; // Number of columns in the matrix
  std::vector<std::vector<double>>
      data_; // Internal 2D vector storing matrix elements

public:
  /**
   * @brief Constructs a Matrix with given dimensions, initialized to zero.
   * @param rows Dimensions for rows
   * @param cols Dimensions for columns
   */
  Matrix(int rows, int cols);

  /**
   * @brief Constructs a Matrix directly from a 2D vector.
   * @param data The 2D vector representing the matrix data.
   */
  Matrix(const std::vector<std::vector<double>>& data);

  int getRows() const; // Returns the number of rows
  int getCols() const; // Returns the number of columns
  const std::vector<std::vector<double>>& getData() const; // Returns raw data

  /**
   * @brief Retrieves the value at the specified row and column.
   * @param row Row index (0-based)
   * @param col Column index (0-based)
   * @return The value at the given location
   */
  double get(int row, int col) const;

  /**
   * @brief Retrieves an entire row as a standard vector.
   * @param row Row index (0-based)
   * @return A vector depicting the entire row.
   */
  std::vector<double> getRow(int row) const;

  /**
   * @brief Sets the value at the specified row and column.
   * @param row Row index (0-based)
   * @param col Column index (0-based)
   * @param value The value to set
   */
  void set(int row, int col, double value);

  /**
   * @brief Prints the matrix to standard output.
   */
  void print() const;

  /**
   * @brief Adds another matrix to this matrix.
   * @param other The matrix to add
   * @return A new Matrix representing the sum
   */
  Matrix add(const Matrix &other) const;

  /**
   * @brief Computes the transpose of this matrix.
   * @return A new Matrix representing the transpose
   */
  Matrix transpose() const;

  /**
   * @brief Multiplies all elements by a scalar value.
   * @param scalar The scalar multiplier
   * @return A new Matrix representing the scaled result
   */
  Matrix scalarMultiply(double scalar) const;

  /**
   * @brief Multiplies this matrix by another matrix (matrix multiplication).
   * @param other The right-hand side matrix
   * @return A new Matrix representing the product
   */
  Matrix multiply(const Matrix &other) const;

  /**
   * @brief Computes the inverse of the matrix using Gauss-Jordan elimination.
   *        This requires augmenting the matrix with an identity matrix and
   *        performing row-reduction operations.
   * @return A new Matrix representing the inverse.
   * @throws std::runtime_error if the matrix is not square or is singular.
   */
  Matrix inverse() const;
};