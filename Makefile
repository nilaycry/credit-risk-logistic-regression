CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin

# Source files (excluding train_regularized.cc from default build)
SRCS = $(filter-out $(SRC_DIR)/train_regularized.cc, $(wildcard $(SRC_DIR)/*.cc))
OBJS = $(SRCS:$(SRC_DIR)/%.cc=$(BIN_DIR)/%.o)

# Executable name
TARGET = $(BIN_DIR)/credit_risk_app

# Regularized experiment executable
REG_SRCS = $(SRC_DIR)/train_regularized.cc $(SRC_DIR)/data_loader.cc $(SRC_DIR)/matrix.cc $(SRC_DIR)/normalizer.cc $(SRC_DIR)/logistic_regression.cc
REG_OBJS = $(REG_SRCS:$(SRC_DIR)/%.cc=$(BIN_DIR)/%.o)
REG_TARGET = $(BIN_DIR)/credit_risk_regularized

# Default target
all: $(TARGET)

# Compile object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cc | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the main executable
$(TARGET): $(OBJS) | directories
	$(CXX) $(CXXFLAGS) $^ -o $@

# Link the regularized experiment executable
$(REG_TARGET): $(REG_OBJS) | directories
	$(CXX) $(CXXFLAGS) $^ -o $@

# Create directories if they don't exist
directories:
	mkdir -p $(BIN_DIR)

# Build regularized experiment
regularized: $(REG_TARGET)

# Clean build files
clean:
	rm -rf $(BIN_DIR)/*.o $(TARGET) $(REG_TARGET)

.PHONY: all clean directories regularized
