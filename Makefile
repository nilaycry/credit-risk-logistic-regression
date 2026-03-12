CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cc)
OBJS = $(SRCS:$(SRC_DIR)/%.cc=$(BIN_DIR)/%.o)

# Executable name
TARGET = $(BIN_DIR)/credit_risk_app

# Default target
all: $(TARGET)

# Compile object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cc | directories
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the executable
$(TARGET): $(OBJS) | directories
	$(CXX) $(CXXFLAGS) $^ -o $@

# Create directories if they don't exist
directories:
	mkdir -p $(BIN_DIR)

# Clean build files
clean:
	rm -rf $(BIN_DIR)/*.o $(TARGET)

.PHONY: all clean directories
