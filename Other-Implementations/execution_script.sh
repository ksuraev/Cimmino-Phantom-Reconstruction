#!/bin/bash

# Script to compile and run sequential.cpp and openmp.cpp multiple times with varying iteration counts 
# To run this script from the Other-Implementations directory, you must export the PROJECT_BASE_PATH variable
# Example: export PROJECT_BASE_PATH=/path/to/your/Other-Implementations

# Compilation
clang++ -o sequential_exec sequential/sequential.cpp utilities/utilities.cpp
g++-15 -fopenmp -o openmp_exec openmp/openmp.cpp utilities/utilities.cpp

# Array of iteration counts
iteration_counts=(100 500 1000)

# Run each program 3 times for each iteration count - the iteration count is passed as a command line argument
for iterations in "${iteration_counts[@]}"; do
    echo "Running sequential.cpp with $iterations iterations"
    for i in {1..3}; do
        echo "Run $i:"
        ./sequential_exec $iterations
    done

    echo "Running openmp.cpp with $iterations iterations"
    for i in {1..3}; do
        echo "Run $i:"
        ./openmp_exec $iterations
    done
done