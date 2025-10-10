# Run Cmake project for various iteration counts
#!/bin/bash
ITERATIONS=(100 500 1000)

# Build the project
cmake -S . -B build
cmake --build build --config Release

for i in "${ITERATIONS[@]}"
do
   echo "Running with $i iterations"
   ./build/Project $i
done