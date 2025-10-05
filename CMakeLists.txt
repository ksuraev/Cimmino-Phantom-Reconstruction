# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(Project VERSION 1.0 LANGUAGES CXX)
project(NormalisationProfiler VERSION 1.0 LANGUAGES CXX)
project(AlgorithmTester VERSION 1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, RelWithDebInfo)" FORCE)
endif()

# Set the C++ standard (metal-cpp requires C++17 or later)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Add source files for main project
file(GLOB_RECURSE PROJECT_SOURCES ${CMAKE_SOURCE_DIR}/metal-src/*.cpp ${CMAKE_SOURCE_DIR}/metal-src/*.mm)
add_executable(Project ${PROJECT_SOURCES})

# Add source files for test normalisation project, excluding main.mm to avoid duplicate main function
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*metal-src/main.mm")
add_executable(NormalisationProfiler
    normalisation-profiler/main.mm
    ${PROJECT_SOURCES}
)

add_executable(AlgorithmTester
    algorithm-tester/main.mm
    ${PROJECT_SOURCES}
)

target_include_directories(Project PUBLIC ${CMAKE_SOURCE_DIR}/metal-include)
target_include_directories(NormalisationProfiler PUBLIC ${CMAKE_SOURCE_DIR}/metal-include)
target_include_directories(AlgorithmTester PUBLIC ${CMAKE_SOURCE_DIR}/metal-include)

# Find and include GLFW and Apple frameworks
find_package(glfw3 REQUIRED)
include_directories(/opt/homebrew/Cellar/glfw/3.4/include)
find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

# Find and link the required Apple frameworks for Metal
target_include_directories(Project
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp-library
)
target_include_directories(NormalisationProfiler
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp-library
)
target_include_directories(AlgorithmTester
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp-library
)

target_link_libraries(Project
  glfw
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)

target_link_libraries(NormalisationProfiler 
  glfw
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)
target_link_libraries(AlgorithmTester 
  glfw
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)


# Define PROJECT_BASE_PATH for use in the code
set(PROJECT_BASE_PATH "${CMAKE_SOURCE_DIR}")
target_compile_definitions(Project PRIVATE PROJECT_BASE_PATH="${PROJECT_BASE_PATH}")
target_compile_definitions(NormalisationProfiler PRIVATE PROJECT_BASE_PATH="${PROJECT_BASE_PATH}")
target_compile_definitions(AlgorithmTester PRIVATE PROJECT_BASE_PATH="${PROJECT_BASE_PATH}")

# Compile metal shaders to a metallib file
set(METAL_SHADER_SRC ${CMAKE_SOURCE_DIR}/metal-shaders/kernels.metal)
set(METAL_LIBRARY_OUTPUT ${CMAKE_BINARY_DIR}/metallibrary.metallib)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/metallibrary.ir
    COMMAND xcrun -sdk macosx metal -O2 -o ${CMAKE_BINARY_DIR}/metallibrary.ir -c ${METAL_SHADER_SRC}
    DEPENDS ${METAL_SHADER_SRC}
    COMMENT "Compiling Metal shader to AIR file"
)
add_custom_command(
    OUTPUT ${METAL_LIBRARY_OUTPUT}
    COMMAND xcrun -sdk macosx metallib -o ${METAL_LIBRARY_OUTPUT} ${CMAKE_BINARY_DIR}/metallibrary.ir
    DEPENDS ${CMAKE_BINARY_DIR}/metallibrary.ir
    COMMENT "Linking AIR file to Metal library"
)
add_custom_target(
    MetalLibrary ALL
    DEPENDS ${METAL_LIBRARY_OUTPUT}
    COMMENT "Building Metal library"
)
add_dependencies(Project MetalLibrary)
add_dependencies(NormalisationProfiler MetalLibrary)
add_dependencies(AlgorithmTester MetalLibrary)