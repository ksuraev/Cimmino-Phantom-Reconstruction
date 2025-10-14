# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(Project VERSION 1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_source_files_properties(${PROJECT_SOURCES} PROPERTIES LANGUAGE OBJCXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, RelWithDebInfo)" FORCE)
endif()

set(PROJECT_BASE_PATH "${CMAKE_SOURCE_DIR}")
set(SRC_DIR ${CMAKE_SOURCE_DIR}/metal-src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/metal-include)
set(METAL_LIB_DIR ${CMAKE_SOURCE_DIR}/metal-cpp-library)
set(SHADER_SRC ${CMAKE_SOURCE_DIR}/metal-shaders/kernels.metal)
set(METAL_LIB_OUTPUT ${CMAKE_BINARY_DIR}/metallibrary.metallib)

# Find and include GLFW and Apple frameworks
find_package(glfw3 REQUIRED)
find_library(FWK_FOUNDATION Foundation REQUIRED)
find_library(FWK_QUARTZ QuartzCore REQUIRED)
find_library(FWK_METAL Metal REQUIRED)

set(APPLE_FRAMEWORKS
  ${FWK_FOUNDATION}
  ${FWK_QUARTZ}
  ${FWK_METAL}
)

# Compile metal shaders to a metallib file
add_custom_command(
  OUTPUT ${METAL_LIB_OUTPUT}
  COMMAND xcrun -sdk macosx metal -c ${SHADER_SRC} -o ${CMAKE_BINARY_DIR}/metallibrary.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metallibrary.air -o ${METAL_LIB_OUTPUT}
  DEPENDS ${SHADER_SRC}
  COMMENT "Building Metal shader library"
)
add_custom_target(MetalLibrary ALL DEPENDS ${METAL_LIB_OUTPUT})

function(add_metal_target target_name)
  add_executable(${target_name} ${ARGN})
  target_include_directories(${target_name} PRIVATE ${INCLUDE_DIR} ${METAL_LIB_DIR})
  target_link_libraries(${target_name} PRIVATE glfw ${APPLE_FRAMEWORKS})
  target_compile_definitions(${target_name} PRIVATE PROJECT_BASE_PATH="${CMAKE_SOURCE_DIR}")
  add_dependencies(${target_name} MetalLibrary)
endfunction()


# Add source files for main project
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
  ${SRC_DIR}/*.cpp
  ${SRC_DIR}/*.mm
)

# Exclude algorithm-tester and normalisation-profiler classes from Project
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*metal-src/AlgorithmTester.mm")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*metal-src/NormalisationProfiler.mm")
add_metal_target(Project ${PROJECT_SOURCES})

# Add source files for normalisation profiler project
add_metal_target(NormalisationProfiler
  normalisation-profiler/main.mm
  ${SRC_DIR}/NormalisationProfiler.mm
  ${SRC_DIR}/MetalComputeEngine.mm
  ${SRC_DIR}/MetalUtilities.mm
  ${SRC_DIR}/Utilities.cpp
  ${SRC_DIR}/MTLImplementation.cpp
)

# Add source files for algorithm tester project
add_metal_target(AlgorithmTester
  algorithm-tester/main.mm
  ${SRC_DIR}/AlgorithmTester.mm
  ${SRC_DIR}/MetalComputeEngine.mm
  ${SRC_DIR}/MetalUtilities.mm
  ${SRC_DIR}/Utilities.cpp
  ${SRC_DIR}/MTLImplementation.cpp
)

target_compile_options(Project PRIVATE -Wno-elaborated-enum-base)
target_compile_options(NormalisationProfiler PRIVATE -Wno-elaborated-enum-base)
target_compile_options(AlgorithmTester PRIVATE -Wno-elaborated-enum-base)

source_group(TREE ${SRC_DIR} FILES ${PROJECT_SOURCES})