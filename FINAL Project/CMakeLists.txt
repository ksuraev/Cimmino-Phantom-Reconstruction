# CMakeLists.txt

# Minimum CMake version and project name
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
project(Project)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug, Release, etc.)" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, etc.)" FORCE)
endif()

# Set the C++ standard (metal-cpp requires C++17 or later)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")


# Tell CMake where to find the metal-cpp headers
include_directories(metal-cpp)

file(GLOB_RECURSE PROJECT_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.mm)
add_executable(Project ${PROJECT_SOURCES})

# Find and include GLFW and Apple frameworks
find_package(glfw3 REQUIRED)
include_directories(/opt/homebrew/Cellar/glfw/3.4/include)
find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)

target_include_directories(Project PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Find and link the required Apple frameworks for Metal
target_include_directories(Project
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
  PRIVATE ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(Project
  glfw
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_QUARTZ_CORE}
  ${APPLE_FWK_METAL}
)

target_compile_definitions(Project PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_compile_definitions(Project PRIVATE $<$<CONFIG:Release>:NDEBUG>)

target_compile_options(Project PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_options(Project PRIVATE ${MPI_CXX_LINK_FLAGS})


set(METAL_SHADER_SRC ${CMAKE_SOURCE_DIR}/include/kernels.metal)
set(METAL_LIBRARY_OUTPUT ${CMAKE_BINARY_DIR}/metallibrary.metallib)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/metallibrary.ir
    COMMAND xcrun -sdk macosx metal -o ${CMAKE_BINARY_DIR}/metallibrary.ir -c ${METAL_SHADER_SRC}
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