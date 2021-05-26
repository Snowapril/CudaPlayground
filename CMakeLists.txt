# CMake version
cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

# Declare project
project(CudaPlayground LANGUAGES CXX CUDA)

# Check cuda support
enable_language("CUDA")
include(CheckLanguage)
check_language(CUDA)

# Set common cuda architectures over this project
set(CMAKE_CUDA_ARCHITECTURES "35;50;72")

# include & link useful CudaToolkit Utils
find_package(CUDAToolkit REQUIRED)
find_file(CUDA_THRUST_VERSION_H version.h PATHS "${CUDAToolkit_INCLUDE_DIRS}/thrust" REQUIRED)
get_filename_component(CUDA_THRUST_THRUST_DIR "${CUDA_THRUST_VERSION_H}" DIRECTORY)
get_filename_component(CUDA_THRUST_INCLUDE_DIR "${CUDA_THRUST_THRUST_DIR}" DIRECTORY)
add_library(CUDA::thrust INTERFACE IMPORTED)
add_library(CUDA::cublas_static INTERFACE IMPORTED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUDA_THRUST_INCLUDE_DIR})

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Includes
include_directories(Examples)
include_directories(Libraries)
include_directories(Libraries/doctest)
include_directories(Libraries/cxxopts)

# Set enable output of compile commands during generation   
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type - Release by default
message("CMake build type: " ${CMAKE_BUILD_TYPE})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
endif()

# CUDA Standard definition
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Overrides
set(CMAKE_MACOSX_RPATH ON)

# Set resources
set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Resources)
add_subdirectory(Examples/Introduction)
add_subdirectory(Examples/RayTracer)