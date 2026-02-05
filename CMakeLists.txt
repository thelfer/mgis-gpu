cmake_minimum_required(VERSION 3.12)
project(mgis-gpu
  LANGUAGES CXX
  )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Threads REQUIRED)
find_package(MFrontGenericInterface REQUIRED)
if(NOT MGIS_FUNCTION_SUPPORT)
  message(FATAL_ERROR "MGIS/Function is required")
endif(NOT MGIS_FUNCTION_SUPPORT)
if(NOT MGIS_HAVE_TFEL)
  message(FATAL_ERROR "TFEL support in MGIS is required")
endif(NOT MGIS_HAVE_TFEL)

# CUDA support
option(enable-cuda-kernels "produce kernels based on the CUDA programming model" OFF)
if(enable-cuda-kernels)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif(enable-cuda-kernels)

# testing
set(CTEST_CONFIGURATION_TYPE "${JOB_BUILD_CONFIGURATION}")
# (must be placed *before* any add_subdirectory, cmake bug ?)
enable_testing()

# add subdirectory
add_subdirectory(src)
add_subdirectory(elasticity)
add_subdirectory(hyperelasticity)
add_subdirectory(plasticity)
add_subdirectory(unit-tests)
