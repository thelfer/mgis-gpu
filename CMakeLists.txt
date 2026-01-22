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
if(NOT MGIS_FUNCTION_SUPPORT)
  message(FATAL_ERROR "TFEL support in MGIS is required")
endif(NOT MGIS_FUNCTION_SUPPORT)

add_subdirectory(elasticity)