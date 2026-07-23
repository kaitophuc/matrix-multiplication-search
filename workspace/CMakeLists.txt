cmake_minimum_required(VERSION 3.28)

project(MatrixMultiplicationSearch VERSION 1.0 LANGUAGES CXX CUDA)

cmake_policy(SET CMP0048 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_compile_options(-O3)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fconcepts>
    $<$<COMPILE_LANGUAGE:CXX>:-fpermissive>
  )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas='--allow-expensive-optimizations=true --opt-level=3'")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread -O3 --dopt=on")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../../bin)

add_compile_definitions(EXECUTABLE_PATH="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

add_library(MapleCOMSPS STATIC IMPORTED GLOBAL)
set_target_properties(MapleCOMSPS PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/MapleCOMSPS.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/thirdparty/MapleCOMSPS_lib"
)

add_library(nauty STATIC IMPORTED GLOBAL)
set_target_properties(nauty PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/nauty.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/thirdparty/nauty_lib"
)

add_subdirectory(src)
