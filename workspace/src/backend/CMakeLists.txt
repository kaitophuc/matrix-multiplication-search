set(MATMULT_BACKEND_SOURCES
  3DM_to_SAT.cpp
  GpuTask.cpp
  Permutation.cpp
  Puz.cpp
  TDM.cpp
  TDM_cuda.cu
  TDM_cuda_enhanced.cu
  bloom_filtering.cpp
  canonization.cpp
  checker.cpp
  fitness.cpp
  heuristic.cpp
  ils_stage.cpp
  matching.cpp
  prg.cpp
  search.cpp
  set.cpp
  special.cpp
  stats.cpp
)

add_library(matmult_backend STATIC ${MATMULT_BACKEND_SOURCES})

target_include_directories(matmult_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/includes
  ${CMAKE_SOURCE_DIR}/thirdparty/MapleCOMSPS_lib
  ${CMAKE_SOURCE_DIR}/thirdparty/nauty_lib
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(matmult_backend PUBLIC
  MapleCOMSPS
  nauty
  CUDA::cudart
  cudadevrt
  OpenMP::OpenMP_CXX
  Threads::Threads
)

set_property(TARGET matmult_backend PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET matmult_backend PROPERTY CUDA_STANDARD 17)
set_property(TARGET matmult_backend PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET matmult_backend PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_options(matmult_backend PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>)
target_link_options(matmult_backend PRIVATE -fno-lto)
