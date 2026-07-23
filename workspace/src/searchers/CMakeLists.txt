set(SEARCHER_TARGETS
  search_ils_staged
  search_ils_staged_enhance
)

foreach(target ${SEARCHER_TARGETS})
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backend/includes
  )
  set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES native)
  set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_compile_options(${target} PRIVATE -fno-lto)
  target_link_options(${target} PRIVATE -fno-lto)
  target_link_libraries(${target} PRIVATE
    matmult_backend
    CUDA::cudart
    OpenMP::OpenMP_CXX
    Threads::Threads
    z
  )
endforeach()

