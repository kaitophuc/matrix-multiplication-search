#pragma once

/*
 * Enhanced TDM Algorithm Header
 *
 * Author: Minh Phuc Nguyen, Matt Anderson
 *
 * Workflow Summary:
 * This header file defines the structures and functions required for the enhanced TDM
 * (Three-Dimensional Matrix) algorithm. The enhanced version is optimized for GPU acceleration and
 * efficient memory management.
 *
 * Key Features:
 * - **GpuResources Struct**: Centralized memory management for GPU resources, including pinned host
 * memory and device memory.
 * - **CUDA Integration**: Provides CUDA kernel declarations and utility functions for GPU
 * operations.
 *
 * Usage:
 * - Initialize the `GpuResources` struct to allocate and manage GPU memory.
 * - Use the `make_tests` or `make_tests_enhance` functions to prepare input data for the TDM
 * algorithm.
 * - Call the `outer_Kernel` function to execute the TDM algorithm on the GPU.
 * - Ensure proper cleanup of GPU resources by destroying streams and freeing memory in the
 * `GpuResources` destructor.
 *
 * Technical Details:
 * - **Pinned Memory**: Allocates pinned host memory for faster data transfer between CPU and GPU.
 * - **CUDA Streams**: Uses CUDA streams for asynchronous memory operations and kernel execution.
 * - **Memory Layout**: Organizes device memory into contiguous buffers for efficient access.
 * - **Scalability**: Supports multiple trials and large problem sizes with dynamic memory
 * allocation.
 *
 * This file contains AI-generated code. Be careful when editing to ensure correctness.
 */

#include "Puz.h"
#include <future>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <cstdint>
#include <sys/resource.h>  // For getrusage (memory usage)

#include <vector>
#include "cuda_runtime.h"
#include <iostream>
#include <algorithm>
#include <TDM.h>
#include <Puz.h>
#include <chrono>
#include <fstream>
#include <cooperative_groups.h>

#include <unistd.h>

// Forward declarations
#define TRIALS 256
using e_type = unsigned long long;  // Assuming e_type for clarity

struct GpuResources;

void outer_Kernel(GpuResources& gpu_resources, int s, int k, int trials = TRIALS);
void outer_Kernel(GpuResources& gpu_resources, const e_type* puz_pinned_buffer,
                  int* results_pinned_buffer, int s, int k, int trials = TRIALS);
void outer_KernelDeviceInput(GpuResources& gpu_resources, int* results_pinned_buffer,
                             int s, int k, int trials = TRIALS);
bool check_size(vector<Puz>& puzzles);

__device__ inline void enhanced_check_cuda_error(const char* c = "default") {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s\n", c);
    printf("Cuda Error: %s \n", cudaGetErrorString(err));
    return;
  }
}

__host__ inline void enhanced_Hcheck_cuda_error(string c = "default") {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cout << c << endl;
    printf("Cuda Error: %s \n", cudaGetErrorString(err));
    exit(-1);
  }
}

// A struct to hold the shared GPU resources (single buffer)
struct GpuResources {
  int device_id;
  int max_trials;

  cudaStream_t stream;
  e_type* puz_pinned_buffer;
  int* results_pinned_buffer;

  char* dev_buf;
  size_t size_buf;

  e_type* dev_Puz;
  uint8_t* dev_TDM;
  int* dev_result;
  int* dev_no_change;
  int* dev_ranges;
  int* dev_active_count;
  int* dev_changed;
  uint8_t* dev_projection;

  size_t size_TDM, size_Puz, size_result, size_no_change, size_ranges, size_active_count, size_changed,
      size_projection, size;

  GpuResources(int dev_id, int s, int k, int trials) : device_id(dev_id), max_trials(trials) {
    cudaSetDevice(device_id);
    enhanced_Hcheck_cuda_error("Setting CUDA device");

    unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);

    size_TDM = sizeof(uint8_t) * s * s * s * trials;
    size_Puz = sizeof(e_type) * s * entries_per_row * trials;
    size_result = sizeof(int) * trials;
    size_no_change = sizeof(int) * trials;
    size_ranges = sizeof(int) * s * trials;
    size_active_count = sizeof(int);
    size_changed = sizeof(int) * trials;
    size_projection = sizeof(uint8_t) * s * s * trials;

    size_TDM = (size_TDM + 7) & ~7;
    size_Puz = (size_Puz + 7) & ~7;
    size_result = (size_result + 7) & ~7;
    size_no_change = (size_no_change + 7) & ~7;
    size_ranges = (size_ranges + 7) & ~7;
    size_active_count = (size_active_count + 7) & ~7;
    size_changed = (size_changed + 7) & ~7;
    size_projection = (size_projection + 7) & ~7;

    size = size_Puz + size_TDM + size_result + size_no_change + size_ranges + size_active_count +
           size_changed + size_projection;
    size_t offset = 0;

    size_buf = size;

    // Create stream
    cudaStreamCreate(&stream);
    enhanced_Hcheck_cuda_error("Creating CUDA stream");

    // Allocate pinned host memory
    cudaMallocHost(&puz_pinned_buffer, size_Puz);
    cudaMallocHost(&results_pinned_buffer, size_result);
    enhanced_Hcheck_cuda_error("Allocating pinned host memory");

    // Allocate device memory
    cudaMalloc(&dev_buf, size);
    enhanced_Hcheck_cuda_error("Allocating dev_buf");

    dev_Puz = (e_type*) (dev_buf + offset);
    offset += size_Puz;
    dev_result = (int*) (dev_buf + offset);
    offset += size_result;
    dev_no_change = (int*) (dev_buf + offset);
    offset += size_no_change;
    dev_ranges = (int*) (dev_buf + offset);
    offset += size_ranges;
    dev_active_count = (int*) (dev_buf + offset);
    offset += size_active_count;
    dev_TDM = (uint8_t*) (dev_buf + offset);
    offset += size_TDM;
    dev_changed = (int*) (dev_buf + offset);
    offset += size_changed;
    dev_projection = (uint8_t*) (dev_buf + offset);
    offset += size_projection;

    cudaMemsetAsync(dev_buf, 0, size_buf, stream);
    enhanced_Hcheck_cuda_error("Memset dev_buf");
  }

  ~GpuResources() {
    cudaSetDevice(device_id);

    cudaStreamDestroy(stream);
    cudaFreeHost(puz_pinned_buffer);
    cudaFreeHost(results_pinned_buffer);
    cudaFree(dev_buf);

    enhanced_Hcheck_cuda_error("Freeing resources");
  }

  void make_tests(int s, int k, int trials) {
    unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
    memset(puz_pinned_buffer, 0, sizeof(e_type) * s * entries_per_row * trials);
    memset(results_pinned_buffer, 0, sizeof(int) * trials);
    for (int test = 0; test < trials; ++test) {
      Puz p(s, k);
      p.setRandom();
      for (int i = 0; i < s * entries_per_row; i++) {
        puz_pinned_buffer[test * s * entries_per_row + i] = p.getData(i);
      }
    }
  }

  void make_tests_enhance(vector<Puz>& puzzles, int trials) {
    unsigned int s = puzzles[0].getHeight();
    unsigned int k = puzzles[0].getWidth();
    unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
    memset(puz_pinned_buffer, 0, sizeof(e_type) * s * entries_per_row * trials);
    memset(results_pinned_buffer, 0, sizeof(int) * trials);
    for (int test = 0; test < trials; ++test) {
      Puz& p = puzzles[test];
      for (int i = 0; i < s * entries_per_row; i++) {
        puz_pinned_buffer[test * s * entries_per_row + i] = p.getData(i);
      }
    }
  }
};

