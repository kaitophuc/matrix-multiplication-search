// TDM_cuda_enhanced.cu
/**
 * CUDA Enhanced TDM Algorithm Implementation
 *
 * Author: Minh Phuc Nguyen, Matt Anderson
 *
 * Workflow Summary:
 * This file contains the CUDA implementation of the enhanced TDM (Three-Dimensional Matrix)
 * algorithm. The enhanced version is optimized for running multiple TDM computations on a GPU,
 * leveraging CUDA streams and double buffering for efficient memory management and parallelism.
 *
 * Key Features:
 * - **GPU Acceleration**: Utilizes CUDA kernels to perform TDM simplification and computation.
 * - **Double Buffering**: Implements pinned memory buffers for overlapping data transfer and
 * computation.
 * - **Optimized for Small k**: Focuses on k < 16 for faster witness computation, with a general
 * implementation for larger k.
 * - **Cooperative Groups**: Uses cooperative groups for efficient thread synchronization within
 * CUDA kernels.
 * - **Bit Manipulation**: Employs optimized bit manipulation techniques for TDM operations.
 *
 * Usage:
 * - Call the `outer_Kernel` function to execute the TDM algorithm on the GPU.
 * - Pass the array of puzzles, results, and iterations arrays to the function. The results and
 * iterations arrays will be populated with the output of the TDM algorithm.
 * - Ensure that the input puzzles are properly formatted and that the GPU resources are initialized
 * using the `GpuResources` class.
 *
 * Technical Details:
 * - **CUDA Kernels**: Includes multiple kernels for projection calculation, forward/backward
 * reachability, TDM modification, and result summation.
 * - **Memory Management**: Manages device and host memory using pinned buffers and asynchronous
 * memory transfers.
 * - **Parallelism**: Exploits thread-level and block-level parallelism for high performance.
 */

#include "matching.h"
#include "TDM_cuda_enhanced.h"
#include <algorithm>
#include <cstdlib>
#include <cstdint>

#if __has_include(<cuda/std/span>)
#include <cuda/std/span>
// Work around CUDA 13.2 + GCC 13 ambiguity between std::data and cuda::std::data
// for cuda::std::span when CUB uses unqualified data/size calls.
namespace cuda {
namespace std {
template <class T, size_t Extent>
__host__ __device__ constexpr auto data(span<T, Extent> s) noexcept -> decltype(s.data()) {
  return s.data();
}

template <class T, size_t Extent>
__host__ __device__ constexpr auto size(span<T, Extent> s) noexcept -> decltype(s.size()) {
  return s.size();
}
}  // namespace std
}  // namespace cuda
#endif

#include <cub/cub.cuh>

namespace cg = cooperative_groups;

#define HASH_CONST 1023
#define MAX_S 500
#define TDM_THREADS 256

__host__ __forceinline__ int gcd_host_int(int a, int b) {
  a = (a < 0) ? -a : a;
  b = (b < 0) ? -b : b;
  while (b != 0) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return (a == 0) ? 1 : a;
}

__host__ bool check_size(std::vector<Puz>& puzzles) {
  int s = puzzles[0].getHeight();
  int k = puzzles[0].getWidth();

  for (int i = 1; i < puzzles.size(); i++) {
    if (puzzles[i].getHeight() != s || puzzles[i].getWidth() != k) {
      return false;
    }
  }

  return true;
}

__global__ void calc_projection_Kernel(int s, const uint8_t* __restrict__ dev_tdm,
                                       uint8_t* __restrict__ dev_projection,
                                       int* __restrict__ no_change, int face, int trials) {
  const size_t N = (size_t) s * s * s;
  const size_t P = (size_t) s * s;

  int trial = (int) blockIdx.y;
  if (trial >= trials || no_change[trial] >= 3)
    return;

  size_t p_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (p_idx >= P)
    return;

  uint8_t out = 0;

  if (face == 0) {
    int y = (int) (p_idx / (size_t) s);
    int z = (int) (p_idx - (size_t) y * (size_t) s);
    size_t base = (size_t) trial * N + (size_t) y * (size_t) s + (size_t) z;
    for (int x = 0; x < s; ++x) {
      if (dev_tdm[base + (size_t) x * (size_t) s * (size_t) s]) {
        out = 1;
        break;
      }
    }
  } else if (face == 1) {
    int x = (int) (p_idx / (size_t) s);
    int z = (int) (p_idx - (size_t) x * (size_t) s);
    size_t base = (size_t) trial * N + (size_t) x * (size_t) s * (size_t) s + (size_t) z;
    for (int y = 0; y < s; ++y) {
      if (dev_tdm[base + (size_t) y * (size_t) s]) {
        out = 1;
        break;
      }
    }
  } else {  // face == 2
    int x = (int) (p_idx / (size_t) s);
    int y = (int) (p_idx - (size_t) x * (size_t) s);
    size_t base =
        (size_t) trial * N + (size_t) x * (size_t) s * (size_t) s + (size_t) y * (size_t) s;
    const uint8_t* __restrict__ p = dev_tdm + base;

    int z = 0;
    for (; z + 4 <= s; z += 4) {
      out |= (uint8_t) (p[z] | p[z + 1] | p[z + 2] | p[z + 3]);
      if (out)
        break;
    }
    for (; z < s && !out; ++z)
      out |= p[z];
  }

  dev_projection[(size_t) trial * P + p_idx] = out;
}

/* Boolean Transitive Closure Kernels */

__global__ void fb_kern_TC_warp_Kernel(int s, const uint8_t* __restrict__ d_projection,
                                       int* __restrict__ d_ranges,
                                       const int* __restrict__ no_change, int trials) {
  int trial = blockIdx.x * blockDim.y + threadIdx.y;
  if (trial >= trials)
    return;
  if (no_change[trial] >= 3)
    return;

  int u = threadIdx.x;  // 0 to 31
  uint32_t my_row = 0;

  if (u < s) {
    const uint8_t* p = d_projection + (size_t) trial * s * s;
    for (int v = 0; v < s; ++v) {
      if (p[u * s + v])
        my_row |= (1u << v);
    }
    my_row |= (1u << u);  // Every node can reach itself
  }

  // Floyd-Warshall Transitive Closure via warp intrinsics
  for (int k = 0; k < s; ++k) {
    uint32_t k_row = __shfl_sync(0xFFFFFFFF, my_row, k);
    if (my_row & (1u << k)) {
      my_row |= k_row;
    }
  }

  // Find SCC: u and v are in the same SCC if u reaches v and v reaches u
  uint32_t my_tc_col = 0;
  for (int v = 0; v < s; ++v) {
    uint32_t v_row = __shfl_sync(0xFFFFFFFF, my_row, v);
    if (v_row & (1u << u)) {
      my_tc_col |= (1u << v);
    }
  }

  // SCC ID is the minimum node index in the component
  if (u < s) {
    uint32_t my_scc = my_row & my_tc_col;
    d_ranges[trial * s + u] = __ffs(my_scc) - 1;
  }
}

__global__ void fb_kern_TC_block_Kernel(int s, const uint8_t* __restrict__ d_projection,
                                        int* __restrict__ d_ranges,
                                        const int* __restrict__ no_change) {
  int trial = blockIdx.x;
  if (no_change[trial] >= 3)
    return;

  int words_per_row = (s + 31) / 32;
  extern __shared__ uint32_t shared_tc[];  // Size: s * words_per_row

  // Initialize shared_tc to 0
  for (int i = threadIdx.x; i < s * words_per_row; i += blockDim.x) {
    shared_tc[i] = 0;
  }
  __syncthreads();

  // Load projection into bits
  for (int i = threadIdx.x; i < s * s; i += blockDim.x) {
    if (d_projection[(size_t) trial * s * s + i]) {
      int row = i / s;
      int col = i % s;
      atomicOr(&shared_tc[row * words_per_row + col / 32], 1u << (col % 32));
    }
  }
  __syncthreads();

  // Add self-reachability
  for (int u = threadIdx.x; u < s; u += blockDim.x) {
    atomicOr(&shared_tc[u * words_per_row + u / 32], 1u << (u % 32));
  }
  __syncthreads();

  // Floyd-Warshall
  for (int k = 0; k < s; ++k) {
    int total_words = s * words_per_row;
    for (int idx = threadIdx.x; idx < total_words; idx += blockDim.x) {
      int i = idx / words_per_row;
      int w = idx % words_per_row;

      uint32_t i_word_k = shared_tc[i * words_per_row + k / 32];
      if (i_word_k & (1u << (k % 32))) {
        uint32_t k_val = shared_tc[k * words_per_row + w];
        if (k_val) {
          shared_tc[idx] |= k_val;
        }
      }
    }
    __syncthreads();
  }

  // Assign ranges
  for (int u = threadIdx.x; u < s; u += blockDim.x) {
    int min_v = u;
    for (int v = 0; v < u; ++v) {
      uint32_t u_reaches_v = shared_tc[u * words_per_row + v / 32] & (1u << (v % 32));
      if (u_reaches_v) {
        uint32_t v_reaches_u = shared_tc[v * words_per_row + u / 32] & (1u << (u % 32));
        if (v_reaches_u) {
          min_v = v;
          break;
        }
      }
    }
    d_ranges[trial * s + u] = min_v;
  }
}

__host__ int parse_positive_env_int(const char* name, int default_value) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  int parsed = atoi(value);
  return parsed > 0 ? parsed : default_value;
}

__global__ void modify_TDM_Kernel(uint8_t* dev_tdm, int s, int* dev_ranges, int face,
                                  int* dev_change, int* no_change, int trials,
                                  int blocks_per_trial) {
  const size_t N = (size_t) s * s * s;

  int global_block = (int) blockIdx.x;
  int trial = global_block / blocks_per_trial;
  int block_in_tr = global_block % blocks_per_trial;

  if (trial >= trials)
    return;
  if (no_change[trial] >= 3)
    return;

  uint8_t* trial_tdm = dev_tdm + (size_t) trial * N;
  int* trial_ranges = dev_ranges + (size_t) trial * s;

  __shared__ int block_changed;
  if (threadIdx.x == 0)
    block_changed = 0;
  __syncthreads();

  for (size_t idx = (size_t) block_in_tr * blockDim.x + threadIdx.x; idx < N;
       idx += (size_t) blockDim.x * blocks_per_trial) {
    if (trial_tdm[idx] == 0)
      continue;

    int x = (int) (idx / (size_t) (s * s));
    int rem = (int) (idx - (size_t) x * (size_t) (s * s));
    int y = rem / s;
    int z = rem - y * s;

    bool toDelete = ((face == 0) && (trial_ranges[y] != trial_ranges[z])) ||
                    ((face == 1) && (trial_ranges[x] != trial_ranges[z])) ||
                    ((face == 2) && (trial_ranges[x] != trial_ranges[y]));

    if (toDelete) {
      trial_tdm[idx] = 0;
      atomicExch(&block_changed, 1);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && block_changed) {
    atomicExch(&dev_change[trial], 1);
  }
}

// face==2 fast path with the same multi-block launch shape as modify_TDM_Kernel.
__global__ void modify_TDM_face2_Kernel(uint8_t* dev_tdm, int s, int* dev_ranges, int* dev_change,
                                        int* no_change, int trials, int blocks_per_trial) {
  const size_t N = (size_t) s * s * s;
  const size_t XY = (size_t) s * s;

  int global_block = (int) blockIdx.x;
  int trial = global_block / blocks_per_trial;
  int block_in_tr = global_block % blocks_per_trial;

  if (trial >= trials)
    return;
  if (no_change[trial] >= 3)
    return;

  uint8_t* trial_tdm = dev_tdm + (size_t) trial * N;
  int* ranges = dev_ranges + (size_t) trial * (size_t) s;

  __shared__ int block_changed;
  if (threadIdx.x == 0)
    block_changed = 0;
  __syncthreads();

  for (size_t xy = (size_t) block_in_tr * blockDim.x + threadIdx.x; xy < XY;
       xy += (size_t) blockDim.x * blocks_per_trial) {
    int x = (int) (xy / (size_t) s);
    int y = (int) (xy - (size_t) x * (size_t) s);

    if (ranges[x] == ranges[y])
      continue;

    size_t base = (size_t) x * (size_t) s * (size_t) s + (size_t) y * (size_t) s;
    uint8_t* line = trial_tdm + base;

    int z = 0;
    for (; z + 4 <= s; z += 4) {
      uint8_t a = line[z];
      uint8_t b = line[z + 1];
      uint8_t c = line[z + 2];
      uint8_t d = line[z + 3];
      uint8_t any = (uint8_t) (a | b | c | d);
      if (any) {
        line[z] = 0;
        line[z + 1] = 0;
        line[z + 2] = 0;
        line[z + 3] = 0;
        atomicExch(&block_changed, 1);
      }
    }
    for (; z < s; ++z) {
      if (line[z]) {
        line[z] = 0;
        atomicExch(&block_changed, 1);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && block_changed) {
    atomicExch(&dev_change[trial], 1);
  }
}

__global__ void update_no_change_Kernel(const int* dev_change, int* no_change, int* active_count,
                                        int trials) {
  int trial = (int) blockIdx.x * blockDim.x + threadIdx.x;
  if (trial >= trials)
    return;

  int nc = no_change[trial];
  if (nc >= 3)
    return;

  if (dev_change[trial]) {
    nc = 0;
  } else {
    nc += 1;
  }
  no_change[trial] = nc;
  if (active_count != nullptr && nc < 3)
    atomicAdd(active_count, 1);
}

template <int BLOCK_THREADS>
__global__ void sum(int* __restrict__ d_result, const uint8_t* __restrict__ d_TDM, int s,
                    int trials, int blocks_per_trial) {
  using BlockReduce = cub::BlockReduce<int, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const size_t N = static_cast<size_t>(s) * s * s;

  int global_block_id = blockIdx.x;
  int trial = global_block_id / blocks_per_trial;
  int block_in_trial = global_block_id % blocks_per_trial;

  if (trial >= trials)
    return;

  const uint8_t* trial_tdm = d_TDM + trial * N;

  int thread_sum = 0;

  size_t start = (size_t) block_in_trial * (size_t) blockDim.x;
  for (size_t idx = start + (size_t) threadIdx.x; idx < N;
       idx += (size_t) blockDim.x * (size_t) blocks_per_trial) {
    thread_sum += static_cast<int>(trial_tdm[idx]);
  }

  int block_sum = BlockReduce(temp_storage).Sum(thread_sum);

  if (threadIdx.x == 0) {
    atomicAdd(&d_result[trial], block_sum);
  }
}

__device__ __forceinline__ bool isWitness16_Kernel_u32(uint32_t row1, uint32_t row2, uint32_t row3,
                                                       uint32_t allow, bool strong) {
  uint32_t y, m1, m2, m3, t;

  y = row1 ^ 0u;
  m1 = ~(y | (y >> 1)) & 0x55555555u & allow;

  y = row2 ^ 0x55555555u;
  m2 = ~(y | (y >> 1)) & 0x55555555u & allow;

  y = row3 ^ 0xAAAAAAAAu;
  m3 = ~(y | (y >> 1)) & 0x55555555u & allow;

  t = (m1 & m2) | (m1 & m3) | (m2 & m3);
  if (!strong)
    return t != 0u;

  return (t & ~((m1 & m2) & m3)) != 0u;
}

// one block per (trial, r1): cache row packed values once, then write r2/r3 plane
__global__ void computeTDM_smallK_cached_Kernel(const e_type* __restrict__ d_Puz,
                                                uint8_t* __restrict__ d_TDM, unsigned int s,
                                                unsigned int entries_per_row, uint32_t allow,
                                                bool strong, int trials) {
  int trial = (int) blockIdx.y;
  if (trial >= trials)
    return;

  int r1 = (int) blockIdx.x;
  if ((unsigned) r1 >= s)
    return;

  extern __shared__ uint32_t rowv[];
  const e_type* __restrict__ puz = d_Puz + (size_t) trial * (size_t) s * (size_t) entries_per_row;

  for (int r = (int) threadIdx.x; r < (int) s; r += (int) blockDim.x) {
    rowv[r] = (uint32_t) puz[(size_t) r * (size_t) entries_per_row];
  }
  __syncthreads();

  const uint32_t row1 = rowv[r1];

  const size_t N = (size_t) s * (size_t) s * (size_t) s;
  uint8_t* __restrict__ tdm = d_TDM + (size_t) trial * N;

  const size_t base = (size_t) r1 * (size_t) s * (size_t) s;

  for (unsigned int idx = (unsigned int) threadIdx.x; idx < s * s;
       idx += (unsigned int) blockDim.x) {
    unsigned int r2 = idx / s;
    unsigned int r3 = idx - r2 * s;

    const uint32_t row2 = rowv[r2];
    const uint32_t row3 = rowv[r3];

    tdm[base + (size_t) idx] = (uint8_t) (!isWitness16_Kernel_u32(row1, row2, row3, allow, strong));
  }
}

__device__ __forceinline__ bool isWitness_general_Kernel(const e_type* __restrict__ puz,
                                                         unsigned entries_per_row, unsigned k,
                                                         unsigned r1, unsigned r2, unsigned r3,
                                                         bool strong) {
  for (unsigned c = 0; c < k; ++c) {
    unsigned entry = c / (unsigned) ELTS_PER_ENTRY;
    unsigned shift = (c - entry * (unsigned) ELTS_PER_ENTRY) * 2;

    unsigned cur1 = r1 * entries_per_row + entry;
    unsigned cur2 = r2 * entries_per_row + entry;
    unsigned cur3 = r3 * entries_per_row + entry;

    unsigned val1 = (puz[cur1] >> shift) & 0x3u;
    unsigned val2 = (puz[cur2] >> shift) & 0x3u;
    unsigned val3 = (puz[cur3] >> shift) & 0x3u;

    unsigned count = (val1 == 0u) + (val2 == 1u) + (val3 == 2u);

    if (strong) {
      if (count == 2u)
        return true;
    } else {
      if (count >= 2u)
        return true;
    }
  }
  return false;
}

__device__ __forceinline__ bool isWitness_Kernel(const e_type* __restrict__ puz,
                                                 unsigned entries_per_row, unsigned k, unsigned r1,
                                                 unsigned r2, unsigned r3, bool strong) {
  if (k <= 16u) {
    const uint64_t mask =
        (k == 0u) ? 0ull : ((k == 16u) ? 0xFFFFFFFFull : ((1ull << (k * 2u)) - 1ull));
    const uint32_t allow = (uint32_t) (mask & 0x55555555ull);

    uint32_t row1 = (uint32_t) puz[(size_t) r1 * entries_per_row];
    uint32_t row2 = (uint32_t) puz[(size_t) r2 * entries_per_row];
    uint32_t row3 = (uint32_t) puz[(size_t) r3 * entries_per_row];

    return isWitness16_Kernel_u32(row1, row2, row3, allow, strong);
  }

  return isWitness_general_Kernel(puz, entries_per_row, k, r1, r2, r3, strong);
}

__global__ void computeTDM_Kernel(e_type* d_Puz, uint8_t* d_TDM, unsigned int s, unsigned int k,
                                  unsigned int entries_per_row, bool strong, int trials,
                                  const size_t elements_per_trial, const size_t total_elements) {
  for (size_t global_idx = (size_t) blockIdx.x * (size_t) blockDim.x + (size_t) threadIdx.x;
       global_idx < total_elements; global_idx += (size_t) gridDim.x * (size_t) blockDim.x) {
    int trial = (int) (global_idx / elements_per_trial);
    size_t local_idx = global_idx % elements_per_trial;

    e_type* puz = d_Puz + (size_t) trial * (size_t) s * (size_t) entries_per_row;
    uint8_t* tdm = d_TDM + (size_t) trial * elements_per_trial;

    unsigned r1 = (unsigned) (local_idx / (size_t) (s * s));
    unsigned rem = (unsigned) (local_idx - (size_t) r1 * (size_t) (s * s));
    unsigned r2 = rem / (unsigned) s;
    unsigned r3 = rem - r2 * (unsigned) s;

    tdm[local_idx] = (uint8_t) (!isWitness_Kernel(puz, entries_per_row, k, r1, r2, r3, strong));
  }
}

__host__ static void outer_Kernel_impl(GpuResources& gpu_resources, const e_type* puz_pinned_buffer,
                                       int* results_pinned_buffer, int s, int k, int trials,
                                       bool copy_puzzles_to_device) {
  if (trials <= 0)
    return;
  if (trials > gpu_resources.max_trials) {
    throw std::invalid_argument("outer_Kernel trials exceed allocated GpuResources capacity.");
  }

  const unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
  const size_t elements_per_trial = (size_t) s * s * s;
  const size_t total_elements = elements_per_trial * (size_t) trials;
  const size_t puzzle_bytes = sizeof(e_type) * (size_t) s * entries_per_row * (size_t) trials;
  const size_t result_bytes = sizeof(int) * (size_t) trials;

  int device_id = gpu_resources.device_id;
  cudaSetDevice(device_id);

  if (copy_puzzles_to_device) {
    cudaMemcpyAsync(gpu_resources.dev_Puz, puz_pinned_buffer, puzzle_bytes, cudaMemcpyHostToDevice,
                    gpu_resources.stream);
  }
  cudaMemsetAsync(gpu_resources.dev_no_change, 0, sizeof(int) * trials, gpu_resources.stream);

  const int TPB_TDM = TDM_THREADS;
  const int BLOCKS_PER_TRIAL_SUM = 4;
  const int default_modify_blocks_per_trial = (s >= 32) ? 4 : 2;
  const int BLOCKS_PER_TRIAL_MOD = std::min(
      16, parse_positive_env_int("ILS_GPU_MOD_BLOCKS_PER_TRIAL", default_modify_blocks_per_trial));

  if ((unsigned) k <= 16u && entries_per_row == 1u) {
    const uint64_t mask =
        ((unsigned) k == 0u)
            ? 0ull
            : (((unsigned) k == 16u) ? 0xFFFFFFFFull : ((1ull << ((unsigned) k * 2u)) - 1ull));
    const uint32_t allow = (uint32_t) (mask & 0x55555555ull);

    dim3 grid((unsigned) s, (unsigned) trials, 1);
    size_t shmem = (size_t) s * sizeof(uint32_t);

    computeTDM_smallK_cached_Kernel<<<grid, TPB_TDM, shmem, gpu_resources.stream>>>(
        gpu_resources.dev_Puz, gpu_resources.dev_TDM, (unsigned) s, entries_per_row, allow,
        /*strong=*/true, trials);
  } else {
    const int BPG_TDM = (int) ((total_elements + (size_t) TPB_TDM - 1) / (size_t) TPB_TDM);
    computeTDM_Kernel<<<BPG_TDM, TPB_TDM, 0, gpu_resources.stream>>>(
        gpu_resources.dev_Puz, gpu_resources.dev_TDM, (unsigned) s, (unsigned) k, entries_per_row,
        true, trials, elements_per_trial, total_elements);
  }

  int active_trials = trials;
  const int projection_blocks_per_trial = (s * s + TPB_TDM - 1) / TPB_TDM;

  int sms = 1;
  int projection_blocks_per_sm = 1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&projection_blocks_per_sm, calc_projection_Kernel,
                                                TPB_TDM, 0);
  const int projection_wave_blocks = std::max(1, sms * projection_blocks_per_sm);
  const int projection_trial_quantum =
      projection_wave_blocks / gcd_host_int(projection_wave_blocks, projection_blocks_per_trial);
  const int projection_trials_launch =
      ((trials + projection_trial_quantum - 1) / projection_trial_quantum) *
      projection_trial_quantum;
  int sum_blocks_per_sm = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sum_blocks_per_sm, sum<TDM_THREADS>, TDM_THREADS,
                                                0);
  const int sum_wave_blocks = std::max(1, sms * sum_blocks_per_sm);
  const int sum_trial_quantum =
      sum_wave_blocks / gcd_host_int(sum_wave_blocks, BLOCKS_PER_TRIAL_SUM);
  const int sum_trials_launch =
      ((trials + sum_trial_quantum - 1) / sum_trial_quantum) * sum_trial_quantum;

  const int max_iter = 3 * s * s;  // convergence guaranteed within O(s^2) iterations
  for (int iter = 0; iter < max_iter; ++iter) {
    const int face = iter % 3;

    cudaMemsetAsync(gpu_resources.dev_ranges, 0, sizeof(int) * s * trials, gpu_resources.stream);
    cudaMemsetAsync(gpu_resources.dev_changed, 0, sizeof(int) * trials, gpu_resources.stream);
    cudaMemsetAsync(gpu_resources.dev_projection, 0, sizeof(uint8_t) * s * s * trials,
                    gpu_resources.stream);
    cudaMemsetAsync(gpu_resources.dev_active_count, 0, sizeof(int), gpu_resources.stream);

    calc_projection_Kernel<<<dim3((unsigned) projection_blocks_per_trial,
                                  (unsigned) projection_trials_launch, 1),
                             TPB_TDM, 0, gpu_resources.stream>>>(
        s, gpu_resources.dev_TDM, gpu_resources.dev_projection, gpu_resources.dev_no_change, face,
        trials);

    if (s <= 32) {
      int trials_per_block = 256 / 32;  // 8 trials per block
      int blocks = (trials + trials_per_block - 1) / trials_per_block;
      fb_kern_TC_warp_Kernel<<<blocks, dim3(32, trials_per_block), 0, gpu_resources.stream>>>(
          s, gpu_resources.dev_projection, gpu_resources.dev_ranges, gpu_resources.dev_no_change,
          trials);
    } else {
      int words_per_row = (s + 31) / 32;
      int shared_mem_size = s * words_per_row * sizeof(uint32_t);
      fb_kern_TC_block_Kernel<<<trials, TDM_THREADS, shared_mem_size, gpu_resources.stream>>>(
          s, gpu_resources.dev_projection, gpu_resources.dev_ranges, gpu_resources.dev_no_change);
    }

    if (face == 2) {
      modify_TDM_face2_Kernel<<<trials * BLOCKS_PER_TRIAL_MOD, TPB_TDM, 0, gpu_resources.stream>>>(
          gpu_resources.dev_TDM, s, gpu_resources.dev_ranges, gpu_resources.dev_changed,
          gpu_resources.dev_no_change, trials, BLOCKS_PER_TRIAL_MOD);
    } else {
      modify_TDM_Kernel<<<trials * BLOCKS_PER_TRIAL_MOD, TPB_TDM, 0, gpu_resources.stream>>>(
          gpu_resources.dev_TDM, s, gpu_resources.dev_ranges, face, gpu_resources.dev_changed,
          gpu_resources.dev_no_change, trials, BLOCKS_PER_TRIAL_MOD);
    }

    update_no_change_Kernel<<<(trials + TPB_TDM - 1) / TPB_TDM, TPB_TDM, 0, gpu_resources.stream>>>(
        gpu_resources.dev_changed, gpu_resources.dev_no_change, gpu_resources.dev_active_count,
        trials);

    cudaMemcpyAsync(&active_trials, gpu_resources.dev_active_count, sizeof(int),
                    cudaMemcpyDeviceToHost, gpu_resources.stream);
    cudaStreamSynchronize(gpu_resources.stream);
    if (active_trials == 0)
      break;
  }

  cudaMemsetAsync(gpu_resources.dev_result, 0, sizeof(int) * trials, gpu_resources.stream);

  sum<TDM_THREADS>
      <<<sum_trials_launch * BLOCKS_PER_TRIAL_SUM, TDM_THREADS, 0, gpu_resources.stream>>>(
          gpu_resources.dev_result, gpu_resources.dev_TDM, s, trials, BLOCKS_PER_TRIAL_SUM);

  cudaMemcpyAsync(results_pinned_buffer, gpu_resources.dev_result, result_bytes,
                  cudaMemcpyDeviceToHost, gpu_resources.stream);

  cudaStreamSynchronize(gpu_resources.stream);

  TDM::num_gpu_simplify.fetch_add(static_cast<unsigned long long>(trials),
                                  std::memory_order_relaxed);
}

__host__ void outer_Kernel(GpuResources& gpu_resources, int s, int k, int trials) {
  outer_Kernel(gpu_resources, gpu_resources.puz_pinned_buffer, gpu_resources.results_pinned_buffer,
               s, k, trials);
}

__host__ void outer_Kernel(GpuResources& gpu_resources, const e_type* puz_pinned_buffer,
                           int* results_pinned_buffer, int s, int k, int trials) {
  outer_Kernel_impl(gpu_resources, puz_pinned_buffer, results_pinned_buffer, s, k, trials, true);
}

__host__ void outer_KernelDeviceInput(GpuResources& gpu_resources, int* results_pinned_buffer,
                                      int s, int k, int trials) {
  outer_Kernel_impl(gpu_resources, nullptr, results_pinned_buffer, s, k, trials, false);
}
