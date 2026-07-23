#pragma once

#include "TDM_cuda_enhanced.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

enum class GpuTaskState : unsigned char {
  Empty,
  Filling,
  Ready,
  Processing,
  Done,
};

const char* gpuTaskStateName(GpuTaskState state);

class GpuTaskIdQueue {
 public:
  explicit GpuTaskIdQueue(size_t capacity);

  GpuTaskIdQueue(const GpuTaskIdQueue&) = delete;
  GpuTaskIdQueue& operator=(const GpuTaskIdQueue&) = delete;

  bool try_push(int value);
  bool try_pop(int& value);
  bool empty() const;
  size_t size_approx() const;
  size_t capacity() const;

 private:
  struct Cell {
    std::atomic<size_t> sequence;
    int value;

    Cell() : sequence(0), value(-1) {}
  };

  size_t cap;
  std::unique_ptr<Cell[]> cells;
  std::atomic<size_t> head;
  std::atomic<size_t> tail;
};

struct GPUResult {
  std::vector<int> results;
  GPUResult() = default;
};

class GpuTask {
 public:
  std::vector<Puz> puzzles;
  int s, k;

  GPUResult result;
  bool done = false;
  std::mutex m;
  std::condition_variable cv;

  int batch_size = 0;
  int origin_worker_id = -1;
  int slot_id = -1;
  unsigned long long generation = 0;
  int q_prior = 0;
  double gap = 0.0;
  int frontier_type = 0;
  std::atomic<GpuTaskState> state;

  e_type* puz_pinned_buffer = nullptr;
  int* results_pinned_buffer = nullptr;
  bool owns_pinned_buffers = false;
  int max_s = 0;
  int max_k = 0;
  int max_batch = 0;
  unsigned int max_entries_per_row = 0;
  size_t pinned_puz_entries = 0;

  GpuTask() : s(1), k(1), state(GpuTaskState::Empty) {}

  GpuTask(const GpuTask& other)
      : puzzles(other.puzzles),
        s(other.s),
        k(other.k),
        result(other.result),
        done(other.done),
        batch_size(other.batch_size),
        origin_worker_id(other.origin_worker_id),
        slot_id(other.slot_id),
        generation(other.generation),
        q_prior(other.q_prior),
        gap(other.gap),
        frontier_type(other.frontier_type),
        state(other.state.load(std::memory_order_acquire)),
        m(),
        cv(),
        puz_pinned_buffer(nullptr),
        results_pinned_buffer(nullptr),
        owns_pinned_buffers(false),
        max_s(0),
        max_k(0),
        max_batch(0),
        max_entries_per_row(0),
        pinned_puz_entries(0) {
    // Note: synchronization primitives and pinned buffers are not copied.
  }

  GpuTask& operator=(const GpuTask& other) {
    if (this != &other) {
      s = other.s;
      k = other.k;
      done = other.done;
      batch_size = other.batch_size;
      origin_worker_id = other.origin_worker_id;
      slot_id = other.slot_id;
      generation = other.generation;
      q_prior = other.q_prior;
      gap = other.gap;
      frontier_type = other.frontier_type;
      state.store(other.state.load(std::memory_order_acquire), std::memory_order_release);
      result.results.clear();
      puzzles.clear();
      for (const auto& val : other.result.results) {
        result.results.push_back(val);
      }
      for (const auto& p : other.puzzles) {
        puzzles.push_back(p);
      }
      // Note: m and cv are not copied.
    }
    return *this;
  }

  ~GpuTask();

  long long createId() {
    static std::atomic<long long> counter{0};
    return ++counter;
  }

  void allocatePinned(int max_s, int max_k, int max_batch, int slot_id, int owner_worker_id);
  void releasePinned();
  bool tryBeginFilling(int worker_id, int s, int k, int q_prior, double gap, int frontier_type);
  void appendPuzzle(const Puz& p);
  void appendPuzzle(Puz&& p);
  void markReady();
  bool markProcessing();
  void markDone();
  void markEmpty();
  GpuTaskState getState() const;
  void waitForState(GpuTaskState desired);
  void copyResultsToPuzzles();

  void solve(GpuResources& gpu_resources);
  void solveFromPinnedBuffers(GpuResources& gpu_resources);

  void generateRandom(int s, int k, int trials);

  /**
   * @brief Encodes the puzzle data into a binary file.
   * The format is:
   * 1. Number of puzzles (size_t)
   * 2. Height of puzzles (s)
   * 3. Width of puzzles (k)
   * 4. For each puzzle:
   *    - Size of serialized data (size_t)
   *    - Serialized data (unsigned char[])
   * 5. GPUResult:
   *   - Results (int[])
   * 6. Done flag (bool)
   * 7. Mutex state (std::mutex)
   * 8. Condition variable state (std::condition_variable)
   */
  void encode(const std::string& filename) const;

  /**
   * @brief Decodes puzzle data from a binary file.
   * @param filename The name of the file to read from.
   */
  void decode(const std::string& filename);
};

void solveGpuTask(GpuResources& gpu_resources, const std::string& input_filename,
                  const std::string& output_filename);

void solveGpuTaskByHTCondor(GpuTask& task);
