#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <vector>

struct GpuPuzzleSize {
  unsigned int rows;
  unsigned int columns;

  bool operator==(GpuPuzzleSize other) const {
    return rows == other.rows && columns == other.columns;
  }
};

// All coordinator lanes draw from the same size-specific queues.
class GpuBatchQueue {
 public:
  using Clock = std::chrono::steady_clock;

  GpuBatchQueue(std::size_t slot_capacity, std::size_t group_capacity,
                std::chrono::microseconds max_wait);

  // Called by the search control thread. Removed sizes remain available to consumers until drained.
  void setActiveSizes(const std::vector<GpuPuzzleSize>& sizes, std::ostream& log);
  void submit(GpuPuzzleSize size, int slot_id);
  bool take(GpuPuzzleSize& size, std::vector<int>& slots);
  void complete(GpuPuzzleSize size, std::size_t slots, std::size_t candidates);
  void stop();

 private:
  struct Slot {
    int id;
    Clock::time_point submitted;
  };

  struct SizeQueue {
    GpuPuzzleSize size;
    std::vector<Slot> slots;
    std::vector<std::size_t> group_sizes;
    bool active = true;
    std::size_t head = 0;
    std::size_t pending = 0;
    std::size_t in_flight = 0;
    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t candidates = 0;
    std::size_t timed_groups = 0;
    std::size_t drain_groups = 0;
    std::chrono::nanoseconds total_wait{};
    std::chrono::nanoseconds longest_wait{};
  };

  SizeQueue& queueFor(GpuPuzzleSize size);

  const std::size_t slot_capacity;
  const std::size_t group_capacity;
  const std::chrono::microseconds max_wait;
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<SizeQueue> queues;
  std::size_t next_queue = 0;
  bool stopping = false;
};
