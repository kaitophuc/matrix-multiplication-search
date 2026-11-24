#include "GpuBatchQueue.h"

#include <array>
#include <atomic>
#include <future>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;

namespace {
void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("GPU batch queue invariant failed");
  }
}

void testRetirement() {
  const GpuPuzzleSize old{48, 9};
  GpuBatchQueue queue(32, 2, 500ms);
  std::ostringstream log;
  queue.setActiveSizes({old, {49, 9}, {50, 9}, {51, 9}}, log);
  queue.submit(old, 0);
  queue.submit({51, 9}, 1);
  queue.submit(old, 2);
  queue.submit(old, 3);

  GpuPuzzleSize size{};
  std::vector<int> slots;
  require(queue.take(size, slots) && size == old && slots == std::vector<int>{0, 2});

  auto transition = std::async(std::launch::async, [&] {
    queue.setActiveSizes({{49, 9}, {50, 9}, {51, 9}, {52, 9}}, log);
  });
  require(queue.take(size, slots) && size == old && slots == std::vector<int>{3});
  queue.complete(old, 1, 227);
  require(transition.wait_for(0ms) == std::future_status::timeout);
  queue.complete(old, 2, 512);
  transition.get();
  require(log.str().find("retired (48,9) pending=0 in_flight=0 submitted=3 completed=3") !=
          std::string::npos);
  require(log.str().find("retired (48,9)") < log.str().find("opened (52,9)"));
  bool rejected = false;
  try {
    queue.submit(old, 4);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  require(rejected);

  queue.submit({52, 9}, 0);
  queue.stop();
  require(queue.take(size, slots) && size == GpuPuzzleSize{51, 9} && slots == std::vector<int>{1});
  queue.complete(size, 1, 256);
  require(queue.take(size, slots) && size == GpuPuzzleSize{52, 9} && slots == std::vector<int>{0});
  queue.complete(size, 1, 256);
  require(!queue.take(size, slots));
  queue.setActiveSizes({}, log);
}

void testConcurrentConsumers() {
  const std::array<GpuPuzzleSize, 4> sizes{{{49, 9}, {50, 9}, {51, 9}, {51, 8}}};
  constexpr int slot_count = 1024;
  GpuBatchQueue queue(slot_count, 14, 1ms);
  std::ostringstream log;
  queue.setActiveSizes(std::vector<GpuPuzzleSize>(sizes.begin(), sizes.end()), log);

  std::array<std::atomic<int>, slot_count> completions{};
  for (auto& count : completions) {
    count.store(0);
  }
  std::array<std::thread, 4> lanes;
  for (auto& lane : lanes) {
    lane = std::thread([&] {
      GpuPuzzleSize size{};
      std::vector<int> slots;
      slots.reserve(14);
      while (queue.take(size, slots)) {
        require(slots.size() <= 14);
        for (int id : slots) {
          require(size == sizes[id % sizes.size()]);
          completions[id].fetch_add(1);
        }
        queue.complete(size, slots.size(), slots.size() * 256);
      }
    });
  }
  std::array<std::thread, 4> producers;
  for (std::size_t worker = 0; worker < producers.size(); ++worker) {
    producers[worker] = std::thread([&, worker] {
      for (int id = worker; id < slot_count; id += producers.size()) {
        queue.submit(sizes[id % sizes.size()], id);
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  queue.stop();
  for (auto& lane : lanes) {
    lane.join();
  }
  for (auto& count : completions) {
    require(count.load() == 1);
  }
  queue.setActiveSizes({}, log);
}

void testQueueReuseAndFairness() {
  GpuBatchQueue queue(4, 2, 0us);
  std::ostringstream log;
  const GpuPuzzleSize a{51, 9}, b{52, 9};
  queue.setActiveSizes({a, b}, log);
  for (int id = 0; id < 4; ++id) {
    queue.submit(a, id);
  }
  queue.submit(b, 4);
  queue.submit(b, 5);

  GpuPuzzleSize size{};
  std::vector<int> slots;
  require(queue.take(size, slots) && size == a && slots == std::vector<int>{0, 1});
  queue.complete(size, 2, 512);
  queue.submit(a, 0);
  queue.submit(a, 1);
  require(queue.take(size, slots) && size == b && slots == std::vector<int>{4, 5});
  queue.complete(size, 2, 512);
  require(queue.take(size, slots) && size == a && slots == std::vector<int>{2, 3});
  queue.complete(size, 2, 512);
  require(queue.take(size, slots) && size == a && slots == std::vector<int>{0, 1});
  queue.complete(size, 2, 512);
  queue.setActiveSizes({}, log);
}

void testPartialGroupDeadline() {
  GpuBatchQueue queue(16, 14, 5ms);
  std::ostringstream log;
  GpuPuzzleSize size{52, 9};
  queue.setActiveSizes({size}, log);
  auto start = GpuBatchQueue::Clock::now();
  queue.submit(size, 0);
  std::vector<int> slots;
  require(queue.take(size, slots) && slots == std::vector<int>{0});
  require(GpuBatchQueue::Clock::now() - start >= 5ms);
  queue.complete(size, 1, 256);
  queue.setActiveSizes({}, log);
  require(log.str().find("timed_groups=1") != std::string::npos);
}
}  // namespace

int main() {
  testRetirement();
  testConcurrentConsumers();
  testQueueReuseAndFairness();
  testPartialGroupDeadline();
}
