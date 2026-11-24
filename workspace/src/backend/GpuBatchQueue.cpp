#include "GpuBatchQueue.h"

#include <algorithm>
#include <ostream>
#include <stdexcept>

GpuBatchQueue::GpuBatchQueue(std::size_t slot_capacity, std::size_t group_capacity,
                             std::chrono::microseconds max_wait)
    : slot_capacity(slot_capacity), group_capacity(group_capacity), max_wait(max_wait) {
  if (slot_capacity == 0 || group_capacity == 0 || max_wait.count() < 0) {
    throw std::invalid_argument("Invalid GPU batch queue capacity or wait time");
  }
}

void GpuBatchQueue::setActiveSizes(const std::vector<GpuPuzzleSize>& sizes, std::ostream& log) {
  std::unique_lock<std::mutex> lock(mutex);
  for (auto& queue : queues) {
    queue.active = std::find(sizes.begin(), sizes.end(), queue.size) != sizes.end();
  }
  ready.notify_all();
  ready.wait(lock, [&] {
    return std::all_of(queues.begin(), queues.end(), [](const SizeQueue& queue) {
      return queue.active || (queue.pending == 0 && queue.in_flight == 0);
    });
  });

  for (const auto& queue : queues) {
    if (queue.active) {
      continue;
    }
    log << "GPU queue retired (" << queue.size.rows << ',' << queue.size.columns
        << ") pending=" << queue.pending << " in_flight=" << queue.in_flight
        << " submitted=" << queue.submitted << " completed=" << queue.completed
        << " candidates=" << queue.candidates
        << " mean_wait_us="
        << (queue.completed ? queue.total_wait.count() / 1000.0 / queue.completed : 0.)
        << " max_wait_us=" << queue.longest_wait.count() / 1000.0
        << " timed_groups=" << queue.timed_groups << " drain_groups=" << queue.drain_groups
        << " groups_by_slots=";
    for (std::size_t count = 1; count < queue.group_sizes.size(); ++count) {
      if (queue.group_sizes[count]) {
        log << ' ' << count << '=' << queue.group_sizes[count];
      }
    }
    log << std::endl;
  }
  queues.erase(std::remove_if(queues.begin(), queues.end(), [](const SizeQueue& queue) {
                 return !queue.active;
               }),
               queues.end());

  for (auto size : sizes) {
    auto existing = std::find_if(queues.begin(), queues.end(), [size](const SizeQueue& queue) {
      return queue.size == size;
    });
    if (existing == queues.end()) {
      queues.push_back({size, std::vector<Slot>(slot_capacity),
                        std::vector<std::size_t>(group_capacity + 1)});
      log << "GPU queue opened (" << size.rows << ',' << size.columns << ')' << std::endl;
    }
  }
  next_queue = 0;
}

GpuBatchQueue::SizeQueue& GpuBatchQueue::queueFor(GpuPuzzleSize size) {
  auto queue = std::find_if(queues.begin(), queues.end(), [size](const SizeQueue& queue) {
    return queue.size == size;
  });
  if (queue == queues.end()) {
    throw std::logic_error("GPU slot refers to an unregistered puzzle size");
  }
  return *queue;
}

void GpuBatchQueue::submit(GpuPuzzleSize size, int slot_id) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto& queue = queueFor(size);
    if (!queue.active || stopping || queue.pending + queue.in_flight >= slot_capacity) {
      throw std::logic_error("Cannot submit to a closed or full GPU size queue");
    }
    std::size_t tail = (queue.head + queue.pending) % slot_capacity;
    queue.slots[tail] = {slot_id, Clock::now()};
    ++queue.pending;
    ++queue.submitted;
  }
  ready.notify_one();
}

bool GpuBatchQueue::take(GpuPuzzleSize& size, std::vector<int>& slots) {
  slots.clear();
  slots.reserve(group_capacity);
  std::unique_lock<std::mutex> lock(mutex);
  for (;;) {
    auto now = Clock::now();
    auto next_deadline = Clock::time_point::max();
    for (std::size_t offset = 0; offset < queues.size(); ++offset) {
      std::size_t index = (next_queue + offset) % queues.size();
      auto& queue = queues[index];
      if (queue.pending == 0) {
        continue;
      }
      auto deadline = queue.slots[queue.head].submitted + max_wait;
      if (queue.pending < group_capacity && now < deadline && queue.active && !stopping) {
        next_deadline = std::min(next_deadline, deadline);
        continue;
      }

      size = queue.size;
      std::size_t count = std::min(queue.pending, group_capacity);
      for (std::size_t i = 0; i < count; ++i) {
        const Slot& slot = queue.slots[queue.head];
        slots.push_back(slot.id);
        auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(now - slot.submitted);
        queue.total_wait += wait;
        queue.longest_wait = std::max(queue.longest_wait, wait);
        queue.head = (queue.head + 1) % slot_capacity;
      }
      queue.pending -= count;
      queue.in_flight += count;
      ++queue.group_sizes[count];
      if (count < group_capacity) {
        if (queue.active && !stopping) {
          ++queue.timed_groups;
        } else {
          ++queue.drain_groups;
        }
      }
      next_queue = (index + 1) % queues.size();
      lock.unlock();
      ready.notify_one();
      return true;
    }
    if (stopping) {
      return false;
    }
    if (next_deadline == Clock::time_point::max()) {
      ready.wait(lock);
    } else {
      ready.wait_until(lock, next_deadline);
    }
  }
}

void GpuBatchQueue::complete(GpuPuzzleSize size, std::size_t slots, std::size_t candidates) {
  bool retired;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto& queue = queueFor(size);
    if (slots > queue.in_flight) {
      throw std::logic_error("GPU completed more slots than it took");
    }
    queue.in_flight -= slots;
    queue.completed += slots;
    queue.candidates += candidates;
    retired = !queue.active && queue.pending == 0 && queue.in_flight == 0;
  }
  if (retired) {
    ready.notify_all();
  }
}

void GpuBatchQueue::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    stopping = true;
  }
  ready.notify_all();
}
