#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "BoundedPuzPriorityQueue.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

q_elt_t candidate(double gap, int id) {
  return {gap, Puz(1, 1), id};
}

void test_ties() {
  BPPQ queue(3);
  queue.enqueue(candidate(3, 1));
  queue.enqueue(candidate(2, 2));
  queue.enqueue(candidate(2, 3));
  require(!queue.enqueue(candidate(2, 4)), "an equal cutoff must be rejected");
  require(queue.enqueue(candidate(3, 5)), "a better score must be accepted");
  require(queue.size() == 3, "replacing one element removed a tied element");

  q_elt_t chosen;
  require(queue.dequeue(chosen) && chosen.elt_type == 1, "first best element was lost");
  require(queue.size() == 2 && queue.best() == 3, "dequeue removed a second best element");
  require(queue.dequeue(chosen) && chosen.elt_type == 5, "second best element was lost");
  require(queue.size() == 1 && queue.best() == 2 && queue.worst() == 2,
          "singleton bounds disagree with the remaining element");
  require(queue.dequeue(chosen) && chosen.elt_type == 2, "wrong tied element evicted");
}

void test_batch_replay() {
  std::mt19937 random(73);
  int next_id = 0;
  for (size_t limit = 1; limit <= 16; ++limit) {
    BPPQ queue(limit);
    std::vector<q_elt_t> reference;
    for (int step = 0; step < 100; ++step) {
      BPPQ source(32);
      std::vector<q_elt_t> incoming;
      for (unsigned int i = random() % 32; i > 0; --i) {
        q_elt_t entry = candidate(static_cast<int>(random() % 11) - 10, next_id++);
        source.enqueue(entry);
        incoming.push_back(std::move(entry));
      }
      std::stable_sort(incoming.begin(), incoming.end(), cmp{});

      size_t count = 0;
      for (const auto& entry : incoming) {
        if (reference.size() == limit) {
          if (entry.gap <= reference.back().gap) {
            continue;
          }
          reference.pop_back();
        }
        auto position = std::find_if(reference.begin(), reference.end(),
                                     [&](const auto& old) { return old.gap < entry.gap; });
        reference.insert(position, entry);
        ++count;
      }

      require(queue.merge_from(source) == count, "batch admission counts differ");
      require(source.size() == incoming.size() - count, "rejected suffix size differs");
      require(queue.size() == reference.size(), "frontier size differs after merge");
      if (!reference.empty()) {
        require(queue.best() == reference.front().gap && queue.worst() == reference.back().gap,
                "frontier bounds differ after merge");
      }

      for (unsigned int i = random() % 8; i > 0 && !reference.empty(); --i) {
        q_elt_t chosen;
        require(queue.dequeue(chosen), "frontier unexpectedly empty");
        require(chosen.gap == reference.front().gap &&
                    chosen.elt_type == reference.front().elt_type,
                "frontier dequeue order differs");
        reference.erase(reference.begin());
      }
    }
  }
}

void test_checkpoint() {
  std::stringstream data;
  const size_t size = 2;
  const size_t limit = 3;
  const double stale_bound = 42;
  data.write(reinterpret_cast<const char*>(&size), sizeof(size));
  candidate(3, 9).serialize(data);
  candidate(3, 10).serialize(data);
  data.write(reinterpret_cast<const char*>(&stale_bound), sizeof(stale_bound));
  data.write(reinterpret_cast<const char*>(&stale_bound), sizeof(stale_bound));
  data.write(reinterpret_cast<const char*>(&limit), sizeof(limit));

  BPPQ queue;
  queue.deserialize(data);
  require(static_cast<bool>(data) && queue.size() == size && queue.best() == 3 &&
              queue.worst() == 3,
          "legacy checkpoint should derive actual bounds from its entries");
  std::stringstream saved;
  queue.serialize(saved);
  BPPQ resumed;
  resumed.deserialize(saved);
  q_elt_t chosen;
  require(resumed.dequeue(chosen) && chosen.elt_type == 9, "checkpoint lost first tied entry");
  require(resumed.dequeue(chosen) && chosen.elt_type == 10, "checkpoint lost second tied entry");
}

}  // namespace

int main() {
  test_ties();
  test_batch_replay();
  test_checkpoint();
}
