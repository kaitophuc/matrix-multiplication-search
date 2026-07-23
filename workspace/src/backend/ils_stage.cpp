#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <strings.h>
#include <omp.h>
#include <queue>
#include <memory>
#include <assert.h>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <utility>
#include <cstdlib>

#include "ils_stage.h"
#include "Puz.h"
#include "BoundedPuzPriorityQueue.h"
#include "special.h"
#include "checker.h"
#include "fitness.h"
#include "canonization.h"
#include "TDM_cuda_enhanced.h"
#include "GpuTask.h"
#include "utils.h"

// Macros that depend on variable s
#define MAX_RESTARTS 1000
#define MAX_RESULTS 20
#define Q_LEVELS 5
#define Q_PRIORS 5
#define REQUIRE_SPECIAL 0
#define RANDOM_FRONTIER 10 * (((double) stats.since_progress / 1000.0 + 1))

#define NUM_STAGES 4

// Loading Probabilities
#define PROB_PREV 0.3
#define PROB_GET_PREV 0.005
#define PROB_CURR (1.0 - PROB_GET_PREV - PROB_PREV)

using namespace std;

extern std::vector<std::unique_ptr<GpuTaskIdQueue> > gpuTaskQueues;
extern std::atomic<unsigned int> nextGpuTaskQueue;
extern std::vector<std::unique_ptr<GpuTask> > gpuTaskSlots;
extern std::mutex gpuQueueMutex;
extern std::condition_variable gpuQueueCv;
extern int gpuSlotsPerWorker;

namespace {
unsigned long long num_rows_for_width(int k) {
  unsigned long long rows = 1;
  for (int c = 0; c < k; c++) {
    rows *= 3ULL;
  }
  return rows;
}

bool env_enabled(const char* name, bool default_value) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  if (value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
      value[0] == 'n' || value[0] == 'N') {
    return false;
  }
  return true;
}

bool use_streaming_row_replacements() {
  static bool enabled = env_enabled("ILS_GPU_STREAM_ROW_REPLACEMENTS", true);
  return enabled;
}

int gpu_done_drain_limit() {
  static int limit = [] {
    const char* value = getenv("ILS_GPU_DONE_DRAIN_LIMIT");
    if (value == nullptr || value[0] == '\0') {
      return 4;
    }
    int parsed = atoi(value);
    return parsed >= 0 ? parsed : 4;
  }();
  return limit;
}

void append_row_replacements(vector<Puz>& puzzles, const Puz& base, int row_to_replace, int k) {
  const unsigned long long num_rows = num_rows_for_width(k);
  puzzles.reserve(puzzles.size() + (size_t) num_rows);
  for (unsigned long long row = 0; row < num_rows; row++) {
    Puz p_curr = base;
    unsigned long long curr = row;
    for (int c = 0; c < k; c++) {
      p_curr.set(row_to_replace, c, curr % 3 + 1);
      curr /= 3;
    }
    puzzles.push_back(std::move(p_curr));
  }
}
} // namespace

void ILSStage::init_stats() {
  stats = {0, 0, 0, 0, 0, 0, 1, 0, 0, ILS_INACTIVE, chrono::high_resolution_clock::now()};
  TDM::setNumSimplify(0);
  TDM::setNumGPUSimplify(0);
  bzero(stats.searched_types, NUM_ELT_TYPE * sizeof(int));
}

bool ILSStage::is_active(int r, const vector<int>& idxs) {
  if (idxs.empty())
    return true;  // Used when CHECK_FULL.
  for (size_t i = 0; i < idxs.size(); i++)
    if (idxs[i] == r)
      return true;
  return false;
}

void ILSStage::update_queue(BPMFQ* q, BPPQ** lists, unsigned int q_level, stats_t& stats) {
  size_t size = 0;
  for (int i = 0; i < Q_PRIORS; i++) {
    size += lists[i]->size();
  }

  size_t enqueued = q->mass_enqueue(lists, q_level);
  stats.dropped += size - enqueued;
  stats.evals += size;
}

unsigned int ILSStage::crop_prior(int q_prior) {
  if (q_prior < 0)
    return 0;
  if (q_prior >= Q_PRIORS)
    return Q_PRIORS - 1;
  return (unsigned int) q_prior;
}

bool ILSStage::add_to_frontier(const Puz& p, FitTester* ft, double gap, BPPQ** new_frontiers,
                               unsigned int q_prior, int type) {
  double curr_fit, max_fit;
  tie(curr_fit, max_fit) = ft->getFit(p);
  // Time consuming

  if (curr_fit - max_fit < gap) {
    // Got worse, reset to 0 priority.
    new_frontiers[crop_prior(q_prior - 2)]->enqueue(
        q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
  } else if (curr_fit - max_fit == gap) {
    // Stayed same, keep same priority.
    new_frontiers[crop_prior(q_prior - 1)]->enqueue(
        q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
  } else {
    // Got better, improve priority.
    new_frontiers[crop_prior(q_prior + 1)]->enqueue(
        q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
  }

  return (prev != NULL && curr_fit == max_fit);
}

void ILSStage::expand_frontier_GPU(vector<Puz>& puzzles, check_mode_t mode, bool strong,
                                   FitTester* ft, BPPQ** new_frontiers, int q_prior, double gap,
                                   int num, elt_type type) {
  if (num <= 0 || puzzles.empty())
    return;

  if (gpuTaskQueues.empty() || gpuTaskSlots.empty()) {
    throw std::runtime_error("GPU task slots are not initialized.");
  }

  int worker_id = omp_in_parallel() ? omp_get_thread_num() : 0;
  int slots_per_worker = std::max(2, gpuSlotsPerWorker);
  int first_slot_id = slots_per_worker * worker_id;
  int last_slot_id = first_slot_id + slots_per_worker;
  if (last_slot_id > (int) gpuTaskSlots.size()) {
    throw std::runtime_error("OpenMP worker does not have GPU task slots.");
  }
  vector<int> slot_ids;
  slot_ids.reserve((size_t) slots_per_worker);
  for (int slot_id = first_slot_id; slot_id < last_slot_id; slot_id++) {
    slot_ids.push_back(slot_id);
  }

  int outstanding = 0;
  bool stop_processing = false;

  auto process_done_slot = [&](GpuTask& slot) {
    if (slot.getState() != GpuTaskState::Done) {
      return false;
    }

    if (!stop_processing) {
      slot.copyResultsToPuzzles();
      for (int puz = 0; puz < slot.batch_size; puz++) {
        if (add_to_frontier(slot.puzzles[puz], ft, slot.gap, new_frontiers, slot.q_prior,
                            slot.frontier_type)) {
          stop_processing = true;
          break;
        }
        if (prev != nullptr && numResult() != 0) {
          stop_processing = true;
          break;
        }
      }
    }

    slot.markEmpty();
    outstanding--;
    return true;
  };

  auto process_done_slots = [&](int max_slots) {
    bool progressed = false;
    int processed = 0;
    for (int slot_id : slot_ids) {
      if (max_slots > 0 && processed >= max_slots) {
        break;
      }
      bool slot_progressed = process_done_slot(*gpuTaskSlots[slot_id]);
      if (slot_progressed) {
        processed++;
      }
      progressed = slot_progressed || progressed;
    }
    return progressed;
  };

  auto process_any_done_slot = [&]() {
    return process_done_slots(0);
  };

  auto process_some_done_slots = [&]() {
    int max_slots = gpu_done_drain_limit();
    if (max_slots == 0) {
      return false;
    }
    return process_done_slots(max_slots);
  };

  auto wait_for_slot_progress = [&]() {
    for (int slot_id : slot_ids) {
      GpuTask& slot = *gpuTaskSlots[slot_id];
      GpuTaskState state = slot.getState();
      if (state == GpuTaskState::Ready || state == GpuTaskState::Processing) {
        std::unique_lock<std::mutex> lock(slot.m);
        slot.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
          GpuTaskState current = slot.getState();
          return current == GpuTaskState::Done || current == GpuTaskState::Empty;
        });
        return;
      }
    }
    std::this_thread::yield();
  };

  auto acquire_slot = [&]() -> GpuTask* {
    while (true) {
      process_some_done_slots();
      if (stop_processing) {
        return nullptr;
      }

      for (int slot_id : slot_ids) {
        GpuTask& slot = *gpuTaskSlots[slot_id];
        if (slot.tryBeginFilling(worker_id, s, k, q_prior, gap, (int) type)) {
          return &slot;
        }
      }

      bool drained_slot = false;
      for (int slot_id : slot_ids) {
        GpuTask& slot = *gpuTaskSlots[slot_id];
        if (slot.getState() == GpuTaskState::Done) {
          process_done_slot(slot);
          drained_slot = true;
          break;
        }
      }
      if (drained_slot) {
        continue;
      }

      wait_for_slot_progress();
    }
  };

  auto submit_slot = [&](GpuTask& slot) {
    slot.markReady();
    const size_t queue_count = gpuTaskQueues.size();
    size_t queue_id =
        nextGpuTaskQueue.fetch_add(1, std::memory_order_acq_rel) % queue_count;
    while (true) {
      for (size_t attempt = 0; attempt < queue_count; attempt++) {
        GpuTaskIdQueue& queue = *gpuTaskQueues[(queue_id + attempt) % queue_count];
        if (queue.try_push(slot.slot_id)) {
          outstanding++;
          gpuQueueCv.notify_all();
          return;
        }
      }
      process_some_done_slots();
      std::this_thread::yield();
    }
  };

  for (size_t next = 0; next < puzzles.size() && !stop_processing;) {
    GpuTask* slot = acquire_slot();
    if (slot == nullptr) {
      break;
    }
    size_t batch = std::min((size_t) TRIALS, puzzles.size() - next);
    for (size_t i = 0; i < batch; i++) {
      slot->appendPuzzle(std::move(puzzles[next + i]));
    }
    submit_slot(*slot);
    next += batch;
  }

  while (outstanding > 0) {
    if (!process_any_done_slot()) {
      wait_for_slot_progress();
    }
  }
}

void ILSStage::expand_frontier_GPU_row_replacements(const Puz& base, int row_to_replace,
                                                    check_mode_t mode, bool strong,
                                                    FitTester* ft, BPPQ** new_frontiers,
                                                    int q_prior, double gap, elt_type type) {
  const unsigned long long num_rows = num_rows_for_width(k);
  if (num_rows == 0) {
    return;
  }

  if (gpuTaskQueues.empty() || gpuTaskSlots.empty()) {
    throw std::runtime_error("GPU task slots are not initialized.");
  }

  int worker_id = omp_in_parallel() ? omp_get_thread_num() : 0;
  int slots_per_worker = std::max(2, gpuSlotsPerWorker);
  int first_slot_id = slots_per_worker * worker_id;
  int last_slot_id = first_slot_id + slots_per_worker;
  if (last_slot_id > (int) gpuTaskSlots.size()) {
    throw std::runtime_error("OpenMP worker does not have GPU task slots.");
  }

  vector<int> slot_ids;
  slot_ids.reserve((size_t) slots_per_worker);
  for (int slot_id = first_slot_id; slot_id < last_slot_id; slot_id++) {
    slot_ids.push_back(slot_id);
  }

  int outstanding = 0;
  bool stop_processing = false;

  auto process_done_slot = [&](GpuTask& slot) {
    if (slot.getState() != GpuTaskState::Done) {
      return false;
    }

    if (!stop_processing) {
      slot.copyResultsToPuzzles();
      for (int puz = 0; puz < slot.batch_size; puz++) {
        if (add_to_frontier(slot.puzzles[puz], ft, slot.gap, new_frontiers, slot.q_prior,
                            slot.frontier_type)) {
          stop_processing = true;
          break;
        }
        if (prev != nullptr && numResult() != 0) {
          stop_processing = true;
          break;
        }
      }
    }

    slot.markEmpty();
    outstanding--;
    return true;
  };

  auto process_done_slots = [&](int max_slots) {
    bool progressed = false;
    int processed = 0;
    for (int slot_id : slot_ids) {
      if (max_slots > 0 && processed >= max_slots) {
        break;
      }
      bool slot_progressed = process_done_slot(*gpuTaskSlots[slot_id]);
      if (slot_progressed) {
        processed++;
      }
      progressed = slot_progressed || progressed;
    }
    return progressed;
  };

  auto process_any_done_slot = [&]() {
    return process_done_slots(0);
  };

  auto process_some_done_slots = [&]() {
    int max_slots = gpu_done_drain_limit();
    if (max_slots == 0) {
      return false;
    }
    return process_done_slots(max_slots);
  };

  auto wait_for_slot_progress = [&]() {
    for (int slot_id : slot_ids) {
      GpuTask& slot = *gpuTaskSlots[slot_id];
      GpuTaskState state = slot.getState();
      if (state == GpuTaskState::Ready || state == GpuTaskState::Processing) {
        std::unique_lock<std::mutex> lock(slot.m);
        slot.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
          GpuTaskState current = slot.getState();
          return current == GpuTaskState::Done || current == GpuTaskState::Empty;
        });
        return;
      }
    }
    std::this_thread::yield();
  };

  auto acquire_slot = [&]() -> GpuTask* {
    while (true) {
      process_some_done_slots();
      if (stop_processing) {
        return nullptr;
      }

      for (int slot_id : slot_ids) {
        GpuTask& slot = *gpuTaskSlots[slot_id];
        if (slot.tryBeginFilling(worker_id, s, k, q_prior, gap, (int) type)) {
          return &slot;
        }
      }

      bool drained_slot = false;
      for (int slot_id : slot_ids) {
        GpuTask& slot = *gpuTaskSlots[slot_id];
        if (slot.getState() == GpuTaskState::Done) {
          process_done_slot(slot);
          drained_slot = true;
          break;
        }
      }
      if (drained_slot) {
        continue;
      }

      wait_for_slot_progress();
    }
  };

  auto submit_slot = [&](GpuTask& slot) {
    slot.markReady();
    const size_t queue_count = gpuTaskQueues.size();
    size_t queue_id =
        nextGpuTaskQueue.fetch_add(1, std::memory_order_acq_rel) % queue_count;
    while (true) {
      for (size_t attempt = 0; attempt < queue_count; attempt++) {
        GpuTaskIdQueue& queue = *gpuTaskQueues[(queue_id + attempt) % queue_count];
        if (queue.try_push(slot.slot_id)) {
          outstanding++;
          gpuQueueCv.notify_all();
          return;
        }
      }
      process_some_done_slots();
      std::this_thread::yield();
    }
  };

  GpuTask* current_slot = nullptr;
  auto flush_current_slot = [&]() {
    if (current_slot != nullptr && current_slot->batch_size > 0) {
      submit_slot(*current_slot);
      current_slot = nullptr;
    }
  };

  for (unsigned long long row = 0; row < num_rows && !stop_processing; row++) {
    if (current_slot == nullptr) {
      current_slot = acquire_slot();
      if (current_slot == nullptr) {
        break;
      }
    }

    Puz p_curr = base;
    unsigned long long curr = row;
    for (int c = 0; c < k; c++) {
      p_curr.set(row_to_replace, c, curr % 3 + 1);
      curr /= 3;
    }

    current_slot->appendPuzzle(std::move(p_curr));
    if (current_slot->batch_size == TRIALS) {
      flush_current_slot();
      process_some_done_slots();
    }
  }

  flush_current_slot();

  while (outstanding > 0) {
    if (!process_any_done_slot()) {
      wait_for_slot_progress();
    }
  }
}

void ILSStage::expand_frontier(Puz& p, check_mode_t mode, bool strong, FitTester* ft,
                               BPPQ** new_frontiers, int q_prior, double gap, bool brute) {
  int s = p.getHeight();
  int k = p.getWidth();

  if (brute) {
    if (use_streaming_row_replacements()) {
      expand_frontier_GPU_row_replacements(p, s - 1, mode, strong, ft, new_frontiers,
                                            q_prior, gap, ALL);
    } else {
      vector<Puz> simplify_needed;
      append_row_replacements(simplify_needed, p, s - 1, k);
      expand_frontier_GPU(simplify_needed, mode, strong, ft, new_frontiers, q_prior,
                          gap, simplify_needed.size(), ALL);
    }

    // Break early if already have a result and are current.
    if (prev != NULL && numResult() != 0) {
      return;
    }

  }

  else {
    vector<int> idxs;
    if (mode != CHECK_FULL) {
      TDM tdm(p, strong, TDM::NO_GPU);
      if (mode == CHECK_OBVIOUS)
        tdm.simplify(TDM::EAGER);
      tdm.count(&idxs);
    }

    if (!idxs.empty()) {
      int r = idxs[lrand48() % idxs.size()];
      if (r < s - 1) {
        if (use_streaming_row_replacements()) {
          expand_frontier_GPU_row_replacements(p, r, mode, strong, ft, new_frontiers,
                                                q_prior, gap, REPL_ROW);
        } else {
          vector<Puz> simplify_needed;
          append_row_replacements(simplify_needed, p, r, k);
          expand_frontier_GPU(simplify_needed, mode, strong, ft, new_frontiers, q_prior,
                              gap, simplify_needed.size(), REPL_ROW);
        }

        // Break early if already have a result and are current.
        if (prev != NULL && numResult() != 0) {
          return;
        }
      }
    }

    // Explore Derivatives
    // Loop over rows.
    // Theta(s * k * Check(s,k))
    for (int r = 0; r < s; r++) {
      bool found = is_active(r, idxs);
      if (!found)
        continue;
      for (int c = 0; c < k; c++) {
        unsigned int curr = p.get(r, c);
        for (int dv = 1; dv <= 2; dv++) {
          int val = ((curr - 1 + dv) % 3) + 1;
          p.set(r, c, val);

          // Calculate fitness of new puzzle and add to
          // the local search frontier.
          if (add_to_frontier(p, ft, gap, new_frontiers, q_prior, DERV))
            return;
        }
        p.set(r, c, curr);
      }
    }

    // Explore permuting element values in row or column.
    // Theta((s+k) * Check(s,k))
    int perms[6][3] = {{0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

    for (int pi = 1; pi < 6; pi++) {
      for (int i = 0; i < s + k; i++) {
        bool is_col = (i >= s);
        int idx = (is_col ? i - s : i);
        Puz p_curr = p;
        for (int j = 0; j < (is_col ? s : k); j++) {
          int r = (is_col ? j : idx);
          int c = (is_col ? idx : j);
          int val = p_curr.get(r, c);
          p_curr.set(r, c, perms[pi][val - 1] + 1);
        }
        // Calculate fitness of new puzzle and add to
        // the local search frontier.
        if (add_to_frontier(p_curr, ft, gap, new_frontiers, q_prior, PERM_ELT))
          return;
      }
    }
  }
}

// Constructor for an initial stage.
ILSStage::ILSStage(const Puz& p, FitTester* ft) : ft(ft), prev(nullptr) {
  s = p.getHeight();
  k = p.getWidth();
  init_stats();
  stats.mode = ILS_SEARCH;
  work_queues = new BPMFQ(s * 1000, Q_LEVELS, Q_PRIORS);
  double fit, max_fit;
  tie(fit, max_fit) = ft->getFit(p);
  stats.evals++;
  work_queues->enqueue(q_elt_t{(fit - max_fit) - safe_rand(), p, INIT}, 0, 0);
}

// Constructor for a stage starting from a previous stage.
ILSStage::ILSStage(unsigned int s, unsigned int k, FitTester* ft, std::shared_ptr<ILSStage> prev)
    : s(s), k(k), ft(ft), prev(prev) {
  init_stats();
  stats.mode = ILS_SEARCH;
  work_queues = new BPMFQ(s * 1000, Q_LEVELS, Q_PRIORS);
}

ILSStage::~ILSStage() {
  delete work_queues;
}

unsigned int ILSStage::getHeight() {
  return s;
}

// Perform one unit of work on this stage.
bool ILSStage::doWork(check_mode_t mode, bool strong) {
  if (stats.mode == ILS_INACTIVE)
    cerr << "ERROR: STAGE IS INACTIVE" << '\n';
  ;
  assert(stats.mode != ILS_INACTIVE);
  bool working = false;
  Puz p;
  double gap;
  q_elt_t e;
  int q_level;
  int q_prior;

#pragma omp critical(queue)
  {
    bool success1 = false;
    if (prev != nullptr &&
        (work_queues->is_empty() || safe_rand() <= PROB_GET_PREV / (PROB_GET_PREV + PROB_CURR))) {
      Puz* p_ptr = prev->takeResult();
      if (p_ptr != nullptr) {
        working = true;
        success1 = true;
        p = *p_ptr;
        p = p.resizePuz(1);
        delete p_ptr;
        e.elt_type = INIT;
        q_level = -1;
        q_prior = 0;
        double fit, max_fit;
        tie(fit, max_fit) = ft->getFit(p);
        gap = fit - max_fit;
      }
    }
    bool success2 = false;
    if (!success1 && !work_queues->is_empty()) {
      working = true;
#pragma omp critical(stats)
      stats.working++;
      work_queues->dequeue(e, q_level, q_prior);
      gap = e.gap;
      p = e.p;
      success2 = true;
    }
  }

  if (!working)
    return false;

  bool seen = false;
#pragma omp critical(isomorph)
  seen = have_seen_isomorph(p, true);
  if (seen) {
#pragma omp critical(stats)
    {
      stats.repeated++;
      stats.working--;
    }
    return true;
  }

  gap = ceil(gap);
#pragma omp critical(stats)
  {
    if (stats.best_gap > 0 || stats.best_gap < gap) {
      stats.best_gap = gap;
      stats.since_progress = 0;
    }
  }

  if (gap == 0) {
#pragma omp critical(stats)
    {
      stats.max_found++;
      stats.mode = ILS_FEED;
    }
#pragma omp critical(results)
    results.push_back(p);
  }

  BPPQ* local_qs[Q_PRIORS];
  for (int i = 0; i < Q_PRIORS; i++)
    local_qs[i] = new BPPQ(s * 1000);

#pragma omp critical(stats)
  {
    stats.proc++;
    stats.since_progress++;
    stats.searched_types[e.elt_type]++;
  }

  // Expand the frontier to generate new puzzles.
  expand_frontier(p, mode, strong, ft, local_qs, q_prior, gap, e.elt_type == INIT);

#pragma omp critical(queue)
#pragma omp critical(stats)
  {
    update_queue(work_queues, local_qs, (q_level < Q_LEVELS - 1 ? q_level + 1 : Q_LEVELS - 1),
                 stats);
    stats.working--;
  }

  for (int i = 0; i < Q_PRIORS; i++)
    delete local_qs[i];

  return true;
}

// Called by the next stage: returns the next SUSP.
Puz* ILSStage::takeResult(bool pop) {
  Puz* res = nullptr;
#pragma omp critical(results)
  {
    if (!results.empty()) {
      res = new Puz(results.back());
      if (pop)
        results.pop_back();
    }
  }
  return res;
}

unsigned int ILSStage::numResult() {
  unsigned int res = 0;
#pragma omp critical(results)
  res = results.size();
  return res;
}

// Safely deactivates the previous stage.
void ILSStage::deactivatePrev() {
#pragma omp critical(queue)
  { prev.reset(); }
}

void ILSStage::try_restart() {
  work_queues->restart();
  stats.since_progress = 0;
  stats.best_gap = 1;
  stats.restarts++;
  cout << "\n------- Restart #" << stats.restarts << " -------" << '\n';
}

void ILSStage::display_stats(string counts_str) {
  usleep(200000);
#pragma omp critical(stats)
  {
    if (stats.proc != 0) {
      auto stop = chrono::high_resolution_clock::now();
      auto duration = chrono::duration<double>(stop - stats.start + stats.total_time);
      double time_secs = duration.count();
      cout << "\033[150A\r\033[J";
      work_queues->display();
      double eval_rate = stats.evals / time_secs;
      cout << "\n s: " << setw(3) << s << ", #: " << setw(6) << stats.proc << " (" << setw(6)
           << stats.since_progress << ", r: " << setw(3) << stats.restarts
           << "), Gap: " << scientific << setprecision(4) << stats.best_gap;
      cout << "\n Q: " << setw(7) << work_queues->size() << " [" << setprecision(4)
           << work_queues->best() << " .. " << setprecision(4) << work_queues->worst() << "]";
      cout << " (Prev Stages: " << counts_str << "), ";
      cout << "\n Distinct: " << setw(6) << get_num_isomorphs() << ", Repeated: " << setw(6)
           << stats.repeated;
      cout << "\n Eval: " << setprecision(4) << (double) (stats.evals) << " (per sec: " << fixed
           << setprecision(0) << eval_rate << "), ";
      auto num_gpu = TDM::getNumGPUSimplify();
      auto num_cpu = TDM::getNumSimplify();
      auto tot_simplify = num_gpu + num_cpu;
      static const auto live_start = chrono::high_resolution_clock::now();
      static const unsigned long long live_start_gpu = num_gpu;
      static const unsigned long long live_start_cpu = num_cpu;
      double live_time_secs = chrono::duration<double>(stop - live_start).count();
      unsigned long long live_gpu = (num_gpu >= live_start_gpu ? num_gpu - live_start_gpu : num_gpu);
      unsigned long long live_cpu = (num_cpu >= live_start_cpu ? num_cpu - live_start_cpu : num_cpu);
      unsigned long long live_tot_simplify = live_gpu + live_cpu;
      cout << "\n GPU Simplify: " << setw(8) << num_gpu << " (" << setw(3) << fixed
           << setprecision(0) << (tot_simplify ? 100.0 * num_gpu / tot_simplify : 0) << "%) (per sec: "
           << setw(8) << fixed << setprecision(2) << (time_secs ? num_gpu / time_secs : 0) << ")";
      cout << "\n CPU Simplify: " << setw(8) << num_cpu << " (" << setw(3) << fixed
           << setprecision(0) << (tot_simplify ? 100.0 * num_cpu / tot_simplify : 0) << "%) (per sec: "
           << setw(8) << fixed << setprecision(2) << (time_secs ? num_cpu / time_secs : 0) << ")";
      cout << "\n Tot Simplify: " << setw(8) << tot_simplify
           << " (100%) (per sec: " << setw(8) << fixed << setprecision(2)
           << (time_secs ? tot_simplify / time_secs : 0) << ")";
      cout << "\n Live Simplify: " << setw(8) << live_tot_simplify
           << " (GPU " << live_gpu << ", CPU " << live_cpu << ") (per sec: "
           << setw(8) << fixed << setprecision(2)
           << (live_time_secs ? live_tot_simplify / live_time_secs : 0) << ")";
      cout << "\n Total Time: " << fixed << setprecision(2) << time_secs << " secs";
      if (stats.searched_types[0] != 0)
        cerr << "ERROR: searched_types[0] != 0\n";
      assert(stats.searched_types[0] == 0);
      cout << "\n ST: [INIT:" << setw(4) << stats.searched_types[1] << ", DERV:" << setw(4)
           << stats.searched_types[2] << ", RAND:" << setw(4) << stats.searched_types[3]
           << ", PERM_ELT:" << setw(4) << stats.searched_types[4] << ", PERM_ROW:" << setw(4)
           << stats.searched_types[5] << ", ALL:" << setw(4) << stats.searched_types[6]
           << ", COL_SWAP:" << setw(4) << stats.searched_types[7] << ", REPL_ROW:" << setw(4)
           << stats.searched_types[8] << "]" << '\n';
      fflush(stdout);
    }
  }
}

void ILSStage::serialize(std::ostream& os) const {
  stats.serialize(os);
  work_queues->serialize(os);
  size_t res_size = results.size();
  os.write(reinterpret_cast<const char*>(&s), sizeof(s));
  os.write(reinterpret_cast<const char*>(&k), sizeof(k));
  os.write(reinterpret_cast<const char*>(&res_size), sizeof(res_size));
  for (const auto& p : results) {
    p.serialize(os);
  }
  // Serialize additional data
  bool has_prev = (prev != nullptr);
  os.write(reinterpret_cast<const char*>(&has_prev), sizeof(has_prev));
}

void ILSStage::deserialize(std::istream& is) {
  stats.deserialize(is);
  work_queues->deserialize(is);
  size_t res_size;
  is.read(reinterpret_cast<char*>(&s), sizeof(s));
  is.read(reinterpret_cast<char*>(&k), sizeof(k));
  is.read(reinterpret_cast<char*>(&res_size), sizeof(res_size));
  results.clear();
  results.reserve(res_size);
  for (size_t i = 0; i < res_size; i++) {
    Puz p;
    p.deserialize(is);
    results.push_back(p);
  }
  // Deserialize additional data
  bool has_prev;
  is.read(reinterpret_cast<char*>(&has_prev), sizeof(has_prev));
  prev = nullptr;
}

void ILSStage::setPrev(std::shared_ptr<ILSStage> p) {
  prev = p;
}
