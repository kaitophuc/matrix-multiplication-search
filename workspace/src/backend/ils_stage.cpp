#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <strings.h>
#include <omp.h>
#include <queue>
#include <deque>
#include <memory>
#include <assert.h>
#include <stdexcept>
#include <sstream>
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

extern std::vector<std::unique_ptr<GpuTask> > gpuTaskSlots;
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

unsigned int ILSStage::crop_prior(int q_prior) {
  if (q_prior < 0)
    return 0;
  if (q_prior >= Q_PRIORS)
    return Q_PRIORS - 1;
  return (unsigned int) q_prior;
}

bool ILSStage::add_to_frontier(const Puz& p, FitTester* ft, double gap, BPPQ** new_frontiers,
                               unsigned int q_prior, int type, int pair_failures) {
  double curr_fit, max_fit;
  tie(curr_fit, max_fit) = ft->getFitWithPairFailures(p, pair_failures);
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

bool ILSStage::expand_frontier_GPU(vector<Puz>& puzzles, FitTester* ft, BPPQ** new_frontiers,
                                   int q_prior, double gap, elt_type type) {
  if (puzzles.empty())
    return false;

  const int worker_id = omp_in_parallel() ? omp_get_thread_num() : 0;
  const int slots_per_worker = std::max(2, gpuSlotsPerWorker);
  const int first_slot_id = slots_per_worker * worker_id;
  if (first_slot_id + slots_per_worker > (int) gpuTaskSlots.size())
    throw std::runtime_error("OpenMP worker does not have GPU task slots.");

  std::deque<GpuTask*> outstanding;
  const bool stop_if_stage_solved = type == ALL || type == REPL_ROW;
  bool stop_processing = false;

  // In each neighborhood, admit candidates in the original generation order.
  // A later slot can finish first, but only the first successful candidate stops the expansion.
  auto drain_next = [&] {
    GpuTask& slot = *outstanding.front();
    slot.waitForState(GpuTaskState::Done);
    if (!stop_processing) {
      slot.copyResultsToPuzzles();
      for (int i = 0; i < slot.batch_size; ++i) {
        if (add_to_frontier(slot.puzzles[i], ft, slot.gap, new_frontiers, slot.q_prior,
                            slot.frontier_type, slot.pairFailures(i)) ||
            (stop_if_stage_solved && prev != nullptr && numResult() != 0)) {
          stop_processing = true;
          break;
        }
      }
    }
    slot.markEmpty();
    outstanding.pop_front();
  };

  int next_slot = 0;
  for (size_t offset = 0; offset < puzzles.size() && !stop_processing;) {
    if ((int) outstanding.size() == slots_per_worker)
      drain_next();
    if (stop_processing)
      break;

    GpuTask& slot = *gpuTaskSlots[first_slot_id + next_slot];
    if (!slot.tryBeginFilling(worker_id, s, k, q_prior, gap, (int) type))
      throw std::logic_error("GPU neighborhood slot is still in use.");
    const size_t count = std::min((size_t) TRIALS, puzzles.size() - offset);
    for (size_t i = 0; i < count; ++i)
      slot.appendPuzzle(std::move(puzzles[offset + i]));
    submitGpuTask(slot);
    outstanding.push_back(&slot);
    offset += count;
    next_slot = (next_slot + 1) % slots_per_worker;
  }
  while (!outstanding.empty())
    drain_next();
  return stop_processing;
}

void ILSStage::expand_frontier_GPU_row_replacements(const Puz& base, int row_to_replace,
                                                    check_mode_t mode, bool strong,
                                                    FitTester* ft, BPPQ** new_frontiers,
                                                    int q_prior, double gap, elt_type type) {
  const unsigned long long num_rows = num_rows_for_width(k);
  if (num_rows == 0) {
    return;
  }

  if (gpuTaskSlots.empty()) {
    throw std::runtime_error("GPU task slots are not initialized.");
  }

  int worker_id = omp_in_parallel() ? omp_get_thread_num() : 0;
  int slots_per_worker = std::max(2, gpuSlotsPerWorker);
  int first_slot_id = slots_per_worker * worker_id;
  int last_slot_id = first_slot_id + slots_per_worker;
  if (last_slot_id > (int) gpuTaskSlots.size()) {
    throw std::runtime_error("OpenMP worker does not have GPU task slots.");
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
                            slot.frontier_type, slot.pairFailures(puz))) {
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
    for (int slot_id = first_slot_id; slot_id < last_slot_id; ++slot_id) {
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
    for (int slot_id = first_slot_id; slot_id < last_slot_id; ++slot_id) {
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

      for (int slot_id = first_slot_id; slot_id < last_slot_id; ++slot_id) {
        GpuTask& slot = *gpuTaskSlots[slot_id];
        if (slot.tryBeginFilling(worker_id, s, k, q_prior, gap, (int) type)) {
          return &slot;
        }
      }

      bool drained_slot = false;
      for (int slot_id = first_slot_id; slot_id < last_slot_id; ++slot_id) {
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

  GpuTask* current_slot = nullptr;
  auto flush_current_slot = [&]() {
    if (current_slot != nullptr && current_slot->batch_size > 0) {
      submitGpuTask(*current_slot);
      outstanding++;
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
      expand_frontier_GPU(simplify_needed, ft, new_frontiers, q_prior, gap, ALL);
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
          expand_frontier_GPU(simplify_needed, ft, new_frontiers, q_prior, gap, REPL_ROW);
        }

        // Break early if already have a result and are current.
        if (prev != NULL && numResult() != 0) {
          return;
        }
      }
    }

    const bool batch_neighbors = mode == CHECK_OBVIOUS && strong && GpuResources::uses_packed_tdm(s, k);
    vector<Puz> neighbors;
    if (batch_neighbors)
      neighbors.reserve(std::max(2 * s * k, 5 * (s + k)));

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
          if (batch_neighbors) {
            neighbors.push_back(p);
          } else if (add_to_frontier(p, ft, gap, new_frontiers, q_prior, DERV)) {
            return;
          }
        }
        p.set(r, c, curr);
      }
    }

    if (batch_neighbors) {
      if (expand_frontier_GPU(neighbors, ft, new_frontiers, q_prior, gap, DERV))
        return;
      neighbors.clear();
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
        if (batch_neighbors) {
          neighbors.push_back(std::move(p_curr));
        } else if (add_to_frontier(p_curr, ft, gap, new_frontiers, q_prior, PERM_ELT)) {
          return;
        }
      }
    }
    if (batch_neighbors)
      expand_frontier_GPU(neighbors, ft, new_frontiers, q_prior, gap, PERM_ELT);
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
  std::unique_ptr<Puz> previous_result;
  Puz p;
  double gap;
  q_elt_t e;
  int q_level;
  int q_prior;

  {
    std::lock_guard<std::mutex> lock(frontier_mutex);
    if (prev != nullptr &&
        (work_queues->is_empty() || safe_rand() <= PROB_GET_PREV / (PROB_GET_PREV + PROB_CURR))) {
      previous_result.reset(prev->takeResult());
    }
    if (!previous_result && !work_queues->dequeue(e, q_level, q_prior)) {
      return false;
    }
  }

  if (previous_result) {
    p = previous_result->resizePuz(1);
    e.elt_type = INIT;
    q_level = -1;
    q_prior = 0;
    auto [fit, max_fit] = ft->getFit(p);
    gap = fit - max_fit;
  } else {
    gap = e.gap;
    p = std::move(e.p);
  }

#pragma omp critical(stats)
  {
    assert(stats.mode != ILS_INACTIVE);
    stats.working++;
  }

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
    {
      results.push_back(p);
      result_count.store(results.size(), std::memory_order_release);
    }
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

  size_t candidates = 0;
  for (const auto* local_queue : local_qs) {
    candidates += local_queue->size();
  }
  size_t enqueued;
  {
    std::lock_guard<std::mutex> lock(frontier_mutex);
    enqueued = work_queues->merge_from(local_qs, std::min(q_level + 1, Q_LEVELS - 1));
  }
#pragma omp critical(stats)
  {
    stats.dropped += candidates - enqueued;
    stats.evals += candidates;
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
      if (pop) {
        results.pop_back();
        result_count.store(results.size(), std::memory_order_release);
      }
    }
  }
  return res;
}

unsigned int ILSStage::numResult() {
  return result_count.load(std::memory_order_acquire);
}

// Safely deactivates the previous stage.
void ILSStage::deactivatePrev() {
  std::lock_guard<std::mutex> lock(frontier_mutex);
  prev.reset();
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
  stats_t snapshot;
#pragma omp critical(stats)
  { snapshot = stats; }
  if (snapshot.proc == 0) {
    return;
  }
  std::ostringstream queue_details;
  size_t queue_size;
  double queue_best;
  double queue_worst;
  {
    std::lock_guard<std::mutex> lock(frontier_mutex);
    work_queues->display(queue_details);
    queue_size = work_queues->size();
    queue_best = work_queues->best();
    queue_worst = work_queues->worst();
  }
  size_t distinct;
#pragma omp critical(isomorph)
  { distinct = get_num_isomorphs(); }
  const auto& stats = snapshot;
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration<double>(stop - stats.start + stats.total_time);
  double time_secs = duration.count();
  cout << "\033[150A\r\033[J";
  cout << queue_details.str();
  double eval_rate = stats.evals / time_secs;
  cout << "\n s: " << setw(3) << s << ", #: " << setw(6) << stats.proc << " (" << setw(6)
       << stats.since_progress << ", r: " << setw(3) << stats.restarts
       << "), Gap: " << scientific << setprecision(4) << stats.best_gap;
  cout << "\n Q: " << setw(7) << queue_size << " [" << setprecision(4)
       << queue_best << " .. " << setprecision(4) << queue_worst << "]";
  cout << " (Prev Stages: " << counts_str << "), ";
  cout << "\n Distinct: " << setw(6) << distinct << ", Repeated: " << setw(6)
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
  result_count.store(results.size(), std::memory_order_release);
  // Deserialize additional data
  bool has_prev;
  is.read(reinterpret_cast<char*>(&has_prev), sizeof(has_prev));
  prev = nullptr;
}

void ILSStage::setPrev(std::shared_ptr<ILSStage> p) {
  prev = p;
}
