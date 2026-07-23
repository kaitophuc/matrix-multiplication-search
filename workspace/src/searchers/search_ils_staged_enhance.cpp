/*
 * Iterated local search of USP and SUSP.
 * Currently fastest search algorithm as of Spring '22.
 *
 * Author: Matt Anderson, Minh Phuc Nguyen
 *
 * Key Features:
 * - Multi-threaded CPU and GPU processing for high performance.
 * - Pinned memory for efficient data transfer between CPU and GPU.
 * - Centralized memory management using the GpuResources class.
 * - Enhanced TDM simplification using CUDA kernels.
 * - Support for multiple GPU workers to scale across devices.
 *
 * Technical Details:
 * - The program uses OpenMP for multi-threading and CUDA for GPU acceleration.
 * - GPU tasks are managed through a global queue, with worker threads processing tasks
 * asynchronously.
 * - The GpuResources class handles memory allocation and synchronization for double buffering.
 * - The ILS algorithm expands the search frontier by generating new puzzles and evaluating their
 * fitness.
 * - Results are logged to a file for further analysis.
 * - Bloom filtering is optionally used to same memory in isomorph checks.
 * - To use bloom filtering, specify --bloom=<expected>,<fpr> on the command line.
 * Example: --bloom=100000000,0.01
 */

// drain the GPU worker threads when exiting

#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <atomic>
#include <tuple>
#include <set>
#include <cstring>
#include <deque>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <strings.h>
#include <time.h>
#include <curl/curl.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <memory>
#include <thread>
#include <signal.h>
#include <stdexcept>
#include <filesystem>

#include "Puz.h"
#include "BoundedPuzPriorityQueue.h"
#include "special.h"
#include "checker.h"
#include "fitness.h"
#include "canonization.h"
#include "TDM_cuda_enhanced.h"
#include "GpuTask.h"

using namespace std;

#include "elt_type.h"
#include "stats.h"
#include "ils_stage.h"
#include "search.h"

#include "utils.h"

// Macros that depend on variable s
#define RESTART_THRESH (g_search.s * 500)
#define MAX_RESTARTS 1000
#define MAX_RESULTS 20
#define Q_SIZE (g_search.s * 1000)
#define Q_LEVELS 5
#define Q_PRIORS 5
#define REQUIRE_SPECIAL 0
#define RANDOM_FRONTIER 10 * (((double) stats.since_progress / 1000.0 + 1))

#define NUM_STAGES 4

// Loading Probabilities
#define PROB_PREV 0.3
#define PROB_GET_PREV 0.005
#define PROB_CURR (1.0 - PROB_GET_PREV - PROB_PREV)

using BPPQ = BoundedPuzPriorityQueue;
using BPMFQ = BoundedPuzMultilevelFeedbackQueue;

/* Global GPU slot queues */
std::vector<std::unique_ptr<GpuTaskIdQueue>> gpuTaskQueues;
std::atomic<unsigned int> nextGpuTaskQueue(0);
std::vector<std::unique_ptr<GpuTask>> gpuTaskSlots;
std::mutex gpuQueueMutex;
std::condition_variable gpuQueueCv;
std::atomic<bool> stopGpuWorker(false);
std::atomic<bool> should_checkpoint(false);
std::atomic<bool> should_exit_after_checkpoint(false);

/* GPU resources and coordinator */
int deviceCount;
int numGpuDevicesUsed;
int numGpuWorkers;
int gpuLanesPerDevice = 1;
vector<std::unique_ptr<GpuResources>> gpu_resources;
std::vector<std::thread> gpuCoordinatorThreads;
std::vector<unsigned long long> gpuLaunchesCompletedByWorker;
std::vector<unsigned long long> gpuBatchesCompletedByWorker;
std::vector<unsigned long long> gpuPuzzlesCompletedByWorker;
int gpuLaunchSlotBatch = 1;
int gpuLaunchMaxTrials = TRIALS;
int gpuSlotsPerWorker = 2;
int gpuCoalesceWaitUs = 0;

Search g_search;
std::filesystem::path g_repo_root;
static const long long DEFAULT_AUTO_CHECKPOINT_SECONDS = 6LL * 60LL * 60LL;
std::chrono::seconds g_auto_checkpoint_interval(DEFAULT_AUTO_CHECKPOINT_SECONDS);
std::chrono::steady_clock::time_point g_next_auto_checkpoint;
static bool g_bloom_enable = false;
static size_t g_bloom_expected = 100000000;
static double g_bloom_false_positive_rate = 0.01;

bool parse_bloom_option(const std::string& arg, size_t& expected, double& fpr) {
  if (arg == "--bloom") {
    expected = g_bloom_expected;
    fpr = g_bloom_false_positive_rate;
    return true;
  }

  const std::string prefix = "--bloom=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }

  std::string value = arg.substr(prefix.size());
  size_t comma = value.find(',');
  if (comma == std::string::npos) {
    return false;
  }

  std::string expected_str = value.substr(0, comma);
  std::string fpr_str = value.substr(comma + 1);
  if (expected_str.empty() || fpr_str.empty()) {
    return false;
  }

  char* end = nullptr;
  errno = 0;
  unsigned long long expected_ll = strtoull(expected_str.c_str(), &end, 10);
  if (errno != 0 || end == expected_str.c_str() || *end != '\0') {
    return false;
  }

  errno = 0;
  char* end2 = nullptr;
  double fpr_val = strtod(fpr_str.c_str(), &end2);
  if (errno != 0 || end2 == fpr_str.c_str() || *end2 != '\0') {
    return false;
  }

  expected = static_cast<size_t>(expected_ll);
  fpr = fpr_val;
  return true;
}

int parse_positive_env(const char* name, int default_value) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  int parsed = atoi(value);
  return parsed > 0 ? parsed : default_value;
}

long long parse_nonnegative_env(const char* name, long long default_value) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  errno = 0;
  char* end = nullptr;
  long long parsed = strtoll(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0) {
    return default_value;
  }

  return parsed;
}

std::filesystem::path repo_root_from_executable(const char* argv0) {
  namespace fs = std::filesystem;

  fs::path exe_path = argv0;
  if (exe_path.is_relative()) {
    exe_path = fs::current_path() / exe_path;
  }

  std::error_code ec;
  fs::path canonical_path = fs::weakly_canonical(exe_path, ec);
  if (ec) {
    canonical_path = fs::absolute(exe_path, ec);
  }
  if (ec) {
    return fs::current_path();
  }

  fs::path exe_dir = canonical_path.parent_path();
  if (exe_dir.filename() == "bin") {
    return exe_dir.parent_path();
  }
  if (fs::exists(exe_dir / "workspace")) {
    return exe_dir;
  }
  return fs::current_path();
}

bool auto_checkpoint_enabled() {
  return g_auto_checkpoint_interval.count() > 0;
}

void schedule_next_auto_checkpoint(std::chrono::steady_clock::time_point from) {
  if (!auto_checkpoint_enabled()) {
    return;
  }

  g_next_auto_checkpoint = from + g_auto_checkpoint_interval;
}

void maybe_request_auto_checkpoint() {
  if (!auto_checkpoint_enabled()) {
    return;
  }

  if (std::chrono::steady_clock::now() >= g_next_auto_checkpoint) {
    should_checkpoint.store(true, std::memory_order_release);
  }
}

std::filesystem::path make_checkpoint_path() {
  auto now = chrono::system_clock::now();
  auto time_t = chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
  localtime_r(&time_t, &tm_buf);
  std::stringstream ss;
  ss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
  return g_repo_root / "data" / ("checkpoint_" + ss.str() + ".dat");
}

struct CheckpointProbe {
  std::filesystem::path path;
  check_mode_t mode = CHECK_FULL;
  bool strong = false;
  int s_target = 0;
  int search_s = 0;
  int k = 0;
  unsigned int puzzle_s = 0;
  unsigned int puzzle_k = 0;
  std::filesystem::file_time_type modified = std::filesystem::file_time_type::min();
};

bool read_checkpoint_probe(const std::filesystem::path& checkpoint_path, CheckpointProbe& probe) {
  std::ifstream is(checkpoint_path, std::ios::binary);
  if (!is) {
    return false;
  }

  probe.path = checkpoint_path;
  is.read(reinterpret_cast<char*>(&probe.mode), sizeof(probe.mode));
  is.read(reinterpret_cast<char*>(&probe.strong), sizeof(probe.strong));
  is.read(reinterpret_cast<char*>(&probe.s_target), sizeof(probe.s_target));
  is.read(reinterpret_cast<char*>(&probe.search_s), sizeof(probe.search_s));
  is.read(reinterpret_cast<char*>(&probe.k), sizeof(probe.k));
  is.read(reinterpret_cast<char*>(&probe.puzzle_s), sizeof(probe.puzzle_s));
  is.read(reinterpret_cast<char*>(&probe.puzzle_k), sizeof(probe.puzzle_k));
  if (!is) {
    return false;
  }

  std::error_code ec;
  probe.modified = std::filesystem::last_write_time(checkpoint_path, ec);
  if (ec) {
    probe.modified = std::filesystem::file_time_type::min();
  }
  return true;
}

bool find_auto_start_checkpoint(check_mode_t requested_mode, bool requested_strong,
                                int requested_s_target, int requested_k,
                                CheckpointProbe& best_match) {
  namespace fs = std::filesystem;

  if (requested_s_target <= 1) {
    return false;
  }

  fs::path data_dir = g_repo_root / "data";
  std::error_code ec;
  if (!fs::is_directory(data_dir, ec)) {
    return false;
  }

  bool found = false;
  for (fs::directory_iterator it(data_dir, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code entry_ec;
    const fs::directory_entry& entry = *it;
    if (!entry.is_regular_file(entry_ec) || entry.path().extension() != ".dat") {
      continue;
    }

    CheckpointProbe probe;
    if (!read_checkpoint_probe(entry.path(), probe)) {
      continue;
    }
    if (probe.mode != requested_mode || probe.strong != requested_strong) {
      continue;
    }
    if (probe.k != requested_k || static_cast<int>(probe.puzzle_k) != requested_k) {
      continue;
    }
    if (probe.search_s != requested_s_target ||
        static_cast<int>(probe.puzzle_s) != requested_s_target - 1) {
      continue;
    }

    if (!found || probe.modified > best_match.modified) {
      best_match = probe;
      found = true;
    }
  }

  return found;
}

size_t gpu_bytes_per_trial(int s, int k) {
  const unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
  return sizeof(e_type) * (size_t) s * entries_per_row +
         sizeof(uint8_t) * (size_t) s * (size_t) s * (size_t) s + sizeof(int) + sizeof(int) +
         sizeof(int) * (size_t) s + sizeof(int) + sizeof(uint8_t) * (size_t) s * (size_t) s;
}

int choose_gpu_launch_max_trials(int s, int k, int requested_slots) {
  int requested_trials = std::max(1, requested_slots) * TRIALS;
  int max_batch_mb = parse_positive_env("ILS_GPU_MAX_BATCH_MB", 512);
  size_t cap_bytes = (size_t) max_batch_mb * 1024u * 1024u;
  size_t per_trial = std::max<size_t>(1, gpu_bytes_per_trial(s, k));
  int cap_trials = (int) (cap_bytes / per_trial);
  cap_trials = std::max(TRIALS, cap_trials);
  cap_trials = (cap_trials / TRIALS) * TRIALS;
  cap_trials = std::max(TRIALS, cap_trials);
  return std::max(TRIALS, std::min(requested_trials, cap_trials));
}

GpuTask* try_claim_gpu_task(int slot_id) {
  if (slot_id < 0 || slot_id >= (int) gpuTaskSlots.size() || gpuTaskSlots[slot_id] == nullptr) {
    return nullptr;
  }

  GpuTask& task = *gpuTaskSlots[slot_id];
  if (!task.markProcessing()) {
    return nullptr;
  }
  return &task;
}

bool try_take_claimed_gpu_task_from_queue(GpuTaskIdQueue& queue, GpuTask*& task) {
  int slot_id = -1;
  while (queue.try_pop(slot_id)) {
    task = try_claim_gpu_task(slot_id);
    if (task != nullptr) {
      return true;
    }
  }

  task = nullptr;
  return false;
}

bool any_gpu_task_queue_has_work() {
  for (const auto& queue : gpuTaskQueues) {
    if (queue != nullptr && !queue->empty()) {
      return true;
    }
  }
  return false;
}

bool take_claimed_gpu_task(int preferred_queue_id, std::deque<GpuTask*>& deferred_tasks,
                           GpuTask*& task) {
  if (!deferred_tasks.empty()) {
    task = deferred_tasks.front();
    deferred_tasks.pop_front();
    return true;
  }

  const size_t queue_count = gpuTaskQueues.size();
  for (size_t attempt = 0; attempt < queue_count; attempt++) {
    size_t queue_id = ((size_t) preferred_queue_id + attempt) % queue_count;
    if (gpuTaskQueues[queue_id] == nullptr) {
      continue;
    }
    if (try_take_claimed_gpu_task_from_queue(*gpuTaskQueues[queue_id], task)) {
      return true;
    }
  }

  task = nullptr;
  return false;
}

void solve_gpu_task_group(int workerId, int deviceId, GpuResources& gpu_resources,
                          const std::vector<GpuTask*>& tasks) {
  if (tasks.empty()) {
    return;
  }

  const int s = tasks.front()->s;
  const int k = tasks.front()->k;
  const unsigned int entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
  const size_t puzzle_entries_per_trial = (size_t) s * entries_per_row;
  int total_trials = 0;

  for (GpuTask* task : tasks) {
    if (task->s != s || task->k != k) {
      throw std::logic_error("Coalesced GPU task group contains mixed puzzle dimensions.");
    }
    if (task->batch_size <= 0) {
      throw std::logic_error("Coalesced GPU task group contains an empty task.");
    }
    total_trials += task->batch_size;
  }

  if (total_trials > gpu_resources.max_trials) {
    throw std::logic_error("Coalesced GPU task group exceeds GpuResources capacity.");
  }

  if (deviceId >= 0) {
    cudaError_t set_status = cudaSetDevice(deviceId);
    if (set_status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaSetDevice failed before coalesced task copy: ") +
                               cudaGetErrorString(set_status));
    }
  }

  size_t puzzle_offset = 0;
  for (GpuTask* task : tasks) {
    size_t task_entries = puzzle_entries_per_trial * (size_t) task->batch_size;
    cudaError_t copy_status = cudaMemcpyAsync(
        gpu_resources.dev_Puz + puzzle_offset, task->puz_pinned_buffer,
        task_entries * sizeof(e_type), cudaMemcpyHostToDevice, gpu_resources.stream);
    if (copy_status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpyAsync failed for coalesced GPU task copy: ") +
                               cudaGetErrorString(copy_status));
    }
    puzzle_offset += task_entries;
  }

  outer_KernelDeviceInput(gpu_resources, gpu_resources.results_pinned_buffer, s, k, total_trials);

  if (workerId >= 0 && workerId < (int) gpuLaunchesCompletedByWorker.size()) {
    gpuLaunchesCompletedByWorker[workerId]++;
  }

  int result_offset = 0;
  for (GpuTask* task : tasks) {
    memcpy(task->results_pinned_buffer, gpu_resources.results_pinned_buffer + result_offset,
           (size_t) task->batch_size * sizeof(int));
    result_offset += task->batch_size;
    task->markDone();

    if (workerId >= 0 && workerId < (int) gpuBatchesCompletedByWorker.size()) {
      gpuBatchesCompletedByWorker[workerId]++;
      gpuPuzzlesCompletedByWorker[workerId] += (unsigned long long) task->batch_size;
    }
  }
}

void gpuCoordinator(int queueId, int deviceId, GpuResources& gpu_resources) {
  cudaSetDevice(deviceId);
  enhanced_Hcheck_cuda_error("Starting GPU coordinator " + std::to_string(deviceId));

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  enhanced_Hcheck_cuda_error("Setting cache config");

  std::deque<GpuTask*> deferred_tasks;

  while (true) {
    GpuTask* first_task = nullptr;
    if (!take_claimed_gpu_task(queueId, deferred_tasks, first_task)) {
      if (stopGpuWorker.load(std::memory_order_acquire) && !any_gpu_task_queue_has_work()) {
        break;
      }

      std::unique_lock<std::mutex> lock(gpuQueueMutex);
      gpuQueueCv.wait(lock, [&] {
        return stopGpuWorker.load(std::memory_order_acquire) || any_gpu_task_queue_has_work() ||
               !deferred_tasks.empty();
      });
      continue;
    }

    std::vector<GpuTask*> tasks;
    tasks.reserve((size_t) gpuLaunchSlotBatch);
    tasks.push_back(first_task);

    int total_trials = first_task->batch_size;
    const int group_s = first_task->s;
    const int group_k = first_task->k;
    auto coalesce_deadline =
        std::chrono::steady_clock::now() + std::chrono::microseconds(gpuCoalesceWaitUs);

    while ((int) tasks.size() < gpuLaunchSlotBatch && total_trials < gpu_resources.max_trials) {
      GpuTask* next_task = nullptr;
      if (!take_claimed_gpu_task(queueId, deferred_tasks, next_task)) {
        if (gpuCoalesceWaitUs <= 0 || stopGpuWorker.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= coalesce_deadline) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }

      if (next_task->s != group_s || next_task->k != group_k ||
          total_trials + next_task->batch_size > gpu_resources.max_trials) {
        deferred_tasks.push_front(next_task);
        break;
      }

      tasks.push_back(next_task);
      total_trials += next_task->batch_size;
    }

    solve_gpu_task_group(queueId, deviceId, gpu_resources, tasks);
    if (any_gpu_task_queue_has_work()) {
      gpuQueueCv.notify_all();
    }
  }
}

void initializeGpuTaskSlots(int cpu_workers, int s_target, int k, int slots_per_worker,
                            int queue_count) {
  gpuSlotsPerWorker = std::max(2, slots_per_worker);
  int slot_count = std::max(1, cpu_workers) * gpuSlotsPerWorker;
  gpuTaskSlots.clear();
  gpuTaskSlots.reserve(slot_count);
  for (int slot_id = 0; slot_id < slot_count; slot_id++) {
    int owner_worker_id = slot_id / gpuSlotsPerWorker;
    auto slot = std::make_unique<GpuTask>();
    slot->allocatePinned(s_target, k, TRIALS, slot_id, owner_worker_id);
    gpuTaskSlots.push_back(std::move(slot));
  }
  gpuTaskQueues.clear();
  gpuTaskQueues.reserve((size_t) queue_count);
  for (int queue_id = 0; queue_id < std::max(1, queue_count); queue_id++) {
    gpuTaskQueues.push_back(std::make_unique<GpuTaskIdQueue>((size_t) slot_count));
  }
  nextGpuTaskQueue.store(0, std::memory_order_release);
}

void cleanupGpuPipeline() {
  stopGpuWorker.store(true, std::memory_order_release);
  gpuQueueCv.notify_all();

  for (std::thread& coordinator : gpuCoordinatorThreads) {
    if (coordinator.joinable()) {
      coordinator.join();
    }
  }
  gpuCoordinatorThreads.clear();

  std::vector<unsigned long long> launches_by_device((size_t) numGpuDevicesUsed, 0);
  std::vector<unsigned long long> batches_by_device((size_t) numGpuDevicesUsed, 0);
  std::vector<unsigned long long> puzzles_by_device((size_t) numGpuDevicesUsed, 0);
  for (int worker_id = 0; worker_id < numGpuWorkers; worker_id++) {
    if (worker_id >= (int) gpu_resources.size() || !gpu_resources[worker_id]) {
      continue;
    }
    int device_id = gpu_resources[worker_id]->device_id;
    if (device_id < 0 || device_id >= numGpuDevicesUsed) {
      continue;
    }
    if (worker_id < (int) gpuLaunchesCompletedByWorker.size()) {
      launches_by_device[device_id] += gpuLaunchesCompletedByWorker[worker_id];
      batches_by_device[device_id] += gpuBatchesCompletedByWorker[worker_id];
      puzzles_by_device[device_id] += gpuPuzzlesCompletedByWorker[worker_id];
    }
  }

  for (int i = 0; i < numGpuDevicesUsed; i++) {
    double avg_puzzles_per_launch = 0.0;
    if (launches_by_device[i] != 0) {
      avg_puzzles_per_launch = (double) puzzles_by_device[i] / (double) launches_by_device[i];
    }
    cout << "GPU device " << i << " processed " << batches_by_device[i] << " task batch"
         << (batches_by_device[i] == 1 ? "" : "es") << " in " << launches_by_device[i]
         << " CUDA launch" << (launches_by_device[i] == 1 ? "" : "es") << " ("
         << puzzles_by_device[i] << " puzzles, avg " << std::fixed << std::setprecision(1)
         << avg_puzzles_per_launch << " puzzles/launch)." << '\n';
  }

  gpuTaskQueues.clear();
  gpuTaskSlots.clear();

  for (auto& resources : gpu_resources) {
    if (resources) {
      resources.reset();
    }
  }
  gpu_resources.clear();
}

unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
  a = a - b;
  a = a - c;
  a = a ^ (c >> 13);
  b = b - c;
  b = b - a;
  b = b ^ (a << 8);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 13);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 12);
  b = b - c;
  b = b - a;
  b = b ^ (a << 16);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 5);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 3);
  b = b - c;
  b = b - a;
  b = b ^ (a << 10);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 15);
  return c;
}

bool save_checkpoint(const string& filename) {
  namespace fs = std::filesystem;

  fs::path checkpoint_path(filename);
  if (checkpoint_path.has_parent_path()) {
    std::error_code ec;
    fs::create_directories(checkpoint_path.parent_path(), ec);
    if (ec) {
      cerr << "Error: Unable to create checkpoint directory '"
           << checkpoint_path.parent_path().string() << "': " << ec.message() << '\n';
      return false;
    }
  }

  fs::path temp_path = checkpoint_path;
  temp_path += ".tmp";
  ofstream os(temp_path, ios::binary | ios::trunc);
  if (!os) {
    cerr << "Error: Unable to open file '" << temp_path.string() << "' for writing checkpoint.\n";
    return false;
  }
  g_search.serialize(os);
  os.close();
  if (!os) {
    cerr << "Error: Unable to finish writing checkpoint '" << temp_path.string() << "'.\n";
    return false;
  }

  std::error_code ec;
  fs::rename(temp_path, checkpoint_path, ec);
  if (ec) {
    cerr << "Error: Unable to move temporary checkpoint '" << temp_path.string() << "' to '"
         << checkpoint_path.string() << "': " << ec.message() << '\n';
    return false;
  }

  cout << "Checkpoint saved to: " << filename << '\n';
  return true;
}

void restore_checkpoint(const string& filename) {
  ifstream is(filename, ios::binary);
  if (!is) {
    cerr << "Error: Unable to open file '" << filename << "' for reading checkpoint.\n";
    return;
  }
  g_search.deserialize(is);
  is.close();
  cout << "Checkpoint restored from: " << filename << '\n';
}

bool search_ils(bool is_resume = false) {
  if (!is_resume) {
    reset_isomorphs();
    if (g_bloom_enable) {
      enable_bloom_filtering(g_bloom_expected, g_bloom_false_positive_rate);
    } else {
      disable_bloom_filtering();
    }

    if (g_search.mode == CHECK_FULL)
      g_search.ft = new SubsetFitTester(g_search.strong);
    else
      g_search.ft = new TDMSizeFitTester(g_search.mode, g_search.strong, REQUIRE_SPECIAL);

    g_search.stages.resize(g_search.num_stages, nullptr);
    g_search.stages[g_search.curr] = make_shared<ILSStage>(g_search.p, g_search.ft);
  } else {
    if (g_search.mode == CHECK_FULL)
      g_search.ft = new SubsetFitTester(g_search.strong);
    else
      g_search.ft = new TDMSizeFitTester(g_search.mode, g_search.strong, REQUIRE_SPECIAL);

    if (g_search.stages.size() != g_search.num_stages) {
      cerr << "ERROR: Stage count mismatch after checkpoint restore!\n";
      return false;
    }
    for (int i = 0; i < g_search.num_stages; i++) {
      if (g_search.stages[i] != nullptr) {
        g_search.stages[i]->ft = g_search.ft;
      }
    }

    cout << "Resuming search with loaded checkpoint data:\n";
    cout << "- FitTester: " << g_search.ft->getName() << "\n";
    cout << "- Isomorphs: " << g_search.seen_isomorphs.size() << " cached\n";
    cout << "- Bloom filtering: " << (bloom_filter_is_enabled() ? "enabled" : "disabled") << "\n";
    cout << "- Current puzzle: " << g_search.p.getHeight() << "×" << g_search.k << "\n";
  }

  for (; g_search.s <= g_search.s_target; g_search.s++) {
    cout << "\n>>>>>>>> NEW STAGE (s = " << g_search.s << ") <<<<<<<<<\n" << '\n';

    Puz* result = nullptr;
    while (result == nullptr && g_search.stages[g_search.curr] != nullptr) {
      bool restart = false;
#pragma omp parallel
      while (g_search.stages[g_search.curr]->numResult() == 0 && !restart &&
             !should_checkpoint.load(std::memory_order_acquire)) {
        int thread_num = omp_get_thread_num();
        if (thread_num == 0) {
          maybe_request_auto_checkpoint();
#pragma omp critical(stats)
          {
            if (g_search.stages[g_search.curr]->stats.since_progress > RESTART_THRESH) {
              restart = true;
#pragma omp flush(restart)
            }
          }
          string counts_str = "";
          for (int i = g_search.curr - 1; i >= 0; i--) {
            if (g_search.stages[i] != nullptr)
              counts_str += to_string(g_search.stages[i]->numResult());
            else
              counts_str += "X";
            if (i != 0)
              counts_str += " <- ";
          }
          g_search.stages[g_search.curr]->display_stats(counts_str);
        } else {
          for (int i = g_search.curr; i >= 0; i--) {
            if ((i == 0 || g_search.stages[i - 1] == nullptr) ||
                (i < g_search.curr && safe_rand() >= PROB_PREV &&
                 g_search.stages[i]->numResult() < MAX_RESULTS) ||
                (i == g_search.curr && safe_rand() >= PROB_PREV)) {
              g_search.stages[i]->doWork(g_search.mode, g_search.strong);
              break;
            }
          }
        }
      }

      if (should_checkpoint.load(std::memory_order_acquire)) {
        bool exit_after_checkpoint = should_exit_after_checkpoint.load(std::memory_order_acquire);
        std::filesystem::path checkpoint_path = make_checkpoint_path();
        string checkpoint_filename = checkpoint_path.string();
        bool checkpoint_saved = save_checkpoint(checkpoint_filename);
        exit_after_checkpoint =
            should_exit_after_checkpoint.exchange(false, std::memory_order_acq_rel) ||
            exit_after_checkpoint;
        should_checkpoint.store(false, std::memory_order_release);

        if (exit_after_checkpoint) {
          cleanupGpuPipeline();
          return false;
        }

        schedule_next_auto_checkpoint(std::chrono::steady_clock::now());
        if (checkpoint_saved) {
          cout << "Auto checkpoint complete; continuing search.\n";
        } else {
          cerr << "Auto checkpoint failed; continuing search.\n";
        }
      }

      if (restart) {
        g_search.stages[g_search.curr]->try_restart();
      }

// Check for results. Atomically check if the current stage has results.
#pragma omp critical(take_result)
      {
        if (g_search.stages[g_search.curr]->numResult() > 0) {
          result = g_search.stages[g_search.curr]->takeResult(false);
          bool check = result->checkPuz(g_search.mode, g_search.strong);
          if (!check)
            result = nullptr;
        } else {
          result = nullptr;
        }
      }
    }

    if (result == nullptr)
      cerr << "ERROR: RESULT IS NULL" << '\n';
    ;
    assert(result != nullptr);
    g_search.p = *result;
    delete result;
    cout << "\n" << g_search.p << "\n" << '\n';

    ofstream puzzle_log(g_repo_root / "workspace" / "src" / "backend" / "puzzle_log.txt", ios::app);
    puzzle_log << g_search.p << '\n';
    puzzle_log.close();
    bool check = g_search.p.checkPuz(g_search.mode, g_search.strong);
    if (!check)
      cerr << "ERROR: Puzzle check failed!" << '\n';
    ;
    assert(check);

    // Safely remove the oldest stage.
    if (g_search.stages[0] != nullptr) {
      if (g_search.stages[1] == nullptr)
        cerr << "ERROR: stages[1] is NULL" << '\n';
      ;
      assert(g_search.stages[1] != nullptr);
      g_search.stages[1]->deactivatePrev();
      g_search.stages[0].reset();
    }

    for (int i = 0; i < g_search.num_stages - 1; i++)
      g_search.stages[i] = g_search.stages[i + 1];

    g_search.stages[g_search.curr] = make_shared<ILSStage>(g_search.s + 1, g_search.k, g_search.ft,
                                                           g_search.stages[g_search.curr - 1]);
  }

  if (!is_resume) {
    delete g_search.ft;
    g_search.ft = nullptr;
    for (int i = 0; i < g_search.num_stages; i++)
      if (g_search.stages[i] != nullptr)
        g_search.stages[i].reset();
  }

  return true;
}

void ctrlCHandler(int signum) {
  should_exit_after_checkpoint.store(true, std::memory_order_release);
  should_checkpoint.store(true, std::memory_order_release);
}

int main(int argc, char* argv[]) {
  signal(SIGINT, ctrlCHandler);
  g_repo_root = repo_root_from_executable(argv[0]);
  if (argc < 5 || argc > 7) {
    cerr << "usage: search_ils_staged_enhance <mode> <strong> <s_target> <k>"
         << " [<start_file>] [--bloom[=<expected>,<fpr>]]" << '\n';
    cerr << "\n- <mode> = 0 does check" << '\n';
    cerr << "         = 1 does simplifiable (obvious) check" << '\n';
    cerr << "         = 2 does local check" << '\n';
    cerr << "\n- <strong> = 0 does check" << '\n';
    cerr << "           = 1 does strong check" << '\n';
    cerr << "\n- <s_target> is number of rows to search up to." << '\n';
    cerr << "\n- <k> is puzzle width to search for." << '\n';
    cerr << "\n- [<start_file>] is an optional file:" << '\n';
    cerr << "    - .dat file to resume from checkpoint" << '\n';
    cerr << "    - .puz file to start searching from a specific puzzle" << '\n';
    cerr << "\n- [--bloom[=<expected>,<fpr>]] enables bloom filtering for fresh runs only." << '\n';
    cerr << "    - Example: --bloom=100000000,0.01" << '\n';
    cerr << "    - Checkpoint resume always follows the checkpoint bloom flag." << '\n';
    cerr << "\n- Automatic checkpoint backups are saved every 6 hours while searching." << '\n';
    cerr << "    - The search continues after each automatic checkpoint." << '\n';
    cerr << "    - Set ILS_AUTO_CHECKPOINT_SECONDS=0 to disable." << '\n';
    cerr << "\nCommon usage:" << '\n';
    cerr << "  search_ils_staged_enhance 1 1 14 6" << '\n';
    cerr << "  search_ils_staged_enhance 1 1 14 6 checkpoint_20251002_151226.dat" << '\n';
    cerr << "  search_ils_staged_enhance 1 1 14 6 start_puzzle.puz" << '\n';
    cerr << "  search_ils_staged_enhance 1 1 14 6 --bloom=100000000,0.01" << '\n';
    cerr << "  searches for a Simplifiable Strong USP with 6 columns and up to 14 rows." << '\n';
    return -1;
  }

  bool is_resume = false;

  // Parse command-line arguments
  int mode_i = atoi(argv[1]);
  int strong_arg = 2;
  int s_target_arg = 3;
  int k_arg = 4;
  int optional_arg_start = 5;
  g_search.mode = (mode_i == 0 ? CHECK_FULL : (mode_i == 1 ? CHECK_OBVIOUS : CHECK_LOCAL));
  g_search.strong = atoi(argv[strong_arg]) == 1;
  g_search.s_target = atoi(argv[s_target_arg]);
  g_search.k = atoi(argv[k_arg]);

  string start_file;
  bool bloom_requested = false;
  size_t bloom_expected = g_bloom_expected;
  double bloom_fpr = g_bloom_false_positive_rate;

  for (int i = optional_arg_start; i < argc; i++) {
    string arg = argv[i];
    if (arg.rfind("--bloom", 0) == 0) {
      size_t expected = 0;
      double fpr = 0.0;
      if (!parse_bloom_option(arg, expected, fpr)) {
        cerr << "ERROR: Invalid bloom option '" << arg << "'." << '\n';
        return -1;
      }
      bloom_requested = true;
      bloom_expected = expected;
      bloom_fpr = fpr;
    } else if (start_file.empty()) {
      start_file = arg;
    } else {
      cerr << "ERROR: Unexpected argument '" << arg << "'." << '\n';
      return -1;
    }
  }

  g_bloom_enable = bloom_requested;
  g_bloom_expected = bloom_expected;
  g_bloom_false_positive_rate = bloom_fpr;

  const check_mode_t requested_mode = g_search.mode;
  const bool requested_strong = g_search.strong;
  const int requested_s_target = g_search.s_target;
  const int requested_k = g_search.k;

  if (!start_file.empty()) {
    // Check if it's a checkpoint file or a starting puzzle file
    string input_file = start_file;
    size_t dot_pos = input_file.rfind('.');
    string extension = (dot_pos != string::npos) ? input_file.substr(dot_pos) : "";

    if (extension == ".dat") {
      // Resume from checkpoint file
      cout << "Resuming from checkpoint: " << input_file << '\n';
      restore_checkpoint(input_file);
      if (g_search.k != g_search.p.getWidth() || g_search.s_target < g_search.p.getHeight()) {
        cout << g_search.k << " " << g_search.p.getWidth() << " " << g_search.s_target << " "
             << g_search.p.getHeight() << '\n';
        cerr << "ERROR: Puzzle width or height does not match the given parameters." << '\n';
      }
      assert(g_search.k == g_search.p.getWidth());
      assert(g_search.s_target >= g_search.p.getHeight());
      is_resume = true;
      if (bloom_requested) {
        cerr << "Note: --bloom ignored when resuming from checkpoint." << '\n';
      }
    } else if (extension == ".puz") {
      // Start from a specific puzzle file
      cout << "Starting from puzzle file: " << input_file << '\n';
      g_search.p = Puz(input_file.c_str());
      g_search.s = g_search.p.getHeight();
      assert(g_search.k == g_search.p.getWidth());
      assert(g_search.s_target >= g_search.p.getHeight());
      is_resume = false;
    } else {
      cerr << "ERROR: Unknown file type '" << extension << "'. Expected .dat or .puz" << '\n';
      return -1;
    }
  } else {
    CheckpointProbe auto_start_checkpoint;
    if (find_auto_start_checkpoint(requested_mode, requested_strong, requested_s_target,
                                   requested_k, auto_start_checkpoint)) {
      cout << "Auto-starting from checkpoint: " << auto_start_checkpoint.path.string() << std::endl;
      cout << "Checkpoint current puzzle: (" << auto_start_checkpoint.puzzle_s << ","
           << auto_start_checkpoint.puzzle_k << ") SUSP; searching target: (" << requested_s_target
           << "," << requested_k << ")" << std::endl;
      restore_checkpoint(auto_start_checkpoint.path.string());
      g_search.mode = requested_mode;
      g_search.strong = requested_strong;
      g_search.s_target = requested_s_target;
      g_search.k = requested_k;
      is_resume = true;
      if (bloom_requested) {
        cerr << "Note: --bloom ignored when auto-starting from checkpoint." << std::endl;
      }
    } else {
      // Start from scratch with a minimal puzzle.
      g_search.s = 1;
      g_search.p = Puz(g_search.s, g_search.k);
      is_resume = false;
    }
  }

  cout << g_search.p << '\n';

  srandom(mix(clock(), time(NULL), getpid()));

  cudaGetDeviceCount(&deviceCount);
  enhanced_Hcheck_cuda_error("Finding number of CUDA devices");

  if (deviceCount <= 0) {
    cerr << "ERROR: No CUDA devices available for search_ils_staged_enhance." << '\n';
    return -1;
  }

  numGpuDevicesUsed = deviceCount;
  const char* gpu_devices_env = getenv("ILS_GPU_DEVICES");
  const char* legacy_gpu_workers_env = getenv("ILS_GPU_WORKERS");
  const char* gpu_device_count_env = (gpu_devices_env != nullptr && gpu_devices_env[0] != '\0')
                                         ? gpu_devices_env
                                         : legacy_gpu_workers_env;
  if (gpu_device_count_env != nullptr && gpu_device_count_env[0] != '\0') {
    int requested_gpu_devices = atoi(gpu_device_count_env);
    if (requested_gpu_devices > 0) {
      numGpuDevicesUsed = std::min(deviceCount, requested_gpu_devices);
    }
  }
  numGpuDevicesUsed = std::max(1, numGpuDevicesUsed);
  gpuLanesPerDevice = parse_positive_env("ILS_GPU_LANES_PER_DEVICE", 2);
  gpuLanesPerDevice = std::max(1, gpuLanesPerDevice);
  numGpuWorkers = numGpuDevicesUsed * gpuLanesPerDevice;
  cout << "Using " << numGpuDevicesUsed << " of " << deviceCount << " CUDA device"
       << (deviceCount == 1 ? "" : "s") << " for GPU simplification." << '\n';
  cout << "Selected CUDA devices:";
  for (int i = 0; i < numGpuDevicesUsed; i++) {
    cout << " " << i;
  }
  cout << '\n';
  cout << "GPU coordinator lanes: " << gpuLanesPerDevice << " per selected device." << '\n';
  gpuLaunchSlotBatch = parse_positive_env("ILS_GPU_BATCH_SLOTS", 32);
  gpuLaunchMaxTrials =
      choose_gpu_launch_max_trials(g_search.s_target, g_search.k, gpuLaunchSlotBatch);
  gpuLaunchSlotBatch = std::max(1, gpuLaunchMaxTrials / TRIALS);
  gpuSlotsPerWorker = parse_positive_env("ILS_GPU_SLOTS_PER_WORKER", 32);
  gpuSlotsPerWorker = std::max(2, gpuSlotsPerWorker);
  gpuCoalesceWaitUs = parse_positive_env("ILS_GPU_COALESCE_US", 5000);
  cout << "GPU launch coalescing: up to " << gpuLaunchSlotBatch << " task slots ("
       << gpuLaunchMaxTrials << " puzzles) per CUDA launch." << '\n';
  cout << "GPU task buffering: " << gpuSlotsPerWorker << " slots per CPU worker." << '\n';
  cout << "GPU coalescing wait: up to " << gpuCoalesceWaitUs << " us before each launch." << '\n';

  // Initialize GPU resources.
  gpu_resources.resize(numGpuWorkers);
  for (int i = 0; i < numGpuWorkers; i++) {
    int device_id = i % numGpuDevicesUsed;
    cudaSetDevice(device_id);
    enhanced_Hcheck_cuda_error("Setting CUDA device" + std::to_string(device_id));

    gpu_resources[i] = std::make_unique<GpuResources>(device_id, g_search.s_target, g_search.k,
                                                      gpuLaunchMaxTrials);
  }

  initializeGpuTaskSlots(omp_get_max_threads(), g_search.s_target, g_search.k, gpuSlotsPerWorker,
                         numGpuWorkers);
  stopGpuWorker.store(false, std::memory_order_release);
  gpuLaunchesCompletedByWorker.assign(numGpuWorkers, 0);
  gpuBatchesCompletedByWorker.assign(numGpuWorkers, 0);
  gpuPuzzlesCompletedByWorker.assign(numGpuWorkers, 0);
  gpuCoordinatorThreads.reserve(numGpuWorkers);
  for (int i = 0; i < numGpuWorkers; i++) {
    gpuCoordinatorThreads.emplace_back(gpuCoordinator, i, gpu_resources[i]->device_id,
                                       std::ref(*gpu_resources[i]));
  }

  g_auto_checkpoint_interval = std::chrono::seconds(
      parse_nonnegative_env("ILS_AUTO_CHECKPOINT_SECONDS", DEFAULT_AUTO_CHECKPOINT_SECONDS));
  schedule_next_auto_checkpoint(std::chrono::steady_clock::now());
  if (auto_checkpoint_enabled()) {
    cout << "Auto checkpoint backups: every " << g_auto_checkpoint_interval.count()
         << " seconds to " << (g_repo_root / "data").string() << '\n';
  } else {
    cout << "Auto checkpoint backups disabled." << '\n';
  }

  bool completed = search_ils(is_resume);
  if (!completed) {
    return 0;
  }
  bool check = g_search.p.checkPuz(g_search.mode, g_search.strong);
  if (!check)
    cerr << "ERROR: Final puzzle check failed!" << '\n';
  assert(check);
  g_search.p.sort();
  cout << "\n\n" << g_search.p << "\n" << '\n';
  cout << "is an "
       << (g_search.mode == CHECK_OBVIOUS ? "obvious "
                                          : (g_search.mode == CHECK_LOCAL ? "local " : ""))
       << (g_search.strong ? "strong " : "") << "(" << g_search.p.getHeight() << "," << g_search.k
       << ")-USP." << '\n';

  bool hier = hierarchically_special_unordered(g_search.p) == 0;
  if (hier) {
    cout << "is hierarchically special with reorder" << '\n';
  } else {
    cout << "is NOT hierarchically special with reorder" << '\n';
  }

  reset_isomorphs();
  cout << "Cleaning up GPU workers..." << '\n';
  cleanupGpuPipeline();

  cout << "GPU cleanup completed successfully." << '\n';

  return 0;
}
