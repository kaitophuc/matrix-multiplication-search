#include "GpuTask.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

const char* gpuTaskStateName(GpuTaskState state) {
  switch (state) {
    case GpuTaskState::Empty:
      return "Empty";
    case GpuTaskState::Filling:
      return "Filling";
    case GpuTaskState::Ready:
      return "Ready";
    case GpuTaskState::Processing:
      return "Processing";
    case GpuTaskState::Done:
      return "Done";
  }
  return "Unknown";
}

GpuTaskIdQueue::GpuTaskIdQueue(size_t capacity)
    : cap(capacity), cells(new Cell[capacity]), head(0), tail(0) {
  if (capacity == 0) {
    throw std::invalid_argument("GpuTaskIdQueue capacity must be greater than zero.");
  }
  for (size_t i = 0; i < cap; i++) {
    cells[i].sequence.store(i, std::memory_order_relaxed);
  }
}

bool GpuTaskIdQueue::try_push(int value) {
  size_t current_tail = tail.load(std::memory_order_relaxed);
  while (true) {
    Cell& cell = cells[current_tail % cap];
    size_t sequence = cell.sequence.load(std::memory_order_acquire);
    intptr_t diff = (intptr_t) sequence - (intptr_t) current_tail;

    if (diff == 0) {
      if (tail.compare_exchange_weak(current_tail, current_tail + 1,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        cell.value = value;
        cell.sequence.store(current_tail + 1, std::memory_order_release);
        return true;
      }
    } else if (diff < 0) {
      return false;
    } else {
      current_tail = tail.load(std::memory_order_relaxed);
    }
  }
}

bool GpuTaskIdQueue::try_pop(int& value) {
  size_t current_head = head.load(std::memory_order_relaxed);
  while (true) {
    Cell& cell = cells[current_head % cap];
    size_t sequence = cell.sequence.load(std::memory_order_acquire);
    intptr_t diff = (intptr_t) sequence - (intptr_t) (current_head + 1);

    if (diff == 0) {
      if (head.compare_exchange_weak(current_head, current_head + 1,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        value = cell.value;
        cell.sequence.store(current_head + cap, std::memory_order_release);
        return true;
      }
    } else if (diff < 0) {
      return false;
    } else {
      current_head = head.load(std::memory_order_relaxed);
    }
  }
}

bool GpuTaskIdQueue::empty() const {
  return size_approx() == 0;
}

size_t GpuTaskIdQueue::size_approx() const {
  size_t current_tail = tail.load(std::memory_order_acquire);
  size_t current_head = head.load(std::memory_order_acquire);
  return current_tail - current_head;
}

size_t GpuTaskIdQueue::capacity() const {
  return cap;
}

GpuTask::~GpuTask() {
  releasePinned();
}

void GpuTask::allocatePinned(int max_s, int max_k, int max_batch, int slot_id,
                             int owner_worker_id) {
  releasePinned();

  this->max_s = max_s;
  this->max_k = max_k;
  this->max_batch = max_batch;
  this->slot_id = slot_id;
  this->origin_worker_id = owner_worker_id;
  this->max_entries_per_row =
      (unsigned int) ((max_k + ELTS_PER_ENTRY - 1) / ELTS_PER_ENTRY);
  this->pinned_puz_entries =
      (size_t) max_s * (size_t) max_entries_per_row * (size_t) max_batch;

  cudaMallocHost(&puz_pinned_buffer, pinned_puz_entries * sizeof(e_type));
  cudaMallocHost(&results_pinned_buffer, (size_t) max_batch * sizeof(int));
  enhanced_Hcheck_cuda_error("Allocating pinned GpuTask slot memory");
  owns_pinned_buffers = true;

  puzzles.reserve(max_batch);
  result.results.reserve(max_batch);
}

void GpuTask::releasePinned() {
  if (owns_pinned_buffers && puz_pinned_buffer != nullptr) {
    cudaFreeHost(puz_pinned_buffer);
  }
  if (owns_pinned_buffers && results_pinned_buffer != nullptr) {
    cudaFreeHost(results_pinned_buffer);
  }
  puz_pinned_buffer = nullptr;
  results_pinned_buffer = nullptr;
  owns_pinned_buffers = false;
  pinned_puz_entries = 0;
}

bool GpuTask::tryBeginFilling(int worker_id, int new_s, int new_k, int new_q_prior,
                              double new_gap, int new_frontier_type) {
  GpuTaskState expected = GpuTaskState::Empty;
  if (!state.compare_exchange_strong(expected, GpuTaskState::Filling,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
    return false;
  }

  s = new_s;
  k = new_k;
  batch_size = 0;
  origin_worker_id = worker_id;
  q_prior = new_q_prior;
  gap = new_gap;
  frontier_type = new_frontier_type;
  done = false;
  generation++;
  puzzles.clear();
  result.results.clear();

  if (results_pinned_buffer != nullptr) {
    memset(results_pinned_buffer, 0, (size_t) max_batch * sizeof(int));
  }
  return true;
}

namespace {
void appendPuzzleToPinnedBuffer(GpuTask& task, const Puz& p) {
  if (task.state.load(std::memory_order_acquire) != GpuTaskState::Filling) {
    throw std::logic_error("GpuTask slot must be Filling before appending puzzles.");
  }
  if (task.puz_pinned_buffer == nullptr) {
    throw std::logic_error("GpuTask slot does not have pinned puzzle memory.");
  }
  if (p.getHeight() != (unsigned int) task.s || p.getWidth() != (unsigned int) task.k) {
    throw std::invalid_argument("Puzzle dimensions do not match GpuTask slot dimensions.");
  }
  if (task.batch_size >= task.max_batch) {
    throw std::out_of_range("GpuTask slot batch is full.");
  }

  const unsigned int entries_per_row =
      (unsigned int) ((task.k + ELTS_PER_ENTRY - 1) / ELTS_PER_ENTRY);
  e_type* dst = task.puz_pinned_buffer +
                (size_t) task.batch_size * (size_t) task.s * entries_per_row;
  for (int i = 0; i < task.s * (int) entries_per_row; i++) {
    dst[i] = p.getData(i);
  }
}
} // namespace

void GpuTask::appendPuzzle(const Puz& p) {
  appendPuzzleToPinnedBuffer(*this, p);
  puzzles.push_back(p);
  batch_size++;
}

void GpuTask::appendPuzzle(Puz&& p) {
  appendPuzzleToPinnedBuffer(*this, p);
  puzzles.push_back(std::move(p));
  batch_size++;
}

void GpuTask::markReady() {
  {
    std::lock_guard<std::mutex> lock(m);
    if (state.load(std::memory_order_acquire) != GpuTaskState::Filling) {
      throw std::logic_error("GpuTask::markReady requires state Filling.");
    }
    state.store(GpuTaskState::Ready, std::memory_order_release);
  }
  cv.notify_all();
}

bool GpuTask::markProcessing() {
  GpuTaskState expected = GpuTaskState::Ready;
  bool changed = state.compare_exchange_strong(expected, GpuTaskState::Processing,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire);
  if (changed) {
    cv.notify_all();
  }
  return changed;
}

void GpuTask::markDone() {
  {
    std::lock_guard<std::mutex> lock(m);
    if (state.load(std::memory_order_acquire) != GpuTaskState::Processing) {
      throw std::logic_error("GpuTask::markDone requires state Processing.");
    }
    done = true;
    state.store(GpuTaskState::Done, std::memory_order_release);
  }
  cv.notify_all();
}

void GpuTask::markEmpty() {
  {
    std::lock_guard<std::mutex> lock(m);
    if (state.load(std::memory_order_acquire) != GpuTaskState::Done) {
      throw std::logic_error("GpuTask::markEmpty requires state Done.");
    }
    done = false;
    batch_size = 0;
    puzzles.clear();
    result.results.clear();
    state.store(GpuTaskState::Empty, std::memory_order_release);
  }
  cv.notify_all();
}

GpuTaskState GpuTask::getState() const {
  return state.load(std::memory_order_acquire);
}

void GpuTask::waitForState(GpuTaskState desired) {
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&] { return state.load(std::memory_order_acquire) == desired; });
}

void GpuTask::copyResultsToPuzzles() {
  if (state.load(std::memory_order_acquire) != GpuTaskState::Done) {
    throw std::logic_error("GpuTask::copyResultsToPuzzles requires state Done.");
  }
  if (results_pinned_buffer == nullptr) {
    throw std::logic_error("GpuTask slot does not have a result buffer.");
  }
  if (batch_size < 0 || batch_size > (int) puzzles.size()) {
    throw std::logic_error("GpuTask batch size does not match stored puzzles.");
  }

  for (int i = 0; i < batch_size; i++) {
    puzzles[i].setResults(results_pinned_buffer[i]);
    puzzles[i].setSimplified(true);
  }
}

void GpuTask::solve(GpuResources& gpu_resources) {
  int currentDevice;
  cudaGetDevice(&currentDevice);

  int num = puzzles.size();
  gpu_resources.make_tests_enhance(puzzles, num);
  outer_Kernel(gpu_resources, puzzles[0].getHeight(), puzzles[0].getWidth(), num);

  int* results = gpu_resources.results_pinned_buffer;

  GPUResult curResult;
  curResult.results = std::vector<int>(results, results + num);

  {
    std::lock_guard<std::mutex> lk(m);
    result = std::move(curResult);
    done = true;
  }
  cv.notify_all();
}

void GpuTask::solveFromPinnedBuffers(GpuResources& gpu_resources) {
  if (puz_pinned_buffer == nullptr || results_pinned_buffer == nullptr) {
    throw std::logic_error("GpuTask slot must have pinned buffers before GPU solve.");
  }
  if (batch_size <= 0) {
    return;
  }

  outer_Kernel(gpu_resources, puz_pinned_buffer, results_pinned_buffer, s, k, batch_size);
}

void GpuTask::generateRandom(int s, int k, int trials) {
  puzzles.clear();
  for (int i = 0; i < trials; i++) {
    Puz p(s, k);
    p.setRandom();
    puzzles.push_back(p);
  }
  this->s = s;
  this->k = k;
  if (!check_size(puzzles)) {
    throw std::runtime_error("Puzzle size does not match the expected dimensions.");
  }
  result.results.resize(puzzles.size(), 0);

  for (size_t i = 0; i < puzzles.size(); ++i) {
    result.results[i] = rand() % 100;  // Random results for testing
  }
}

void GpuTask::encode(const std::string& filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }

  // Write metadata first
  size_t numPuzzles = puzzles.size();
  ofs.write(reinterpret_cast<const char*>(&numPuzzles), sizeof(numPuzzles));
  ofs.write(reinterpret_cast<const char*>(&s), sizeof(s));
  ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));

  // Write each puzzle's data
  for (const auto& p : puzzles) {
    size_t bytes;
    // Assumes p.serialize() allocates with new unsigned char[]
    void* buf = p.serialize(bytes);

    // Write the size of the data block, then the data itself
    ofs.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    ofs.write(reinterpret_cast<const char*>(buf), bytes);

    // Free the buffer allocated by serialize()
    delete[] reinterpret_cast<unsigned char*>(buf);
  }

  // Encode GPUResult
  ofs.write(reinterpret_cast<const char*>(&result.results[0]), result.results.size() * sizeof(int));

  // Encode done flag
  ofs.write(reinterpret_cast<const char*>(&done), sizeof(done));

  // Encode mutex state (not typically necessary, but included for completeness)
  ofs.write(reinterpret_cast<const char*>(&m), sizeof(m));
  ofs.write(reinterpret_cast<const char*>(&cv), sizeof(cv));

  // Close the file
  if (!ofs) {
    throw std::runtime_error("Failed to write all data to file: " + filename);
  }
  ofs.close();
}

void GpuTask::decode(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Could not open file for reading: " + filename);
  }

  // Read metadata first
  size_t numPuzzles;
  ifs.read(reinterpret_cast<char*>(&numPuzzles), sizeof(numPuzzles));
  if (!ifs)
    throw std::runtime_error("Failed to read puzzle count from file.");

  ifs.read(reinterpret_cast<char*>(&s), sizeof(s));
  if (!ifs)
    throw std::runtime_error("Failed to read puzzle height 's' from file.");

  ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
  if (!ifs)
    throw std::runtime_error("Failed to read puzzle width 'k' from file.");

  puzzles.clear();
  puzzles.reserve(numPuzzles);

  for (size_t i = 0; i < numPuzzles; ++i) {
    // Read the size of the upcoming data block
    size_t bytes;
    ifs.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    if (!ifs)
      throw std::runtime_error("Failed to read size of puzzle data for puzzle #" +
                               std::to_string(i));

    // Use a smart pointer for safe memory management
    std::unique_ptr<unsigned char[]> buf(new unsigned char[bytes]);
    ifs.read(reinterpret_cast<char*>(buf.get()), bytes);
    if (!ifs)
      throw std::runtime_error("Failed to read data for puzzle #" + std::to_string(i));

    // Reconstruct the puzzle from the buffer
    // Assumes a Puz constructor that takes a buffer pointer
    puzzles.emplace_back(buf.get());
  }

  s = puzzles.empty() ? 0 : puzzles[0].getHeight();
  k = puzzles.empty() ? 0 : puzzles[0].getWidth();

  // Decode GPUResult
  size_t resultSize = puzzles.size();
  result.results.resize(resultSize);
  ifs.read(reinterpret_cast<char*>(&result.results[0]), resultSize * sizeof(int));
  if (!ifs)
    throw std::runtime_error("Failed to read results from file.");

  // Decode done flag
  ifs.read(reinterpret_cast<char*>(&done), sizeof(done));
  if (!ifs)
    throw std::runtime_error("Failed to read done flag from file.");

  // Decode mutex state (not typically necessary, but included for completeness)
  ifs.read(reinterpret_cast<char*>(&m), sizeof(m));
  if (!ifs)
    throw std::runtime_error("Failed to read mutex state from file.");
  ifs.read(reinterpret_cast<char*>(&cv), sizeof(cv));
  if (!ifs)
    throw std::runtime_error("Failed to read condition variable state from file.");

  // Close the file
  ifs.close();
  if (!ifs) {
    throw std::runtime_error("Failed to read all data from file: " + filename);
  }
}

void solveGpuTask(GpuResources& gpu_resources, const std::string& input_filename,
                  const std::string& output_filename) {
  std::shared_ptr<GpuTask> task = std::make_shared<GpuTask>();
  try {
    task->decode(input_filename);
    printf("Data read from %s successfully.\n", input_filename.c_str());
  } catch (const std::runtime_error& e) {
    fprintf(stderr, "Error reading from file: %s\n", e.what());
    return;
  }

  task->solve(gpu_resources);

  try {
    task->encode(output_filename);
    printf("Data written to %s successfully.\n", output_filename.c_str());
  } catch (const std::runtime_error& e) {
    fprintf(stderr, "Error writing to file: %s\n", e.what());
    return;
  }
}
