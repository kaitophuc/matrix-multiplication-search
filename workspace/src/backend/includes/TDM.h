/**
 * @file TDM.h
 * @brief 3D Matching verification for Strong Uniquely Solvable Puzzles
 *
 * This file contains the TDM class which constructs and analyzes 3D matching
 * instances from puzzles to verify the (Strong) USP property. The class supports
 * both CPU and GPU-accelerated verification.
 *
 * @note TDM instances cannot be copied (copy constructor/assignment deleted)
 * @author Kaito, Matt
 */

#pragma once

#include <atomic>
#include <vector>
#include <stdio.h>

#define GPU_SIZE_THRESH 20  ///< Minimum puzzle size to consider GPU acceleration
#define MAX_ON_GPU 8        ///< Maximum concurrent GPU simplifications

class Puz;  // Early declaration to avoid header loops.

using namespace std;

/**
 * @class TDM
 * @brief 3D Matching instance for puzzle verification
 *
 * Constructs and analyzes a 3D matching instance from a puzzle to verify
 * whether it satisfies the (Strong) Uniquely Solvable Puzzle property.
 *
 * **How it works**:
 * 1. Constructs a 3D matching instance from puzzle rows
 * 2. Simplifies the instance by removing impossible edges
 * 3. Checks if only the trivial perfect 3D matching exists
 *
 * **GPU Acceleration**:
 * - Automatically uses GPU for puzzles with s ≥ 20 (if ALLOW_GPU)
 * - Can force GPU usage with FORCE_GPU
 * - Tracks concurrent GPU operations (max 8)
 *
 * **Thread Safety**: Not thread-safe. Use separate instances per thread.
 *
 * **Copying**: Explicitly disabled (deleted copy constructor/assignment)
 *
 * Example usage:
 * @code
 * Puz p(14, 6);
 * // ... set puzzle values ...
 *
 * // Create TDM instance (will use GPU if s >= 20)
 * TDM tdm(p, true, TDM::ALLOW_GPU);
 *
 * // Simplify to remove impossible edges
 * tdm.simplify(TDM::EAGER);
 *
 * // Check edge count
 * unsigned int edges = tdm.count();
 * cout << "Edges in 3DM: " << edges << endl;
 * @endcode
 */
class TDM {
 public:
  static unsigned long long concurrent_on_gpu;  ///< Number of concurrent GPU simplifications
  static std::atomic<unsigned long long> num_gpu_simplify;   ///< Total GPU simplifications performed
  static std::atomic<unsigned long long> num_simplify;       ///< Total simplifications performed

  /**
   * @enum gpuUsage
   * @brief GPU usage policy for TDM operations
   */
  enum gpuUsage {
    NO_GPU,     ///< Never use GPU
    ALLOW_GPU,  ///< Use GPU if puzzle size >= GPU_SIZE_THRESH (20)
    FORCE_GPU   ///< Always use GPU regardless of size
  };

  /**
   * @enum simplifyMode
   * @brief Simplification strategy for TDM instance
   */
  enum simplifyMode {
    EAGER,  ///< Fully simplify and update TDM (CPU or GPU)
    LAZY,   ///< Fully simplify but may not update TDM structure on the CPU
    SLOW    ///< Use CPU-only implementation (for debugging)
  };

 private:
  unsigned int s;
  bool* tdm;
  bool strong;
  bool computed;
  bool simplified;
  gpuUsage use_gpu;
  unsigned int count_result = 0;

  Puz* p;

  /*
   * Sets the edge (r1, r2, r3) in the 3DM instance accorinding to val.
   */
  inline bool set(unsigned int r1, unsigned int r2, unsigned int r3, bool val) {
    return tdm[r1 * s * s + r2 * s + r3] = val;
    // return the value set
  }

  /*
   * Calculates the 3D matching instance from the given puzzle at construction.
   */
  void compute();

 public:
  /**
   * @brief Construct TDM instance from puzzle
   *
   * Creates a 3D matching instance by analyzing all triples of puzzle rows.
   * An edge (r1, r2, r3) exists if rows r1, r2, r3 witness the (Strong) USP property.
   *
   * @param p Puzzle to verify
   * @param strong If true, checks Strong USP; if false, checks USP
   * @param use_gpu GPU usage policy (NO_GPU, ALLOW_GPU, or FORCE_GPU)
   *
   * @note Construction time: O(s³ × k) where s = rows, k = columns
   * @note For s ≥ 20, ALLOW_GPU will use GPU acceleration
   *
   * Example:
   * @code
   * Puz p(14, 6);
   * TDM tdm(p, true, TDM::ALLOW_GPU);  // Strong USP, GPU if beneficial
   * @endcode
   */
  TDM(const Puz& p, bool strong, gpuUsage use_gpu = NO_GPU);

  // Prevent implicit / explicit copying of TDM.
  TDM(const TDM& t) = delete;
  TDM(TDM& t) = delete;
  TDM& operator=(const TDM& tr) = delete;

  /**
   * @brief Destructor - frees TDM storage
   */
  ~TDM();

  bool operator==(const TDM& tr) const;

  static void setNumSimplify(unsigned long long num) {
    TDM::num_simplify.store(num, std::memory_order_relaxed);
  }

  static void setNumGPUSimplify(unsigned long long num) {
    TDM::num_gpu_simplify.store(num, std::memory_order_relaxed);
  }

  static unsigned long long getNumSimplify() {
    return TDM::num_simplify.load(std::memory_order_relaxed);
  }

  static unsigned long long getNumGPUSimplify() {
    return TDM::num_gpu_simplify.load(std::memory_order_relaxed);
  }

  /**
   * @brief Get edge existence in 3DM instance
   *
   * @param r1 First row index
   * @param r2 Second row index
   * @param r3 Third row index
   * @return true if edge (r1, r2, r3) exists in the 3DM instance
   *
   * @pre 0 <= r1, r2, r3 < s
   */
  inline bool get(unsigned int r1, unsigned int r2, unsigned int r3) const {
    return tdm[r1 * s * s + r2 * s + r3];
  }

  /**
   * @brief Get puzzle size (number of rows)
   * @return Number of rows in the puzzle (s)
   */
  inline unsigned int size() const {
    return s;
  }

  /**
   * @brief Check if this is a Strong USP verification
   * @return true if checking Strong USP, false for USP
   */
  inline bool isStrong() const {
    return strong;
  }

  /**
   * @brief Simplify the 3DM instance
   *
   * Removes edges that cannot be part of a perfect 3D matching by projecting
   * to 2D faces and iteratively removing impossible edges.
   *
   * **Algorithm**:
   * 1. Project 3DM to three 2D faces of the cube
   * 2. Remove edges not present in any 2D matching
   * 3. Repeat until no changes
   *
   * **Modes**:
   * - **EAGER**: Fully simplify (CPU or GPU) and update TDM instance
   * - **LAZY**: Fully simplify but may only update edge count, not the instance itself
   * - **SLOW**: Use CPU-only implementation (for debugging/comparison)
   *
   * @param mode Simplification strategy
   *
   * @note LAZY cannot be used for cases where you need the TDM instance to be updated
   * @note GPU is used automatically for large puzzles (s ≥ 20) if ALLOW_GPU was set
   *
   * Example:
   * @code
   * TDM tdm(p, true, TDM::ALLOW_GPU);
   * tdm.simplify(TDM::EAGER);  // Simplify and update
   * unsigned int edges = tdm.count();
   * @endcode
   */
  void simplify(simplifyMode mode);

  /**
   * @brief Count edges in the 3DM instance
   *
   * Counts the number of edges, optionally with weighted counting.
   *
   * @param active_idx Optional vector of active indices
   * @param diag_weight Weight for diagonal edges (default: 1)
   * @param near_diag_weight Weight for near-diagonal edges (default: 1)
   * @param off_diag_weight Weight for off-diagonal edges (default: 1)
   * @return Number of edges (or weighted count)
   *
   * Example:
   * @code
   * TDM tdm(p, true);
   * tdm.simplify(TDM::EAGER);
   * unsigned int edge_count = tdm.count();
   * cout << "Simplified to " << edge_count << " edges" << endl;
   * @endcode
   */
  unsigned int count(vector<int>* active_idx = NULL, unsigned int diag_weight = 1,
                     unsigned int near_diag_weight = 1, unsigned int off_diag_weight = 1);

  /**
   * @brief Print TDM instance to file
   * @param f File pointer
   * @param verbose If true, print detailed information
   */
  void fprint(FILE* f, bool verbose = true);

  /**
   * @brief Print TDM instance to console
   * @param verbose If true, print detailed information
   */
  void print(bool verbose = true);

  void setResult(int res) {
    count_result = res;
  }

  void setSimplified(bool simplified) {
    this->simplified = simplified;
    this->computed = simplified;
  }

 private:
  /* Private simplification functions. Used to support different modes
   * of simplify().
   */
  bool simplify_gpu(simplifyMode mode);
  void simplify_cpu();
  void simplify_cpu_slow();
};
