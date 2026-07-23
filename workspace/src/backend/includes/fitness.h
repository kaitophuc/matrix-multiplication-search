/**
 * Classes for fitness testing of puzzles and puzzle-like objects.
 * Contains abstract class FitTester which specifies interface.
 *
 * All fitness testers subclass it and so have a getFit
 * function. getFit returns the fitness of the puzzle and the maximum
 * possible fitness. The fitness of the puzzle equals the maximum
 * fitness iff the puzzle is a SUSP.
 *
 * Provides fitness testers:
 * - SubsetFitTester - Good approximation, fast.
 * - TopDownFitTester - Good approximation, slower.
 * - DefinitionFitTester - Perfect approximation, infeasible for large s.
 * - TimingTester - Terrible approximation, very fast.
 *
 * Author: Matt Anderson
 */

#pragma once

/* Example usage:

#include "fitness.h"

SubsetFitTester sft(time(NULL));

Puzzle p;
// ...construct puzzle...

double fit, fit_max;
tie(fit, fit_max) = sft.getFit(p);

*/

#include <tuple>
#include <chrono>
#include "Puz.h"

using namespace std;

/**
 * Abstract class for fitness testing.
 */
class FitTester {
 public:
  /**
   * Takes a Puzzle and returns the fitness of that Puzzle and
   * the maximum possible fitness.
   */
  // Warning: Naive implementation that doesn't account for internals
  // of DistPuzzle.
  virtual tuple<double, double> getFit(const Puz& p) = 0;
  virtual string getName() {
    return "yo im wrong";
  };

  // Serialization support (simplified for now)
  virtual string getTypeName() const {
    return "FitTester";
  }

  double getFitDerv(const Puz& p, unsigned int r, unsigned int c, unsigned int val) {
    Puz p2(p.getHeight(), p.getWidth());
    for (int a = 0; a < p.getHeight(); a++) {
      for (int b = 0; b < p.getWidth(); b++) {
        p2.set(a, b, p.get(a, b));
      }
    }
    p2.set(r, c, val);
    return get<0>(getFit(p2)) - get<0>(getFit(p));
  };
};

#define DEFAULT_SUBSETS (s * s)  //(pow(2,0.85*s)) // (s*s)

/**
 * A fitness test that randomly sample subpuzzles of a given puzzle
 * and returns number of subpuzzles that satisfy check().  All tests
 * made by the same instance of this class use the same seed for
 * consistency and reproducibility.
 */
class SubsetFitTester : public FitTester {
 private:
  int seed;
  unsigned int force_size;
  bool use_exp_weights;
  bool use_likelihood;
  bool strong;

 public:
  /**
   * Takes a optional arguments seed and rand_s.  Uses the same seed
   * for every fitness test in this instance.  If force_size is set to
   * anything but 0 it only random subpuzzles of size force_size.
   */
  SubsetFitTester(bool strong, int seed = 0, unsigned int force_size = 0,
                  bool use_exp_weights = false, bool use_likelihood = false);

  tuple<double, double> getFit(const Puz& p);

  virtual string getName() override {
    return "Subset Fit Tester";
  }
};

template <class T>
class AdaptiveFitTester : public FitTester {
 protected:
  vector<T> subsets;

 public:
  virtual void evolveFitness() = 0;
  string getName() {
    return "Adaptive Fit Tester";
  }

 protected:
  virtual void updateSubsets() = 0;
  virtual bool lower(unsigned int i) = 0;
  virtual bool raise(unsigned int i) = 0;
};

typedef struct _subset_t {
  vector<bool> subset;
  vector<int> neighbors;
  double ave_sat;
  unsigned int lifetime;
  unsigned int size;

} subset_t;

#define RAISE_THRESH 0.8
#define LOWER_THRESH 0.2
#define UPDATE_RATE 20  // Updates after this many calls to evolveFitness().

/**
 * A fitness test that adaptively samples subpuzzles of a given puzzle
 * and returns number of subpuzzles that satisfy check().  All tests
 * made by the same instance of this class use the same seed for
 * consistency and reproducibility.  fit_max returned by getFit may
 * change over time.
 */
class AdaptiveSubsetFitTester : public AdaptiveFitTester<subset_t> {
 private:
  int seed;
  unsigned int last_update = 0;
  unsigned int s;
  bool use_exp_weights;
  bool use_likelihood;

 public:
  /**
   * Takes the size of the puzzles being tested. Takes a optional
   * arguments seed.  Uses the same seed for every fitness test in
   * this instance.
   */
  AdaptiveSubsetFitTester(unsigned int s = 0, int seed = 0, bool use_exp_weights = false,
                          bool use_likelihood = false);

  tuple<double, double> getFit(const Puz& p);
  void evolveFitness();

  string getName() {
    return "Adaptive Subset Fit Tester";
  }

 private:
  void updateSubsets();
  bool lower(unsigned int i);
  bool raise(unsigned int i);
  int countOccurs(const vector<bool>& subset) const;
};

//==========================================================================

typedef struct _subset_cache_t {
  vector<bool> subset;
  vector<int> neighbors;
  double ave_sat;
  unsigned int lifetime;
  unsigned int size;

  short fitness_cache;
  bool dervs_valid;
  vector<short> dervs_cache;

} subset_cache_t;

/**
 * A fitness test that adaptively samples subpuzzles of a given puzzle
 * and returns number of subpuzzles that satisfy check().  All tests
 * made by the same instance of this class use the same seed for
 * consistency and reproducibility.  fit_max returned by getFit may
 * change over time.
 */
class CachedAdaptiveSubsetFitTester : public AdaptiveFitTester<subset_cache_t> {
 private:
  int seed;
  unsigned int last_update = 0;
  unsigned int s;
  unsigned int k;

  vector<vector<bool> >* skip;
  //  vector<subset_cache_t> subsets;
  Puz cached_p;
  tuple<double, double> fitness_cache;
  vector<double> dervs;

 public:
  /**
   * Takes the size of the puzzles being tested. Takes a optional
   * arguments seed.  Uses the same seed for every fitness test in
   * this instance.
   */
  CachedAdaptiveSubsetFitTester(unsigned int s = 1, unsigned int k = 1, int seed = 0,
                                unsigned int default_subsets = 0,
                                vector<vector<bool> >* skip = NULL);

  tuple<double, double> getFit(const Puz& p);
  double getFitDerv(const Puz& p, unsigned int r, unsigned int c, unsigned int val);

  void displayCache();
  void displayCacheDifference(const CachedAdaptiveSubsetFitTester& prev);
  Puz& getCachedPuzzle() {
    return cached_p;
  };
  void evolveFitness();
  string getName() {
    return "Cached Adaptive Subset Fit Tester";
  }

 private:
  void updateSubsets();
  bool lower(unsigned int i);
  bool raise(unsigned int i);
  int countOccurs(const vector<bool>& subset) const;
  void updateFit(const Puz& p, bool do_derv);
  bool invalidateCache(const Puz& p, bool do_derv);
  void verifyCache(bool do_derv);
};

//==========================================================================

class TopDownFitTester : public FitTester {
 public:
  /**
   * Takes a optional argument for depth to search.
   */
  TopDownFitTester(int max_depth = 3);

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "Top Down Fit Tester";
  }

 private:
  int max_depth;

  double getFitHelper(const Puz& p, int depth);
};

/**
 * An "exact" fitness function.  Counts the number of pairs (pi_2,
 * pi_3) that the puzzle fails the SUSP condition for.  Relaxed to
 * work with real-valued distributional puzzles defined by this
 * class.  CMA-ES struggles to optimize this function, even for k =
 * 4. Run time infeasible for large s.
 */
class DefinitionFitTester : public FitTester {
 public:
  DefinitionFitTester();

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "Definition Fit Tester";
  }
};

/**
 * Tries to estimate fitness using the time it takes to check the
 * puzzle.  CMA-ES seems to be able to be able to optimize this, but
 * it doesn't appear to lead to SUSP.
 */
class TimingFitTester : public FitTester {
 private:
  unsigned int sample_size;

 public:
  TimingFitTester(unsigned int sample_size = 10);

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "Timing Fit Tester";
  }
};

/**
 * Estimates how far the puzzle is from being a USP by testing the 1s,
 * 2s, 3s pieces for duplication.
 */
class PiecesFitTester : public FitTester {
 private:
 public:
  PiecesFitTester(){};

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "Pieces Fit Tester";
  }
};

/**
 * Estimates how far the puzzle is from being a SUSP by counting the
 * number of non-zero entries in the corresponding 3DM instance.
 */
class TDMSizeFitTester : public FitTester {
 private:
  bool strong;
  check_mode_t mode;
  unsigned int require_special;

 public:
  TDMSizeFitTester(check_mode_t mode, bool strong, unsigned int require_special = 0)
      : mode(mode), strong(strong), require_special(require_special){};

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "TDM Size Fit Tester";
  }
};

/**
 * Estimates how far the puzzle is from being hierarchically special.
 */
class HSFitTester : public FitTester {
 private:
  bool strong;
  check_mode_t mode;
  unsigned int require_special;

 public:
  HSFitTester(check_mode_t mode, bool strong, unsigned int require_special = 0)
      : mode(mode), strong(strong), require_special(require_special){};

  tuple<double, double> getFit(const Puz& p);

  string getName() {
    return "HS Fit Tester";
  }

  virtual string getTypeName() const override {
    return "HSFitTester";
  }
};

// Factory function to create FitTester from type name
FitTester* createFitTesterFromType(const string& typeName);
