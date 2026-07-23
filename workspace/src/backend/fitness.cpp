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
#include <iostream>
#include <map>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>

#include "set.h"
#include "Puz.h"
#include "TDM.h"
#include "fitness.h"
#include "checker.h"
#include "special.h"

using namespace std;
using boost::numeric::ublas::matrix;

// ======================================================================
//
//  Classless Helper Functions
//
// ======================================================================

double weight_function(int s, bool use_exp) {
  assert(s >= 2);
  if (use_exp)
    return pow(2, (s - 1)) + 1;  // Exponential weighting on size.
  else
    return 1;  // Constant weighting.
}

template <class T>
int samplecmp(const vector<T> subset1, const vector<T> subset2, unsigned int s) {
  bool smaller = true;
  bool larger = true;

  for (int i = 0; i < s; i++) {
    if (subset1[i] > subset2[i])
      smaller = false;
    if (subset1[i] < subset2[i])
      larger = false;
  }

  bool same = smaller && larger;

  if (same)
    return 0;
  if (smaller)
    return -1;
  if (larger)
    return 1;
  return 2;  // Incomparable.
}

// XXX - Deprecated by removing DistPuzzle.
// double getLikelihood(const Puz& p){

//   double res = 0.0;
//   unsigned int s = p.getHeight();
//   unsigned int k = p.getWidth();
//   double x,y,z;
//   FORALL_RC(tie(x,y,z) = p.getDistEntry(r,c);
// 	    res += (x > y ? x : (y > z ? y : z));)
//     res /= (s * k);
//   assert(res >= 1/2);
//   return res;
// }

// Penalty function for distributions a, b, c where all the weight
// is not on a single element.
double imbalanceCost(double a, double b, double c) {
  double res = 0;
  if (a >= b && a >= c) {
    res = 3 - (fabs(a - 1) + fabs(b) + fabs(c));
  } else if (b >= c) {
    res = 3 - (fabs(a) + fabs(b - 1) + fabs(c));
  } else {
    res = 3 - (fabs(a) + fabs(b) + fabs(c - 1));
  }
  res /= 3.0;
  assert(res >= 0);

  return res;
}

// ======================================================================
//
//  Implementation of SubsetFitTester
//
// ======================================================================

/**
 * Takes a optional arguments seed and rand_s.  Uses the same seed
 * for every fitness test in this instance.  If force_size is set to
 * anything but 0 it only random subpuzzles of size force_size.
 */
SubsetFitTester::SubsetFitTester(bool strong, int seed, unsigned int force_size,
                                 bool use_exp_weights, bool use_likelihood)
    : strong(strong),
      seed(seed),
      force_size(force_size),
      use_exp_weights(use_exp_weights),
      use_likelihood(use_likelihood) {}

tuple<double, double> SubsetFitTester::getFit(const Puz& p) {
  mt19937 mt(seed);

  double fit = 0;
  unsigned int s = p.getHeight();
  unsigned int k = p.getWidth();
  int local_samples = DEFAULT_SUBSETS;
  int fit_max = local_samples * weight_function(s, use_exp_weights);

  fit += (p.checkPuz(CHECK_FULL, strong) ? 1 : 0);
  if (fit == 1) {
    return tuple<double, double>{fit_max, fit_max};
  }

  if (s <= 2) {
    fit *= fit_max;
    return tuple<double, double>{fit, fit_max};
  }

  for (int i = 0; i < local_samples - 1; i++) {
    int sp = force_size;
    if (sp == 0) {
      sp = mt() % (s - 2) + 2;
    } else {
      if (sp > 2) {
        sp = mt() % (sp - 2) + 2;
      }
    }

    vector<bool> to_keep;
    for (int j = 0; j < s; j++) {
      to_keep.push_back(true);
    }

    int count = 0;
    while (count < sp) {
      int r = mt() % s;
      if (to_keep[r]) {
        to_keep[r] = false;
        count++;
      }
    }
    Puz dp = p.getMinor(&to_keep, NULL);
    int res = (dp.checkPuz(CHECK_FULL, strong) ? 1 : 0);
    fit += res * weight_function(sp, use_exp_weights);
  }

  return tuple<double, double>{fit, fit_max};
}

// ======================================================================
//
//  Implementation of AdaptiveSubsetTester
//
// ======================================================================

AdaptiveSubsetFitTester::AdaptiveSubsetFitTester(unsigned int s, int seed, bool use_exp_weights,
                                                 bool use_likelihood)
    : s(s), seed(seed), use_exp_weights(use_exp_weights), use_likelihood(use_likelihood) {
  srand(seed);  // Use the same seed for each size.

  // Only allow as many subsets as can be uniquely constructed.  At
  // most # of subsets of size s excluding empty and singletons.
  int num_subsets = DEFAULT_SUBSETS;

  if (num_subsets > pow(2, s) - 1 - s) {
    num_subsets = pow(2, s) - 1 - s;
  }

  for (int i = 0; i < num_subsets; i++)
    subsets.push_back(subset_t{vector<bool>(), vector<int>(), 0, 0, 0});

  if (num_subsets > 0) {
    // Set first subset to be the full subset.
    for (int r = 0; r < s; r++) {
      subsets[0].subset.push_back(true);
    }
    subsets[0].size = s;
  }

  int i = 1;
  while (i < num_subsets) {
    int size_target = rand() % (s - 2) + 2;
    int count = 0;
    subsets[i].size = size_target;
    // Construct random subset with size_target elements.
    subsets[i].subset = vector<bool>();
    for (int r = 0; r < s; r++) {
      subsets[i].subset.push_back(true);
    }
    while (count < s - size_target) {
      int r = rand() % s;
      if (subsets[i].subset[r]) {
        subsets[i].subset[r] = false;
        count++;
      }
    }

    // Check uniqueness.
    int matches = 1;
    for (int j = 0; j < i && matches != 0; j++)
      matches = samplecmp(subsets[i].subset, subsets[j].subset, s);

    if (matches != 0)
      i++;
  }

  for (int i = 0; i < num_subsets; i++) {
    for (int j = i + 1; j < num_subsets; j++) {
      int res = samplecmp(subsets[i].subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));
      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }
  }
}

void touch_subset(subset_t& ss, int val, unsigned int num_subsets) {
  ss.ave_sat = (ss.ave_sat * (num_subsets - 1) + val) / num_subsets;

  if (ss.ave_sat < RAISE_THRESH && ss.ave_sat > LOWER_THRESH) {
    ss.lifetime = 0;
  }

  ss.lifetime++;
}

bool AdaptiveSubsetFitTester::lower(unsigned int i) {
  vector<bool> subset = subsets[i].subset;
  int num_subsets = subsets.size();
  int size = subsets[i].size;

  if (size == s) {
    // XXX - Want to maintain full set as tested set, could spawn
    // subsets of it instead though.
    subsets[i].lifetime = 0;
    return false;
  }
  if (size == 2) {
    // Cannot lower sets below 2.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO LOWER SIZE 2 -- " << endl;
    return false;
  }

  // Look for a unique superset.
  vector<bool> curr_subset = subset;
  int count = 0;
  int j = rand() % size;
  int idx = 0;
  while (count < j) {
    count += (curr_subset[idx] ? 1 : 0);
    idx++;
  }
  if (idx > 0)
    idx--;  // Index of jth false.

  count = 0;
  while (count < size) {
    while (!curr_subset[idx]) {
      idx = (idx + 1) % s;
    }

    curr_subset[idx] = false;

    if (countOccurs(curr_subset) == 0)
      break;

    curr_subset[idx] = true;
    idx = (idx + 1) % s;

    count++;
  }

  if (count >= size) {
    // All subsets are present, just reset lifetime.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO LOWER SIZE - no space at size " << size-1 << " -- " << endl;
    return false;
  } else {
    // Otherwise, update info for the new subset.
    subsets[i].subset = curr_subset;
    subsets[i].ave_sat = 0;
    subsets[i].lifetime = 0;
    subsets[i].size--;
    subsets[i].neighbors = vector<int>();
    // Update neighbors

    // Delete edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;

      auto it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1));

      if (it != subsets[j].neighbors.end()) {
        int size = subsets[j].neighbors.size();
        subsets[j].neighbors.erase(it);
        assert(subsets[j].neighbors.size() == size - 1);
      }

      it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }
    }

    // Add edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;
      int res = samplecmp(curr_subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));

      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }

    return true;
  }
}

bool AdaptiveSubsetFitTester::raise(unsigned int i) {
  vector<bool> subset = subsets[i].subset;
  int num_subsets = subsets.size();
  int size = subsets[i].size;

  if (s == size) {
    return false;  // Cannot raise the full set.
  }

  // Look for a unique superset.
  vector<bool> curr_subset = subset;
  int count = 0;
  int j = rand() % (s - size);
  int idx = 0;
  while (count < j) {
    count += (!curr_subset[idx] ? 1 : 0);
    idx++;
  }
  if (idx > 0)
    idx--;  // Index of jth false.

  count = 0;
  while (count < s - size) {
    while (curr_subset[idx]) {
      idx = (idx + 1) % s;
    }

    curr_subset[idx] = true;

    if (countOccurs(curr_subset) == 0)
      break;

    curr_subset[idx] = false;
    idx = (idx + 1) % s;

    count++;
  }

  if (count >= s - size) {
    // All supersets are present, just reset lifetime.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO RAISE SIZE - no space at size " << size-1 << " -- " << endl;
    return false;
  } else {
    // Otherwise, update info for the new subset.
    subsets[i].subset = curr_subset;
    assert(countOccurs(subsets[i].subset) == 1);
    subsets[i].ave_sat = 0;
    subsets[i].lifetime = 0;
    subsets[i].size++;
    subsets[i].neighbors = vector<int>();
    // Update neighbors

    // Delete edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;

      auto it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }

      it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }

      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1)) ==
             subsets[j].neighbors.end());
      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1)) ==
             subsets[j].neighbors.end());
    }

    // Add edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;
      int res = samplecmp(curr_subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));

      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }

    return true;
  }
}

int AdaptiveSubsetFitTester::countOccurs(const vector<bool>& subset) const {
  int num_found = 0;
  for (int j = 0; j < subsets.size() && num_found < 2; j++) {
    if (samplecmp(subset, subsets[j].subset, s) == 0)
      num_found++;
  }

  return num_found;
}

void AdaptiveSubsetFitTester::updateSubsets() {
  if (last_update < UPDATE_RATE) {
    last_update++;
    return;
  }
  last_update = 0;

  int raised = 0;
  int lowered = 0;
  int raise_fail = 0;
  int lower_fail = 0;
  int total_size = 0;

  unsigned int size_histo[s + 1];
  for (int i = 0; i <= s; i++) {
    size_histo[i] = 0;
  }
  size_histo[s]++;
  unsigned int row_histo[s];
  for (int i = 0; i < s; i++) {
    row_histo[i] = 0;
  }

  int num_subsets = subsets.size();
  for (int i = 1; i < num_subsets; i++) {  // Skips full set.
    double ave_sat = subsets[i].ave_sat;
    int lifetime = subsets[i].lifetime;
    if (ave_sat > RAISE_THRESH && lifetime >= UPDATE_RATE) {
      // cout << "Found candidate for raising " << i << " " << lifetime << endl;
      if (raise(i)) {
        raised++;
      } else {
        raise_fail++;
      }
    } else if (ave_sat < LOWER_THRESH && lifetime >= UPDATE_RATE) {
      // cout << "Found candidate for lowering " << i << " " << lifetime << endl;
      if (lower(i)) {
        lowered++;
      } else {
        lower_fail++;
      }
    }
    size_histo[subsets[i].size]++;
    for (int j = 0; j < s; j++) {
      row_histo[j] += (subsets[i].subset[j] ? 1 : 0);
    }
    total_size += subsets[i].size;
  }

  cout << "----- Evolving Subsets ---> ";
  cout << "Raise: " << raised << "(failed: " << raise_fail << ") "
       << "Lower: " << lowered << "(failed: " << lower_fail << ") "
       << "Average size: " << total_size / (double) subsets.size() << endl;
  /*
  cout << "Size selection: " << endl;
  for (int i = 0; i <= s; i++){
    cout << i << ": " << size_histo[i] << endl;
  }
  cout << "Row selection: " << endl;
  for (int i = 0; i < s; i++){
    cout << i << ": " << row_histo[i] << endl;
    }*/
}

tuple<double, double> AdaptiveSubsetFitTester::getFit(const Puz& p) {
  double fit = 0;
  double fit_max = 0;
  assert(s == p.getHeight());

  // cout << "STARTING FITGET" << endl;

  int num_subsets = subsets.size();
  vector<bool> covered(num_subsets, false);

  int saved = 0;
  for (int i = 0; i < num_subsets; i++) {
    // Skip subsets already processed.
    if (covered[i]) {
      saved++;
      continue;
    }

    // cout << "i = " << i<< endl;
    Puz dp = p.getMinor(&(subsets[i].subset), NULL);
    int res = (dp.checkStrongUSP() ? 1 : 0);
    double weight = weight_function(subsets[i].size, use_exp_weights);
    fit += res * weight;
    fit_max += weight;

    // Account for and mark all the subsets covered by this one.
    covered[i] = true;
    touch_subset(subsets[i], res, num_subsets);
    for (int j = 0; j < subsets[i].neighbors.size(); j++) {
      int raw_idx = subsets[i].neighbors[j];
      int idx = fabs(raw_idx) - 1;

      if (res == 1 && raw_idx > 0 && !covered[idx]) {
        covered[idx] = true;
        Puz dp = p.getMinor(&(subsets[idx].subset), NULL);
        fit += weight_function(subsets[idx].size, use_exp_weights);
        fit_max += weight_function(subsets[idx].size, use_exp_weights);
        touch_subset(subsets[idx], 1, num_subsets);
      } else if (res == 0 && raw_idx < 0 && !covered[idx]) {
        covered[idx] = true;
        fit += 0;
        fit_max += weight_function(subsets[idx].size, use_exp_weights);
        touch_subset(subsets[idx], 0, num_subsets);
      }
    }
  }

  // cout << "saved: " << saved << "/" << num_subsets << endl;
  // cout << "ENDING FITGET" << endl;
  return tuple<double, double>{fit, fit_max};
}

void AdaptiveSubsetFitTester::evolveFitness() {
  updateSubsets();
}

// ======================================================================
//
//  Implementation of CachedAdaptiveSubsetTester
//
// ======================================================================

#define INDEX(r, c, v) ((r) * (3 * k) + (c) *3 + (v))

CachedAdaptiveSubsetFitTester ::CachedAdaptiveSubsetFitTester(unsigned int s, unsigned int k,
                                                              int seed,
                                                              unsigned int default_subsets,
                                                              vector<vector<bool> >* skip)
    : s(s), k(k), seed(seed), skip(skip) {
  if (default_subsets == 0) {
    default_subsets = DEFAULT_SUBSETS;
  }

  cached_p = Puz(s, k);

  srand(seed);  // Use the same seed for each size.
  dervs = vector<double>(s * k * 3);
  fitness_cache = tuple<double, double>{-1, -1};

  // Only allow as many subsets as can be uniquely constructed.  At
  // most # of subsets of size s excluding empty and singletons.
  int num_subsets = default_subsets;
  // cout << "num_subsets: " << num_subsets;
  if (num_subsets > pow(2, s) - 1 - s) {
    num_subsets = pow(2, s) - 1 - s;
  }

  for (int i = 0; i < num_subsets; i++)
    subsets.push_back(subset_cache_t{vector<bool>(), vector<int>(), 0, 0, 0, -1, false,
                                     vector<short>(3 * k * s, -1)});

  if (num_subsets > 0) {
    // Set first subset to be the full subset.
    for (int r = 0; r < s; r++) {
      subsets[0].subset.push_back(true);
    }
    subsets[0].size = s;
  }

  int i = 1;
  while (i < num_subsets) {
    int size_target = rand() % (s - 2) + 2;
    int count = 0;
    subsets[i].size = size_target;
    // Construct random subset with size_target elements.
    subsets[i].subset = vector<bool>();
    for (int r = 0; r < s; r++) {
      subsets[i].subset.push_back(true);
    }
    while (count < s - size_target) {
      int r = rand() % s;
      if (subsets[i].subset[r]) {
        subsets[i].subset[r] = false;
        count++;
      }
    }

    // Check uniqueness.
    int matches = 1;
    for (int j = 0; j < i && matches != 0; j++)
      matches = samplecmp(subsets[i].subset, subsets[j].subset, s);

    if (matches != 0)
      i++;
  }

  // Sort in decreasing order so that subsets which are SUSPs
  // Propagate most quickly.
  sort(subsets.begin(), subsets.end(),
       [s](const subset_cache_t& a, const subset_cache_t& b) -> bool {
         double a_val = fabs(a.size - 2 - 0.5 * (s - 2)) - 0.5 * (a.size > s * 0.5 ? 1 : 0);
         double b_val = fabs(b.size - 2 - 0.5 * (s - 2)) - 0.5 * (b.size > s * 0.5 ? 1 : 0);
         // return a_val > b_val; // Outer to inner.
         // return a.size < b.size; // Small to large.
         return a.size > b.size;  // Large to small.
       });

  for (int i = 0; i < num_subsets; i++) {
    for (int j = i + 1; j < num_subsets; j++) {
      int res = samplecmp(subsets[i].subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));
      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }
  }
}

bool CachedAdaptiveSubsetFitTester::lower(unsigned int i) {
  vector<bool> subset = subsets[i].subset;
  int num_subsets = subsets.size();
  int size = subsets[i].size;

  if (size == s) {
    // XXX - Want to maintain full set as tested set, could spawn
    // subsets of it instead though.
    subsets[i].lifetime = 0;
    return false;
  }
  if (size == 2) {
    // Cannot lower sets below 2.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO LOWER SIZE 2 -- " << endl;
    return false;
  }

  /*
  for (int r = 0; r < s; r++){
    cout << subsets[i].subset[r];
  }
  cout << endl;
  */

  // Look for a random open subset of this subset with one fewer rows.
  vector<bool> curr_subset = subset;
  int count = 0;
  int j = rand() % size;
  int idx = 0;
  while (count < j) {
    count += (curr_subset[idx] ? 1 : 0);
    idx++;
  }
  if (idx > 0)
    idx--;  // Index of jth false.

  count = 0;
  while (count < size) {
    while (!curr_subset[idx]) {
      idx = (idx + 1) % s;
    }

    curr_subset[idx] = false;

    if (countOccurs(curr_subset) == 0)
      break;

    curr_subset[idx] = true;
    idx = (idx + 1) % s;

    count++;
  }

  if (count >= size) {
    // All subsets are already being used, just reset lifetime. // XXX - Could delete it.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO LOWER SIZE - no space at size " << size-1 << " -- " << endl;
    return false;
  } else {
    /*
    for (int r = 0; r < s; r++){
      cout << subsets[i].subset[r];
    }
    cout << endl;
    */

    // Otherwise, update info for the new subset.
    subsets[i].subset = curr_subset;
    assert(countOccurs(subsets[i].subset) == 1);
    subsets[i].ave_sat = 0;
    subsets[i].lifetime = 0;
    subsets[i].size--;
    subsets[i].neighbors = vector<int>();

    // Update neighbors

    // Delete edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;

      /*
      for (int raw_idx: subsets[j].neighbors){
  cout << " " << raw_idx;
      }
      cout << endl;
      */

      auto it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1));

      if (it != subsets[j].neighbors.end()) {
        // cout << "Deleting from " << j << endl;
        subsets[j].neighbors.erase(it);
      }

      it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }

      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1)) ==
             subsets[j].neighbors.end());
      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1)) ==
             subsets[j].neighbors.end());
    }

    // Add edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;
      int res = samplecmp(curr_subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));

      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }

    // Set flag in cache.
    subsets[i].fitness_cache = -1;
    subsets[i].dervs_valid = false;
    subsets[i].dervs_cache = vector<short>(3 * s * k, -1);
    return true;
  }
}

// XXX - Update
bool CachedAdaptiveSubsetFitTester::raise(unsigned int i) {
  vector<bool> subset = subsets[i].subset;
  int num_subsets = subsets.size();
  int size = subsets[i].size;

  if (s == size) {
    return false;  // Cannot raise the full set.
  }

  // Look for a unique superset.
  vector<bool> curr_subset = subset;
  int count = 0;
  int j = rand() % (s - size);
  int idx = 0;
  while (count < j) {
    count += (!curr_subset[idx] ? 1 : 0);
    idx++;
  }
  if (idx > 0)
    idx--;  // Index of jth false.

  count = 0;
  while (count < s - size) {
    while (curr_subset[idx]) {
      idx = (idx + 1) % s;
    }

    curr_subset[idx] = true;

    if (countOccurs(curr_subset) == 0)
      break;

    curr_subset[idx] = false;
    idx = (idx + 1) % s;

    count++;
  }

  if (count >= s - size) {
    // All supersets are present, just reset lifetime.
    subsets[i].lifetime = 0;
    // cout << "-- FAILED TO RAISE SIZE - no space at size " << size-1 << " -- " << endl;
    return false;
  } else {
    // Otherwise, update info for the new subset.
    subsets[i].subset = curr_subset;
    assert(countOccurs(subsets[i].subset) == 1);
    subsets[i].ave_sat = 0;
    subsets[i].lifetime = 0;
    subsets[i].size++;
    subsets[i].neighbors = vector<int>();
    // Update neighbors

    // Delete edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;

      auto it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }

      it = find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1));

      if (it != subsets[j].neighbors.end()) {
        subsets[j].neighbors.erase(it);
      }

      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), (i + 1)) ==
             subsets[j].neighbors.end());
      assert(find(subsets[j].neighbors.begin(), subsets[j].neighbors.end(), -(i + 1)) ==
             subsets[j].neighbors.end());
    }

    // Add edges.
    for (int j = 0; j < num_subsets; j++) {
      if (i == j)
        continue;
      int res = samplecmp(curr_subset, subsets[j].subset, s);
      if (res == -1) {
        subsets[i].neighbors.push_back(-(j + 1));
        subsets[j].neighbors.push_back((i + 1));

      } else if (res == 1) {
        subsets[i].neighbors.push_back((j + 1));
        subsets[j].neighbors.push_back(-(i + 1));
      }
    }

    subsets[i].fitness_cache = -1;
    subsets[i].dervs_valid = false;
    subsets[i].dervs_cache = vector<short>(3 * s * k, -1);

    return true;
  }
}

int CachedAdaptiveSubsetFitTester::countOccurs(const vector<bool>& subset) const {
  int num_found = 0;
  for (int j = 0; j < subsets.size() && num_found < 2; j++) {
    if (samplecmp(subset, subsets[j].subset, s) == 0)
      num_found++;
  }

  return num_found;
}

void CachedAdaptiveSubsetFitTester::updateSubsets() {
  if (last_update < UPDATE_RATE) {
    last_update++;
    return;
  }
  last_update = 0;

  int raised = 0;
  int lowered = 0;
  int raise_fail = 0;
  int lower_fail = 0;
  double total_size = 0;

  unsigned int size_histo[s + 1];
  for (int i = 0; i <= s; i++) {
    size_histo[i] = 0;
  }
  size_histo[s]++;
  unsigned int row_histo[s];
  for (int i = 0; i < s; i++) {
    row_histo[i] = 0;
  }

  int num_subsets = subsets.size();
  for (int i = 0; i < num_subsets; i++) {
    double ave_sat = subsets[i].ave_sat;
    int lifetime = subsets[i].lifetime;
    if (ave_sat > RAISE_THRESH && lifetime >= UPDATE_RATE) {
      // cout << "Found candidate for raising " << i << " " << lifetime << endl;
      if (raise(i)) {
        raised++;
      } else {
        raise_fail++;
      }
    } else if (ave_sat < LOWER_THRESH && lifetime >= UPDATE_RATE) {
      // cout << "Found candidate for lowering " << i << " " << lifetime << endl;
      if (lower(i)) {
        lowered++;
      } else {
        lower_fail++;
      }
    }
    size_histo[subsets[i].size]++;
    for (int j = 0; j < s; j++) {
      row_histo[j] += (subsets[i].subset[j] ? 1 : 0);
    }
    total_size += subsets[i].size;
  }

  cout << "----- Evolving Subsets ---> ";
  cout << "Raise: " << raised << "(failed: " << raise_fail << ") "
       << "Lower: " << lowered << "(failed: " << lower_fail << ") "
       << "Average size: " << total_size / (double) subsets.size() << endl;
  /*
  cout << "Size selection: " << endl;
  for (int i = 0; i <= s; i++){
    cout << i << ": " << size_histo[i] << endl;
  }
  cout << "Row selection: " << endl;
  for (int i = 0; i < s; i++){
    cout << i << ": " << row_histo[i] << endl;
    }*/
}

void CachedAdaptiveSubsetFitTester::verifyCache(bool do_derv) {
  // cout << "do_derv: " << do_derv << endl;
  // displayCache();

  int i = 0;
  for (subset_cache_t& subset : subsets) {
    /*
    cout << "i: " << i << endl;
    for (int r = 0; r < s; r++){
      cout << subset.subset[r];
    }
    cout << endl;
    assert(subset.size >= 2);
    */

    assert(countOccurs(subset.subset) == 1);

    for (int raw_idx : subset.neighbors) {
      int idx = fabs(raw_idx) - 1;
      // cout << "idx: " << idx << endl;
      subset_cache_t* n = &subsets[idx];

      auto it = find(n->neighbors.begin(), n->neighbors.end(), (raw_idx < 0 ? (i + 1) : -(i + 1)));

      assert(it != n->neighbors.end());

      assert(samplecmp(subset.subset, n->subset, s) == (raw_idx < 0 ? -1 : 1));
    }

    assert(subset.fitness_cache >= 0);
    for (int r = 0; r < s; r++) {
      if (subset.subset[r]) {
        if (do_derv) {
          for (int c = 0; c < k; c++) {
            if (skip != NULL && (*skip)[r][c])
              continue;
            for (int v = 0; v < 3; v++) {
              assert(subset.dervs_cache[INDEX(r, c, v)] >= 0);
              assert(subset.dervs_cache[INDEX(r, c, v)] <= 1);
              assert(dervs[INDEX(r, c, v)] + get<0>(fitness_cache) <= get<1>(fitness_cache));
            }
          }
        }
      }
    }
    i++;
  }
}

bool CachedAdaptiveSubsetFitTester::invalidateCache(const Puz& p, bool do_derv) {
  bool changed_rows[s];
  bool changed = false;

  for (int r = 0; r < s; r++) {
    changed_rows[r] = false;
    for (int c = 0; c < k && !changed_rows[r]; c++)
      if (p.get(r, c) != cached_p.get(r, c)) {
        changed_rows[r] = true;
        changed = true;
      }
  }

  changed = changed || (get<0>(fitness_cache) == -1);

  // cout << "changed = " << changed << endl;
  int to_calc = 0;
  for (subset_cache_t& subset : subsets) {
    bool must_calc = false;
    for (int r = 0; r < s; r++) {
      // cout << subset.fitness_cache << endl;
      if (subset.subset[r] && subset.fitness_cache < 0) {
        changed = true;
        must_calc = true;
      }
      if (subset.subset[r] && do_derv && !subset.dervs_valid) {
        changed = true;
        must_calc = true;
      }
      if (changed_rows[r] && subset.subset[r]) {
        subset.fitness_cache = -1;
        // Implies the dervs of this row are also invalid.
        subset.dervs_valid = false;
        subset.dervs_cache = vector<short>(3 * s * k, -1);
        must_calc = true;
      }
    }
    if (must_calc)
      to_calc++;
  }

  if (changed)
    fitness_cache = tuple<double, double>{-1, -1};

  // About 50% of the subsets should be modified each time.  This is
  // because we sample the sets randomly and each row is 50% likely to
  // be in a given set.
  if (to_calc > 0 && do_derv)
    printf("Subsets to recalculate: %0.2f%%\n", 100 * to_calc / (double) subsets.size());

  return changed;
}

void CachedAdaptiveSubsetFitTester::evolveFitness() {
  updateSubsets();
}

tuple<double, double> CachedAdaptiveSubsetFitTester::getFit(const Puz& p) {
  assert(p.getHeight() == s && p.getWidth() == k);

  // cout << "In getFit"  << endl;
  Puz p2(p);
  // cout << "Checking Cache" << endl;
  if (invalidateCache(p2, false)) {
    // cout << "Cache invalid" << endl;
    updateFit(p, false);
    cached_p = p2;
  }

  return fitness_cache;
}

double CachedAdaptiveSubsetFitTester::getFitDerv(const Puz& p, unsigned int r, unsigned int c,
                                                 unsigned int val) {
  assert(p.getHeight() == s && p.getWidth() == k);
  assert(skip == NULL || !((*skip)[r][c]));

  // cout << "In getFitDerv" << endl;
  Puz p2(p);

  if (invalidateCache(p2, true)) {
    // cout << "getFitDerv Cache invalid" << endl;

    auto start = chrono::high_resolution_clock::now();
    updateFit(p, true);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Seconds to calc dervs: " << duration.count() / (double) 1000 << endl;

    cached_p = p2;
  }

  return dervs[INDEX(r, c, val - 1)];
}

void touch_subset_cache(subset_cache_t& ss, int val, unsigned int num_subsets) {
  ss.ave_sat = (ss.ave_sat * (UPDATE_RATE - 1) + val) / UPDATE_RATE;

  if (ss.ave_sat < RAISE_THRESH && ss.ave_sat > LOWER_THRESH) {
    ss.lifetime = 0;
  }

  ss.lifetime++;
}

void CachedAdaptiveSubsetFitTester::updateFit(const Puz& p, bool do_derv) {
  double fit = 0;
  double fit_max = 0;
  assert(s == p.getHeight());

  int num_subsets = subsets.size();

  if (do_derv) {
    dervs = vector<double>(3 * s * k, 0);
  }

  int total = 0;
  int calced = 0;

  // cout << num_subsets << endl;
  for (auto& ss : subsets) {
    // Update main for p.
    Puz dp = p.getMinor(&(ss.subset), NULL);
    if (ss.fitness_cache < 0)
      ss.fitness_cache = (dp.checkStrongUSP() ? 1 : 0);

    // Accumulate the fit for each subset.
    fit += ss.fitness_cache;
    fit_max += 1;

    // Update all derivatives, if necessary.
    if (do_derv) {
      int offset = 0;  // Tracks number of rows skipped b/c not in ss.subset.
      for (int r = 0; r < s; r++) {
        if (!ss.subset[r]) {
          // Update derivatives that don't overlap with ss.
          offset++;
          // if (ss.fitness_cache)
          for (int c = 0; c < k; c++) {
            if (skip != NULL && (*skip)[r][c])
              continue;
            total += 2;
            for (unsigned int v = 0; v < 3; v++) {
              dervs[INDEX(r, c, v)] += ss.fitness_cache;
            }
          }
        } else {
          // Update derivatives that overlap with ss.
          // cout << "r: " << r << " offset: " << offset << endl;
          for (int c = 0; c < k; c++) {
            if (skip != NULL && (*skip)[r][c]) {
              continue;
            }
            // cout << dp->getHeight() << " by " << dp->getWidth() << endl;
            unsigned int curr = dp.get(r - offset, c);
            total += 2;
            for (unsigned int v = 0; v < 3; v++) {
              if (ss.dervs_cache[INDEX(r, c, v)] < 0) {
                // This value is not already know so set it.
                if (curr == v + 1) {
                  // Keep current value, no change in fitness.
                  ss.dervs_cache[INDEX(r, c, v)] = ss.fitness_cache;
                } else {
                  calced++;
                  // Try new value, recalculate fitness.
                  dp.set(r - offset, c, v + 1);
                  ss.dervs_cache[INDEX(r, c, v)] = (dp.checkStrongUSP() ? 1 : 0);
                }
              }
              // Accumulate this derivative.
              dervs[INDEX(r, c, v)] += ss.dervs_cache[INDEX(r, c, v)];
            }
            // Reset the entry to original.
            dp.set(r - offset, c, curr);
          }
        }
      }
      // Mark valid.
      ss.dervs_valid = true;
    }

    touch_subset_cache(ss, ss.fitness_cache, num_subsets);  // XXX - calculated but not used.

    // Account for and mark all the subsets covered by this one.
    for (int raw_idx : ss.neighbors) {
      int idx = fabs(raw_idx) - 1;
      subset_cache_t* n = &subsets[idx];
      /*
      cout << "fit_max: " << fit_max << endl;
      cout << "raw_idx: " << raw_idx << endl;
      cout << "idx:     " << idx << endl;

      cout << "ss: ";
      for (int i = 0; i < s; i++){
  cout << ss.subset[i];
      }
      cout << " size: " << ss.size;
      cout << endl;

      cout << "n:  ";
      for (int i = 0; i < s; i++){
  cout << n->subset[i];
      }
      cout << endl;
      */
      assert(samplecmp(ss.subset, n->subset, s) == (raw_idx < 0 ? -1 : 1));

      if (raw_idx != 0) {
        int target = (raw_idx > 0 ? 1 : 0);
        // cout << "target: " << target << endl;
        if (ss.fitness_cache == target) {
          if (n->fitness_cache < 0) {
            // Override with target.
            n->fitness_cache = target;
          }
          assert(n->fitness_cache != (1 - target));
        }

        // Check dervs.
        if (do_derv) {
          for (int r = 0; r < s; r++) {
            if (!(n->subset[r] && ss.subset[r]))
              continue;
            for (int c = 0; c < k; c++) {
              if (skip != NULL && (*skip)[r][c]) {
                continue;
              }
              for (unsigned int v = 0; v < 3; v++) {
                int res = ss.dervs_cache[INDEX(r, c, v)];
                if (res == target) {
                  if (n->dervs_cache[INDEX(r, c, v)] < 0) {
                    n->dervs_cache[INDEX(r, c, v)] = target;
                  }
                  assert(n->dervs_cache[INDEX(r, c, v)] != (1 - target));
                }
              }
            }
          }
        }
      }
    }
  }

  assert(fit <= fit_max);
  fitness_cache = tuple<double, double>{fit, fit_max};

  // Adjust to be change in fitness.
  if (do_derv)
    for (int r = 0; r < s; r++)
      for (int c = 0; c < k; c++) {
        if (skip != NULL && (*skip)[r][c])
          continue;
        for (unsigned int v = 0; v < 3; v++) {
          assert(dervs[INDEX(r, c, v)] <= fit_max);
          dervs[INDEX(r, c, v)] -= fit;
        }
      }

  if (total > 0)
    cout << "Actual check() evals: " << calced << "/" << total << endl;
  verifyCache(do_derv);
}

void CachedAdaptiveSubsetFitTester::displayCache() {
  double fit, fit_max;

  tie(fit, fit_max) = fitness_cache;

  cout << "Total: " << fit << "/" << fit_max << endl;
  // Adjust to be change in fitness.
  for (int r = 0; r < s; r++) {
    for (int c = 0; c < k; c++) {
      cout << "(";
      for (unsigned int v = 0; v < 3; v++) {
        if (cached_p.get(r, c) == v + 1) {
          cout << "\033[1;34m";
        } else if (dervs[INDEX(r, c, v)] > 0) {
          cout << "\033[1;32m";
        }
        printf("%5.0f\033[0m ", dervs[INDEX(r, c, v)]);
      }
      cout << ")";
    }
    cout << endl;
  }
}

void CachedAdaptiveSubsetFitTester ::displayCacheDifference(
    const CachedAdaptiveSubsetFitTester& prev) {
  double curr_fit, curr_fit_max;
  double prev_fit, prev_fit_max;

  tie(curr_fit, curr_fit_max) = fitness_cache;
  tie(prev_fit, prev_fit_max) = prev.fitness_cache;

  cout << "Total: " << curr_fit << " (" << (curr_fit - prev_fit) << ")/" << curr_fit_max << " ("
       << (curr_fit_max - prev_fit_max) << ")" << endl;
  // Adjust to be change in fitness.
  for (int r = 0; r < s; r++) {
    for (int c = 0; c < k; c++) {
      cout << "(";
      for (unsigned int v = 0; v < 3; v++) {
        if (cached_p.get(r, c) == v + 1 && dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)] < 0) {
          cout << "\033[0;35m";  // Purple - selected in last step.
        } else if (cached_p.get(r, c) == v + 1) {
          cout << "\033[1;34m";  // Blue - current puzzle entry.
        } else if (dervs[INDEX(r, c, v)] > 0 &&
                   dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)] > 0) {
          cout << "\033[1;32m";  // Green - good first, good second.
        } else if (dervs[INDEX(r, c, v)] > 0) {
          cout << "\033[1;31m";  // Red - good first, bad second.
        } else if (dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)] > 0 &&
                   dervs[INDEX(r, c, v)] >= -(dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)])) {
          // Yellow - bad first, good second and close to changining first.
          cout << "\033[1;33m";
        } else if (dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)] > 0) {
          // Brown - bad first, good second, but far from changing first.
          cout << "\033[0;33m";
        }
        printf("%5.0f\033[1;37m ", dervs[INDEX(r, c, v)] - prev.dervs[INDEX(r, c, v)]);
      }
      cout << ")";
    }
    cout << endl;
  }
}

// ======================================================================
//
//  Implementation of TopDownFitTester
//
// ======================================================================

/**
 * Takes a optional argument for depth to search.
 */
TopDownFitTester::TopDownFitTester(int max_depth) : max_depth(max_depth) {}

tuple<double, double> TopDownFitTester::getFit(const Puz& p) {
  double fit_max = pow(p.getHeight(), p.getHeight() - 1) + 1;
  double fit = getFitHelper(p, 0);

  return tuple<double, double>{fit, fit_max};
}

double TopDownFitTester::getFitHelper(const Puz& p, int depth) {
  // cout << p << endl;

  if (p.getHeight() == 1) {
    return 1.0;
  }
  // cout << "Before check" << endl;

  if (p.checkStrongUSP()) {
    return pow(p.getHeight(), p.getHeight() - 1) + 1;
  } else {
    if (depth > max_depth) {
      return 0.0;
    }
    // cout << "After check" << endl;
    double fit = 0.0;
    vector<bool> to_keep;
    for (int r = 0; r < p.getHeight(); r++) {
      to_keep.push_back(true);
    }
    for (int r = 0; r < p.getHeight(); r++) {
      to_keep[r] = false;
      Puz next = p.getMinor(&to_keep, NULL);
      to_keep[r] = true;
      fit += getFitHelper(next, depth + 1);
    }
    return fit;
  }
}

// ======================================================================
//
//  Implementation of DefinitionFitTester
//
// ======================================================================

DefinitionFitTester::DefinitionFitTester() {}

tuple<double, double> DefinitionFitTester::getFit(const Puz& p) {
  int s = p.getHeight();
  int k = p.getWidth();

  double fit_max = 1;  // s!^2
  for (int i = 1; i <= s; i++) {
    fit_max *= i * i;
  }
  double fit = 0;

  vector<int> pi_2(s);
  vector<int> pi_3(s);

  // We look for pi_2 and pi_3 that witness that U is not a strong
  // USP.

  // Initialize pi_2 to the identity permutation.

  for (int i = 0; i < s; i++)
    pi_2[i] = i;

  // Loop over all permutations pi_2.
  bool go1 = true;
  for (; go1; go1 = next_permutation(pi_2.begin(), pi_2.end())) {
    // Initialize pi_3 to the identity permutation.
    for (int i = 0; i < s; i++)
      pi_3[i] = i;
    int go2 = true;

    // Loop over all permutations pi_3.
    for (; go2; go2 = next_permutation(pi_3.begin(), pi_3.end())) {
      // Skip if pi_1, pi_2, pi_3 are the same -- all identity.
      if (ident(pi_2, s) && ident(pi_3, s)) {
        fit += 1;
        continue;
      }

      // Check whether pi_1, pi_2, pi_3 touch only trues.
      double best = 0;
      for (int r = 0; r < s && best < 1; r++) {
        for (int c = 0; c < k && best < 1; c++) {
          // Not longer distributional without DistPuzzle.
          double one[3] = {0, 0, 0};
          double two[3] = {0, 0, 0};
          double thr[3] = {0, 0, 0};

          one[p.get(r, c) - 1] = 1;
          two[p.get(pi_2[r], c) - 1] = 1;
          thr[p.get(pi_3[r], c) - 1] = 1;

          double val = fabs(one[0]) * fabs(two[1]) * (fabs(thr[0]) + fabs(thr[1])) +
                       fabs(one[0]) * fabs(thr[2]) * (fabs(two[0]) + fabs(two[2])) +
                       fabs(two[1]) * fabs(thr[2]) * (fabs(one[1]) + fabs(one[2]));

          if (val > best) {
            best = val;
          }
        }
      }
      fit += best;
    }
  }
  return tuple<double, double>{fit, fit_max};
}

// ======================================================================
//
//  Implementation of TimingFitTester
//
// ======================================================================

TimingFitTester::TimingFitTester(unsigned int sample_size) : sample_size(sample_size) {}

tuple<double, double> TimingFitTester::getFit(const Puz& p) {
  double fit_max = numeric_limits<double>::infinity();  // Not very useful
  bool is_SUSP;

  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < sample_size; i++) {
    is_SUSP = p.checkStrongUSP();
  }
  auto stop = chrono::high_resolution_clock::now();

  auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
  double fit = duration.count() / (double) sample_size;

  if (is_SUSP) {
    return tuple<double, double>{fit_max, fit_max};
  } else {
    return tuple<double, double>{fit, fit_max};
  }
}

// ======================================================================
//
//  Implementation of PiecesFitTester
//
// ======================================================================

tuple<double, double> PiecesFitTester::getFit(const Puz& p) {
  map<set, int> pieces[3];

  int s = p.getHeight();
  int k = p.getWidth();
  double fit_max = 3 * (s - 1) * (s - 1);
  double fit = 0;

  for (int r = 0; r < s; r++) {
    set piece[3];
    for (int val = 1; val <= 3; val++)
      piece[val - 1] = create_empty_set();

    for (int c = 0; c < k; c++) {
      int e = p.get(r, c);
      piece[e - 1] = set_union(piece[e - 1], create_one_element_set(c));
    }

    for (int val = 1; val <= 3; val++) {
      if (pieces[val - 1].find(piece[val - 1]) == pieces[val - 1].end()) {
        pieces[val - 1][piece[val - 1]] = 1;
      } else {
        pieces[val - 1][piece[val - 1]]++;
      }
    }
  }

  for (int val = 1; val <= 3; val++) {
    // cout << "val = " << val << endl;
    for (const auto& piece : pieces[val - 1]) {
      // cout << piece.first << " " << piece.second << endl;
      fit += (piece.second - 1) * (piece.second - 1);
    }
  }

  fit = fit_max - fit;
  assert(fit >= 0);
  double scale = 1.0;
  return tuple<double, double>{fit * scale, fit_max * scale};
}

// ======================================================================
//
//  Implementation of TDMSizeFitTester
//
// ======================================================================

tuple<double, double> TDMSizeFitTester::getFit(const Puz& p) {
  bool shortcut = true;

  double s = p.getHeight();  // double b/c overflow of large values.
  int k = p.getWidth();

  // Puzzles with one or fewer rows are trivially SUSP.
  if (s <= 1) {
    return tuple<double, double>{1, 1};
  }

  // Parameters for process.
  unsigned int max_fail = 4;
  // Fitness just from # of off diagonal edges.
  unsigned int diag = 0;
  unsigned int near_diag = 1;
  unsigned int off_diag = 1;

  // Max fitness of final stage
  double size_fit_max =
      near_diag * 3 * (s * (s - 1)) + off_diag * (s * s * s - (s + 3 * (s * (s - 1))));

  double fit_max = 0;
  double fit = 0;

  // =================================================
  // Pre-process 1: Force special columns
  // Theta(s * require_special).
  int special = 0;
  int local_require_special = (require_special <= k ? require_special : k);

  bool seen[3] = {false, false, false};
  for (int c = 0; c < k && c < local_require_special; c++) {
    int counts[3] = {0, 0, 0};
    for (int r = 0; r < s; r++)
      counts[p.get(r, c) - 1]++;
    if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0)
      special++;
    for (int i = 0; i < 3; i++) {
      seen[i] = seen[i] || (counts[i] != 0);
    }
  }

  double remaining_max_fit = (s * (k + 1) + 1) * (max_fail + 1) * (size_fit_max + 1);
  fit_max += local_require_special * remaining_max_fit;
  if (seen[0] && seen[1] && seen[2] || local_require_special <= 1)
    fit += special * remaining_max_fit;

  // remaining_max_fit = (max_fail+1)*(size_fit_max+1);
  // unsigned int hs = 0;
  // if (!shortcut || fit == fit_max){
  //   Puz p2 = p;
  //   p2.sort();
  //   hs = hierarchically_special(p2,1);
  //   fit += (s * (k + 1) - hs) * remaining_max_fit;
  // }
  // fit_max += s * (k + 1) * remaining_max_fit;

  // =================================================
  // Pre-process 2: Force pairwise SUSP.
  // O(s^2 * k).
  remaining_max_fit = (s * (k + 1) + 1) * (size_fit_max + 1);

  // To give the function more granularity, also count the number of
  // pairs of rows which are not SUSP, and give a weight comparable
  // to the number maximum 3DM instance size.  To penalize copying
  // of rows to reduce the instance size.
  // XXX - Could be do USP pieces here?
  int fail = 0;
  if (!shortcut || fit == fit_max) {
    for (int r1 = 0; r1 < s - 1 && fail < max_fail; r1++) {
      for (int r2 = r1 + 1; r2 < s && fail < max_fail; r2++) {
        // Could do in time dependent of k using the already
        // simplified TDM.
        check_t res = check2(p, r1, r2, mode, strong);

        if (res != YES_CHECK)
          fail++;
      }
    }
  } else {
    fail = max_fail;
  }
  fit += (max_fail - fail) * remaining_max_fit;
  fit_max += max_fail * remaining_max_fit;

  // Do 2D matching simplification on every face of the
  // 3D matching instance and count the remaining edges in the
  // 3DM instance.
  if (!shortcut || fit == fit_max) {
    if (p.getSimplified()) {
      fit += size_fit_max - (p.getResults() - s);
    }

    else {
      TDM tdm(p, strong, TDM::NO_GPU);
      assert(mode != CHECK_FULL);
      if (mode == CHECK_OBVIOUS)
        tdm.simplify(TDM::LAZY);

      fit += size_fit_max - (tdm.count() - s);
    }
  }
  fit_max += size_fit_max;

  return tuple<double, double>{fit, fit_max};
}

// ======================================================================
//
//  Implementation of HSFitTester
//
// ======================================================================

tuple<double, double> HSFitTester::getFit(const Puz& p) {
  double s = p.getHeight();  // double b/c overflow of large values.
  int k = p.getWidth();

  // Puzzles with one or fewer rows are trivially SUSP.
  if (s <= 1) {
    return tuple<double, double>{1, 1};
  }

  Puz p2 = p;
  p2.sort();

  unsigned int res = hierarchically_special(p2);
  double max_fit = s * k;
  double fit = s * k - res;
  return tuple<double, double>{fit, max_fit};
}

// Factory function to create FitTester from type name
FitTester* createFitTesterFromType(const string& typeName) {
  // For now, return a default SubsetFitTester to avoid compilation errors
  // This needs to be properly implemented based on the actual requirements
  if (typeName == "SubsetFitTester") {
    return new SubsetFitTester(false);
  } else if (typeName == "TDMSizeFitTester") {
    return new TDMSizeFitTester(CHECK_FULL, false);
  } else if (typeName == "HSFitTester") {
    return new HSFitTester(CHECK_FULL, false);
  }
  // Default fallback
  return new SubsetFitTester(false);
}
