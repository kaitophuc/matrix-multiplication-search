/*
 * Author: Matt Anderson
 */

#include "special.h"
#include "Permutation.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

using namespace std;

Puz minor_puzzle(const Puz& p, int c, int r1, int r2) {
  Puz ret(r2 - r1, p.getWidth() - c);

  for (int i = 0; i < ret.getHeight(); i++) {
    for (int j = 0; j < ret.getWidth(); j++) {
      ret.set(i, j, p.get(r1 + i, c + j));
    }
  }
  return ret;
}

bool is_special_col(const Puz& p, int c, int r1, int r2) {
  int counts[3] = {0, 0, 0};
  for (int r = r1; r < r2; r++)
    counts[p.get(r, c) - 1]++;
  if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0)
    return true;
  return false;
}

bool is_special_col(const Puz& p, int c) {
  return is_special_col(p, c, 0, p.getHeight());
}

int count_special(const Puz& p) {
  int count = 0;
  for (int c = 0; c < p.getWidth(); c++)
    count += (is_special_col(p, c) ? 1 : 0);
  return count;
}

unsigned int hierarchically_special(const Puz& p, int r1, int r2, int c, int required) {
  // cout << r1 << ", " << r2 << ", " << c << endl;
  unsigned int fail_val = (r2 - r1) * (p.getWidth() - c);

  // No columns left.
  if (c >= p.getWidth())
    return fail_val;

  bool res = true;
  for (int i = 1; i < required && c + i < p.getWidth(); i++) {
    res = res && is_special_col(p, c + i, r1, r2);
  }
  if (!res) {
    return fail_val;
  }

  if (c == p.getWidth() - 1 && r1 + 1 != r2) {
    return fail_val;
  }

  // Column c isn't special.
  if (!is_special_col(p, c, r1, r2)) {
    // cout << "Failed (not special): " << minor_puzzle(p, c, r1, r2);
    return fail_val;
  }

  // Scan column.
  int val1 = p.get(r1, c);
  int r = r1 + 1;
  while (r < r2 && p.get(r, c) == val1)
    r++;

  // Only one value, check recursively.
  if (r == r2)
    return hierarchically_special(p, r1, r2, c + 1);

  int val2 = p.get(r, c);
  // Out of order.
  if (val1 > val2) {
    // cout << "Failed (out of order): " << minor_puzzle(p, c, r1, r2);
    return fail_val;
  }

  int r3 = r + 1;
  while (r3 < r2 && p.get(r3, c) == val2)
    r3++;
  // Out of order.
  if (r3 != r2) {
    // cout << "Failed (out of order): " << minor_puzzle(p, c, r1, r2);
    return fail_val;
  }

  // Special and in order, check recursively.
  return hierarchically_special(p, r1, r, c + 1) + hierarchically_special(p, r, r2, c + 1);
}

unsigned int hierarchically_special(const Puz& p, int required) {
  return hierarchically_special(p, 0, p.getHeight(), 0);
}

unsigned int hierarchically_special_unordered(const Puz& p, int required, vector<Puz>* ps) {
  int num = 0;
  // Iterate over all permutations of columns and check.
  Permutation perm(p.getWidth());
  bool stop = false;
  unsigned int best = -1;
  for (; !stop && (ps != NULL || best != 0); perm = perm.next()) {
    Puz p2(p.getHeight(), p.getWidth());
    for (int r = 0; r < p.getHeight(); r++) {
      for (int c = 0; c < p.getWidth(); c++) {
        p2.set(r, c, p.get(r, perm.apply(c)));
      }
    }

    p2.sort();
    // cout << p2 << endl;
    unsigned int res = (hierarchically_special(p2));
    if (res <= best) {
      best = res;
      if (res == 0) {
// cout << "hs for perm = " << perm << endl;
#pragma omp critical
        if (ps != NULL) {
          ps->push_back(p2);
          // stop = true;
        }
        num++;
      }
    }

    if (perm.isLast())
      stop = true;
  }

  // cout << "HS for " << num << " orderings" << endl;

  return best;
}
