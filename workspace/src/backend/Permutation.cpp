/*
  Implementation of a data structure representing a permutation.
  Uses a list of outputs indexed by input values. (Equivalent to one line representation)
  Author: Harper
*/

#include "Permutation.h"
#include <vector>
#include <utility>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <numeric>

using namespace std;

// Factorials from zero to 20(max that can fit in 64 bits) so time is not wasted calculating
constexpr size_t factorial[21] = {1,
                                  1,
                                  2,
                                  6,
                                  24,
                                  120,
                                  720,
                                  5040,
                                  40320,
                                  362880,
                                  3628800,
                                  39916800,
                                  479001600,
                                  6227020800,
                                  87178291200,
                                  1307674368000,
                                  20922789888000,
                                  355687428096000,
                                  6402373705728000,
                                  121645100408832000,
                                  2432902008176640000};

/*
  Default constructor, creates identity permutation of size 1.
*/
Permutation::Permutation() {
  s = 1;
  outs = {0};
}

/*
  Parameterized Constructor, creates an identity permutation of a given size.
*/
Permutation::Permutation(int setSize) {
  assert(setSize >= 1);

  s = setSize;
  outs = vector<int>(s);
  for (int i = 0; i < s; i++) {
    outs[i] = i;
  }
}

/*
  Parameterized Constructor, creates a permutation given a size and appropriate ordered output
  vector. Example: (2, {1, 0}) would be the permutation of size two which transposes 1 and 2.
*/
Permutation::Permutation(const vector<int> outputs) {
  assert(validate(outputs));

  s = outputs.size();
  outs = outputs;
}

/*
  Copy constructor, used in overloading = operator.
*/
Permutation::Permutation(const Permutation& p) {
  s = p.s;
  outs = p.outs;
}

// Index constructor.
Permutation::Permutation(int64_t size, uint64_t index) : outs(size), s(size) {
  std::vector<int> elements(size);
  std::iota(elements.begin(), elements.end(), 0);

  int64_t permIndex = index % factorial[size] + 1;
  while (elements.size()) {
    int64_t r = 0, i = 0;
    for (i = elements.size() - 1; i >= 0; --i) {
      r = permIndex - i * factorial[elements.size() - 1];
      if (r <= factorial[elements.size() - 1] && r > 0)
        break;
    }

    permIndex = r;
    outs[size - elements.size()] = elements.at(i);
    elements.erase(elements.begin() + i);
  }
}

/*
  Overloads << operator to display one line representation.
 */
ostream& operator<<(ostream& os, const Permutation& p) {
  os << p.getStringOneLine();
  return os;
}

/*
  Overloads < operator to order Permutations.
  First orders based on size of permutation then on lexicographic order.
  Adapted from code written by Jingyu Yao.
 */
bool Permutation::operator<(const Permutation other) const {
  // Checking size
  if (s < other.s) {
    return true;
  }
  if (s > other.s) {
    return false;
  }

  // Checking lexicographic order
  for (int i = 0; i < s; i++) {
    if (outs[i] < other.outs[i]) {
      return true;
    }
    if (outs[i] > other.outs[i]) {
      return false;
    }
  }

  // Returns false if the two permutations are equal
  return false;
}

/*
  Returns string displaying Permutation as ordered list of all outputs.
  Format: (p(0) p(1) p(2) ... p(n-1))
*/
string Permutation::getStringOneLine() const {
  string out = "(" + to_string(outs[0]);
  for (int i = 1; i < s; i++) {
    out += " " + to_string(outs[i]);
  }
  out += ")";

  return out;
}

/*
  Returns string displaying Permutation as stacked lists of all inputs and corresponding outputs.
  Format: |0 1 2 ... (n-1)|
          |p0 p1 p2 ... p(n-1)|
*/
string Permutation::getStringTwoLine() const {
  string out = "|0";
  for (int i = 1; i < s; i++) {
    out += " " + to_string(i);
  }
  out += "|\n|" + to_string(outs[0]);
  for (int i = 1; i < s; i++) {
    out += " " + to_string(outs[i]);
  }
  out += "|";

  return out;
}

/*
  Returns string displaying Permutation as a product of disjoint cycles
  Format: (021)(34) or similar
*/
string Permutation::getStringDisjointCycles() const {
  vector<vector<int>> cycles = getDisjointCycles();

  string out = "";

  for (int i = 0; i < cycles.size(); i++) {
    out += "(" + to_string(cycles[i][0]);
    for (int j = 1; j < cycles[i].size(); j++) {
      out += " " + to_string(cycles[i][j]);
    }
    out += ")";
  }
  return out;
}

/*
  Returns string displaying Permutation as a product of transpositions.
  Format: (01)(24)(35) or similar.
*/
string Permutation::getStringTranspositions() const {
  vector<pair<int, int>> cycles = getTwoCycles();

  string out = "";

  for (int i = 0; i < cycles.size(); i++) {
    out +=
        string("(") + to_string(cycles[i].first) + " " + to_string(cycles[i].second) + string(")");
  }
  return out;
}

/*
  Returns a vector of vectors representing all disjoint cycles which make up this permutation
*/
vector<vector<int>> Permutation::getDisjointCycles() const {
  bool seen[s] = {false};
  vector<vector<int>> cycles;

  for (int i = 0; i < s; i++) {
    if (!seen[i] && apply(i) != i) {
      vector<int> cyc = {i};
      int current = i;

      while (apply(current) != i) {
        current = apply(current);
        seen[current] = true;
        cyc.push_back(current);
      }
      cycles.push_back(cyc);
    }
  }
  return cycles;
}

/*
  Returns a vector of length two vectors representing the transpositions which make up this perm
*/
vector<pair<int, int>> Permutation::getTwoCycles() const {
  vector<pair<int, int>> two;
  vector<vector<int>> disjoint = getDisjointCycles();

  for (int i = 0; i < disjoint.size(); i++) {
    for (int j = disjoint[i].size() - 1; j > 0; j--) {
      two.push_back(make_pair(disjoint[i][0], disjoint[i][j]));
    }
  }
  return two;
}

/*
  Applies permutation to a single number within it's range (1 to (size - 1))
*/
int Permutation::apply(int in) const {
  assert(!(in < 0 || in >= s));
  return outs[in];
}

/*
  Applies permutation to a given set passed as an array.
  Set should only contain numbers from 1 to the (size - 1) of this permutation.
*/
vector<int> Permutation::apply(const vector<int> nums) const {
  int inSize = nums.size();
  vector<int> out(inSize);

  for (int i = 0; i < nums.size(); i++) {
    out[i] = (apply(nums[i]));
  }
  return out;
}

/*
  Composes this Permutation with another of the same size, returns resulting Permutation.
*/
Permutation Permutation::compose(const Permutation other) const {
  assert(s == other.s);

  return Permutation(other.apply(outs));
}

/*
  Returns whether this permutation is the identity permutation for it's size.
*/
bool Permutation::isIdentity() const {
  for (int i = 0; i < s; i++) {
    if (outs[i] != i) {
      return false;
    }
  }
  return true;
}

/*
  Returns true if this permutation is the last in sequence, false otherwise
*/
bool Permutation::isLast() const {
  for (int i = 0; i < s - 1; i++) {
    if (outs[i] < outs[i + 1]) {
      return false;
    }
  }
  return true;
}

/*
  Generates and returns the next permutation in Knuthian order as new Permutation object.
*/
Permutation Permutation::next() const {
  // No steps necessary for permutations of size 1.
  if (s == 1) {
    return Permutation(1);
  }

  vector<int> newOuts = outs;  // Copy outs.

  // Find latest j s.t. outs[j] <= outs[j+1], and if no such j exists, loop back to identity
  // permutation.
  int j = s - 2;
  while (j >= 0 && newOuts[j] > newOuts[j + 1]) {
    j--;
  }
  if (j == -1) {
    return Permutation(s);
  }

  // Find latest k s.t. outs[j] <= outs[k], then swap the two.
  int k = s - 1;
  while (newOuts[j] > newOuts[k]) {
    k--;
  }

  swap(newOuts[j], newOuts[k]);

  // Reverse order of all elements past j+1
  for (int i = 0; i < floor((s - 1 - j) / 2); i++) {
    swap(newOuts[j + 1 + i], newOuts[s - 1 - i]);
  }

  return Permutation(newOuts);
}

/*
  Constructs and returns inverse of this Permutation as a new Permutation.
  Adapted from code written by Jingyu Yao.
 */
Permutation Permutation::invert() const {
  if (isIdentity()) {
    return Permutation(s);
  }

  vector<int> newOuts = outs;

  for (int i = 0; i < s; i++) {
    if (newOuts[i] < 0) {
      newOuts[i] = -newOuts[i] - 1;
    } else {
      int start = i;
      int prev = i;
      int curr = newOuts[i];
      while (curr != start) {
        int next = newOuts[curr];
        newOuts[curr] = -prev - 1;
        prev = curr;
        curr = next;
      }
      newOuts[start] = prev;
    }
  }

  // cout << newOuts << endl;
  return Permutation(newOuts);
}

/*
  Returns the size of this permutation
*/
int Permutation::getSize() const {
  return s;
}

/*
  Checks to see if passed list is a valid output list, i.e. that for a list of size n we have
  that all numbers from 0-(n-1) appear exactly once.
*/
bool Permutation::validate(const vector<int> nums) const {
  int size = nums.size();

  if (size < 1) {
    return false;
  }

  bool seen[size] = {false};
  int numUnique = 0;

  for (int n : nums) {
    if (n < 0 || n > size - 1) {
      return false;
    }
    if (seen[n]) {
      return false;
    } else {
      seen[n] = true;
      numUnique++;
    }
  }
  return numUnique == size;
}

// Gives the lexicolographical index of this permutation
uint64_t Permutation::getIndex() const {
  uint64_t index = 0;
  for (int64_t i = 0; i < outs.size(); i++) {
    uint64_t count = 0;  // Number of elements greater than that at i
    for (int64_t j = i + 1; j < outs.size(); j++)
      if (outs[i] > outs[j])
        count++;

    index += count * factorial[outs.size() - i - 1];
  }
  return index;
}
