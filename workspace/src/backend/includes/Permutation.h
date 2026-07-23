/*
  Describes a permutation data structure and associated methods.
  In particular, this struture represents a permutation on the set Z/Zn using a vector as a one-line
  representation. Ex: The permutation of size 3 which swaps (aka transposes) 1 and 2, can be
  represented in several way. Cycle: (12) Rules: 0->0 1->2 2->1 Two-Line: 0 1 2 0 2 1 One-Line: Same
  as two line, but with no upper row: 0 2 1. This permutation uses a vector to store a one-line
  representation. Author: Harper
*/

#include <vector>
#include <string>
#include <cstdint>

#pragma once

using namespace std;

class Permutation {
 private:
  vector<int> outs;  // vector representing one line permutation.
                     //  EX: Identity permutation of size 3 would be {0, 1, 2}
  int s;  // size of permutation (will always be <= 1)

  /*
    Returns a vector of vectors representing all disjoint cycles which make up this permutation
  */
  vector<vector<int>> getDisjointCycles() const;

  /*
    Returns a vector int pairs representing the transpositions which make up this perm
  */
  vector<pair<int, int>> getTwoCycles() const;

  /*
    Checks to see if passed list is a valid output list, i.e. that for a list of size n we have
    that all numbers from 0-(n-1) appear exactly once.
  */
  bool validate(const vector<int> nums) const;

 public:
  /*============================================================================

      Constructors

  ============================================================================*/

  /*
    Default constructor, creates identity permutation of size 1.
  */
  Permutation();

  /*
    Parameterized Constructor, creates an identity permutation of a given size.
  */
  Permutation(int size);

  /*
    Parameterized Constructor, creates a permutation given a size and appropriate ordered output
    vector. Example: (2, {1, 0}) would be the permutation of size two which transposes 1 and 2.
  */
  Permutation(const vector<int> outputs);

  /*
    Copy constructor, used in overloading = operator.
  */
  Permutation(const Permutation& p);

  /*
    Construxts the n-th lexicographical permutation with the given size.
Where the index of the identity permutation is 0 and that of the
greatest permutation is factorial(size) - 1. Inputing anything
larger than factorial(size) - 1 will just wrap the index around.

      size -> Size of the permutation
permutationIndex -> Index of the permutation
  */
  Permutation(int64_t size, uint64_t permutationIndex);

  /*============================================================================

      Display Functions (all return strings for flexibility)

  ============================================================================*/

  /*
    Returns string displaying Permutation as ordered list of all outputs.
    Format: (p(0) p(1) p(2) ... p(n-1))
  */
  string getStringOneLine() const;

  /*
    Returns string displaying Permutation as stacked lists of all inputs and corresponding outputs.
    Format: |0 1 2 ... (n-1)|
      |p0 p1 p2 ... p(n-1)|
  */
  string getStringTwoLine() const;

  /*
    Returns string displaying Permutation as a product of disjoint cycles
    Format: (021)(34) or similar
  */
  string getStringDisjointCycles() const;

  /*
    Returns string displaying Permutation as a product of transpositions.
    Format: (01)(24)(35) or similar.
  */
  string getStringTranspositions() const;

  /*============================================================================

      Operations

  ============================================================================*/

  /*
    Returns the size of this permutation
  */
  int getSize() const;

  /*
    Applies permutation to a single number within it's range (1 to size - 1)
  */
  int apply(int in) const;

  /*
    Applies permutation to a given set passed as an array.
    Set should only contain numbers from 1 to the (size - 1) of this permutation.
  */
  vector<int> apply(const vector<int> nums) const;

  /*
    Composes this Permutation with another of the same size in order (this)(other).
    Returns result as a new Permutation.
  */
  Permutation compose(const Permutation other) const;

  /*
    Returns whether this permutation is the identity permutation for it's size.
  */
  bool isIdentity() const;
  /*
    Returns true if this permutation is the last permutation of it's size.
    Last permutation looks like the reverse of the identity permutation (EX: [2, 1])
    Mainly useful when combined with next()
  */
  bool isLast() const;

  /*
    Generates and returns the next permutation in Knuthian order as new Permutation object.
  */
  Permutation next() const;

  /*
    Generates and returns the inverse of this Permutation.
   */
  Permutation invert() const;

  /*
    Overloads == operator
  */
  inline bool operator==(const Permutation other) const {
    return outs == other.outs;
  }

  /*
    Checks that the contents of this permutation are equal to those of the vector.
    For debugging purposes.
   */
  bool operator==(std::vector<int> const& other) const {
    return outs == other;
  }

  /*
    Overloads < operator to order Permutations.
    First orders based on size of Permutation and then on lexicographic order.
   */
  bool operator<(const Permutation other) const;

  /*
    Overloads > operator to order Permutations.
    First orders based on size of Permutation then on lexicographic order.
   */
  inline bool operator>(const Permutation other) const {
    return other < *this;
  }

  /*
    Overloads << operator to display one line representation
  */
  friend ostream& operator<<(ostream& os, const Permutation& p);

  /*
    Returns the lexicographical index of this permutation that can be
  used to construct it again using the index constructor.
  */
  uint64_t getIndex() const;
};
