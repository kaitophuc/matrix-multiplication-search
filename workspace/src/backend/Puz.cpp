#include "Puz.h"
#include <iostream>
#include <bits/stdc++.h>
#include <cstdlib>
#include "checker.h"
#include "fitness.h"
#include "prg.h"
#include "canonization.h"

#include <ctime>
#include <random>
#include <iterator>
#include <cstddef>

/**
 * Set s, k, data, and entries_per_row according to s, k.  Sets to
 * all 1's.
 */
void Puz::setSize(unsigned int s, unsigned int k) {
  assert(data == NULL);

  // cout << "s: " << s << endl;
  // cout << "k: " << k << endl;
  this->s = s;
  this->k = k;
  // we use bitmasks to store the puzzle.  Each entry is 2 bits.
  // With 64-bit integers, we can store 31 entries (ELTS_PER_ENTRY) .
  // We need to store k entries per row, so we need ceil(k/31) integers.
  // The number of entries per row is stored in entries_per_row.
  entries_per_row = (unsigned int) ceil(k / (double) ELTS_PER_ENTRY);
  // cout << "entries_per_row = " << entries_per_row << endl;
  // if (data != NULL) delete data;
  data = new e_type[s * entries_per_row];
  bzero(data, sizeof(e_type) * s * entries_per_row);
}

/**
 * Readed an extended puzzle from file and returns the wildcard coords.
 */
void Puz::readFromFile(const char* filename) {
  assert(filename != NULL);

  FILE* f = fopen(filename, "r");

  assert(f != NULL);

  char buff[256];

  int element;
  // first check whether this file is able to turn into a puzzle
  int bytes_read = fscanf(f, "%s\n", buff);
  assert(bytes_read > 0);

  unsigned int kp = strlen(buff);

  int sp = 1;
  if (buff[0] == '#')
    sp = 0;
  // loop until end of file using feof(f).  Build up puzzle DS.
  while (!feof(f)) {
    bytes_read = fscanf(f, "%s\n", buff);
    assert(bytes_read > 0);
    if (buff[0] == '#')
      continue;  // Comment
    if (kp != strlen(buff)) {
      fprintf(stderr, "Error: Puzzle rows not all same length.\n");
      assert(false);
    }
    for (unsigned int i = 0; i < kp; i++) {
      if (!(buff[i] == '*')) {
        element = buff[i] - '0';
        if (element != 1 && element != 2 && element != 3) {
          fprintf(stderr, "Error: Puzzle entries can only be 1 or 2 or 3.\n");
          assert(false);
        }
      }
    }
    sp++;
  }
  fclose(f);

  setSize(sp, kp);

  f = fopen(filename, "r");
  int r = 0;
  while (!feof(f)) {
    bytes_read = fscanf(f, "%s\n", buff);
    assert(bytes_read > 0);
    if (buff[0] == '#')
      continue;
    for (unsigned int c = 0; c < k; c++) {
      if (buff[c] == '*') {
        wildcards.push_back(tuple<int, int>{r, c});
        // initialize wildcards to 1.
        set(r, c, 1);
      } else {
        set(r, c, buff[c] - '0');
      }
    }
    r++;
  }
  fclose(f);
}

void Puz::readFromString(const char* str) {
  assert(str != NULL);

  int offset = 0;
  char buff[256];

  int element;
  // first check whether this file is able to turn into a puzzle
  int bytes_read = sscanf(str + offset, "%s\n", buff);
  assert(bytes_read > 0);

  unsigned int kp = strlen(buff);
  int sp = 0;

  do {
    bytes_read = sscanf(str + offset, "%s\n", buff);
    for (unsigned int i = 0; i < kp; i++) {
      element = buff[i] - '0';
      if (element != 1 && element != 2 && element != 3) {
        fprintf(stderr, "Error: Puzzle entries can only be 1 or 2 or 3.\n");
        assert(false);
      }
    }
    sp++;
    offset += kp + 1;
  } while (str[offset] != '\0');

  setSize(sp, kp);

  offset = 0;
  int r = 0;

  do {
    bytes_read = sscanf(str + offset, "%s\n", buff);
    for (unsigned int c = 0; c < k; c++)
      set(r, c, buff[c] - '0');
    r++;
    offset += k + 1;
  } while (str[offset] != '\0');

  assert(s == r);
}

void Puz::doCopy(const Puz& other) {
  assert(data == NULL);
  data = NULL;

  // cout << "other s in copy: " << other.s << endl;
  // cout << "other k in copy: " << other.k << endl;
  // data = (e_type *) malloc(sizeof(e_type) * s * entries_per_row);
  setSize(other.s, other.k);
  // cout << "yo im here" <<endl;
  // if (s == 0){
  // cout << "entries per row of other: " << other.entries_per_row << endl;
  // cout << "im printing the false puzzle" << endl;
  // cout << other.toString("|") << endl;
  // cout << "print done" << endl;
  // }
  probability = other.probability;
  fitness = other.fitness;
  memcpy(data, other.data, sizeof(e_type) * s * entries_per_row);

  // memcpy(data, other.data, sizeof(data));
  // cout << "yo im done" <<endl;
  wildcards = other.wildcards;
}

// ==================================================================
//
//    Constructors / Destructors
//
// ==================================================================

/**
 * Constructs a 1,1 puzzle with 1 as entry.
 */
Puz::Puz() {
  data = NULL;
  setSize(1, 1);
}

/**
 * Constructor to create a puzzle. When created, all cells are of value 1.
 * @param s1: number of rows in the puzzle
 *        k1: number of columns in the puzzle
 */
Puz::Puz(unsigned int s, unsigned int k) {
  data = NULL;
  assert(s != 0 && k != 0);
  setSize(s, k);
}

/**
 * Constructor to create a Puz from the output of serialize().
 * Doesn't do full error checking.
 */
Puz::Puz(void* buf) {
  e_type* e_buf = (e_type*) buf;
  data = NULL;
  unsigned int s_buf = e_buf[0];
  unsigned int k_buf = e_buf[1];
  setSize(s_buf, k_buf);
  memcpy(data, e_buf + 2, s * entries_per_row * sizeof(e_type));
}

/**
 * Constructor to create a Puzzle from a file or a string description.
 * @param str: The file's name or string of puzzle.
 */
Puz::Puz(const string& str) {
  data = NULL;

  bool is_file = true;
  for (int i = 0; i < str.length() && is_file; i++) {
    if (str[i] == '\n')
      is_file = false;
  }
  if (is_file)
    readFromFile(str.c_str());
  else
    readFromString(str.c_str());
}

/**
 * Constructor to create a Puzzle from a 2D vector of ints.
 * @param m: 2D vector of {1,2,3}'s.
 */
Puz::Puz(const vector<vector<int> >& m) {
  data = NULL;
  assert(m.size() > 0);
  setSize(m.size(), m[0].size());
  for (int r = 0; r < s; r++) {
    assert(m[r].size() == m[0].size());
    for (int c = 0; c < k; c++) {
      set(r, c, m[r][c]);
    }
  }
}

/**
 * Constructor to create a Puzzle from two Puzzles by unioning the
 * rows sets together.  Fills empty space with 1's.
 * @param p1, p2: Two puzzles.
 * XXX - Not tested
 */
Puz::Puz(const Puz& p1, const Puz& p2) {
  data = NULL;
  setSize(p1.s + p2.s, (p1.k < p2.k ? p2.k : p1.k));

  for (int r = 0; r < p1.s; r++)
    for (int c = 0; c < p1.k; c++)
      set(r, c, p1.get(r, c));

  for (int r = 0; r < p2.s; r++)
    for (int c = 0; c < p2.k; c++)
      set(r + p1.s, c, p2.get(r, c));
}

/**
 * Copy constructor to create a puzzle.
 * @param other: puzzle to copy.
 */
Puz::Puz(const Puz& other) {
  // cout << "In const copy" << endl;
  // cout << "other s: " << other.s << endl;
  // cout << "other k: " << other.k << endl;
  doCopy(other);
  // cout << "In const copy, this = " << this <<  ", data = " << data
  //      << ", other.data = " << other.data << endl;
}

Puz::Puz(Puz& other) {
  // cout << "In copy" << endl;
  // cout << "other 2 s: " << other.s << endl;
  // cout << "other 2 k: " << other.k << endl;
  doCopy(other);
  // cout << "In copy, this = " << this <<  ", data = " << data
  //      << ", other.data = " << other.data << endl;
  // cout << "s = " << s << ", k = " << k << endl;
}

Puz& Puz::operator=(Puz const& other) {
  // cout << "In =copy" << endl;
  if (data != NULL) {
    delete[] data;
    data = NULL;
  }
  doCopy(other);
  // cout << "In =copy, this = " << this <<  ", data = " << data
  //      << ", other.data = " << other.data << endl;
  return *this;
}

Puz::Puz(Puz&& other) noexcept {
  data = other.data;
  other.data = NULL;
  s = other.s;
  k = other.k;
  entries_per_row = other.entries_per_row;
  wildcards = std::move(other.wildcards);
  probability = other.probability;
  fitness = other.fitness;
  isSimplied = other.isSimplied;
  tdm = other.tdm;
  other.tdm = NULL;
  results = other.results;
}

Puz& Puz::operator=(Puz&& other) noexcept {
  if (this != &other) {
    delete[] data;
    data = other.data;
    other.data = NULL;
    s = other.s;
    k = other.k;
    entries_per_row = other.entries_per_row;
    wildcards = std::move(other.wildcards);
    probability = other.probability;
    fitness = other.fitness;
    isSimplied = other.isSimplied;
    tdm = other.tdm;
    other.tdm = NULL;
    results = other.results;
  }
  return *this;
}

/**
 * Evaluates to the Cartesian product of the two puzzles with
 * s1*s2 rows and k1+k2 columns.
 */
Puz Puz::operator*(const Puz& other) const {
  unsigned int s1 = s;
  unsigned int s2 = other.s;
  unsigned int k1 = k;
  unsigned int k2 = other.k;

  Puz p(s1 * s2, k1 + k2);

  for (int r1 = 0; r1 < s1; r1++)
    for (int r2 = 0; r2 < s2; r2++) {
      for (int c1 = 0; c1 < k1; c1++)
        p.set(r1 * s2 + r2, c1, get(r1, c1));
      for (int c2 = 0; c2 < k2; c2++)
        p.set(r1 * s2 + r2, k1 + c2, other.get(r2, c2));
    }

  return p;
}

/**
 * Destructor.
 */
Puz::~Puz() {
  // cout << "In ~Puz(), this = " << this << ", data = " << data << endl;
  delete[] data;
}

// ==================================================================
//
//    Public Functions
//
// ==================================================================

#define TRITS_PER_BYTE 5

size_t Puz::getSize() const {
  return (2 + s * entries_per_row) * sizeof(e_type);
}

void Puz::serializeInPlace(void* buffer) const {
  auto* buf = reinterpret_cast<e_type*>(buffer);
  buf[0] = s;
  buf[1] = k;
  memcpy(buf + 2, data, s * entries_per_row * sizeof(e_type));
}

void* Puz::serialize(size_t& bytes) const {
  bytes = getSize();
  auto* buf = std::malloc(sizeof(e_type) * (2 + s * entries_per_row));
  serializeInPlace(buf);
  return buf;
}

/**
 * Set from other Puz of same size.
 */
void Puz::setRow(unsigned int row, const Puz& other, unsigned int other_row) {
  assert(s > row && other.k == k && other.s > other_row);
  FORALL_C(set(row, c, other.get(other_row, c));)
}

/**
 * Replaces the current puzzle with a canonized version.
 */
void Puz::canonize() {
  canonize_puzzle(*this);
}

/**
 * Returns the minor of a puzzle by deleting rows no in to_keep.
 */
Puz Puz::getMinor(const vector<bool>* rows_to_keep, const vector<bool>* cols_to_keep) const {
  if (rows_to_keep == NULL && cols_to_keep == NULL)
    return *this;

  // cout << "Before: " << *this;

  int sp = s;
  if (rows_to_keep != NULL) {
    assert(rows_to_keep->size() == s);
    sp = 0;
    for (int r = 0; r < s; r++)
      sp += (rows_to_keep->at(r) ? 1 : 0);
  }
  int kp = k;
  if (cols_to_keep != NULL) {
    assert(cols_to_keep->size() == k);
    kp = 0;
    for (int c = 0; c < k; c++)
      kp += (cols_to_keep->at(c) ? 1 : 0);
  }

  assert(sp >= 1 && kp >= 1);

  // cout << sp << ", " << kp << endl;

  Puz ret(sp, kp);

  int row_correction = 0;
  for (int r = 0; r < s; r++) {
    if (rows_to_keep != NULL && !(rows_to_keep->at(r))) {
      row_correction++;
      continue;
    }

    int col_correction = 0;
    for (int c = 0; c < k; c++) {
      if (cols_to_keep != NULL && !(cols_to_keep->at(c))) {
        col_correction++;
        continue;
      }

      assert(r - row_correction >= 0 && c - col_correction >= 0);
      // cout << "s = " << s << ", k = " << k << ", r = " << r << ", c = " << c << endl;
      ret.set(r - row_correction, c - col_correction, get(r, c));
    }
  }
  // cout << "AFTER" << endl;
  // cout << ret.getHeight() << ", " << ret.getWidth() << endl;

  // cout << "After: " << endl;
  // cout << ret << endl;

  return ret;
}

Puz Puz::getMinor(const vector<int>& rows_to_keep) const {
  Puz ret(rows_to_keep.size(), k);

  // cout << "ret.s = " << ret.s << ", ret.k = " << ret.k << ", k = " << k << endl;

  for (int r = 0; r < ret.s; r++)
    for (int c = 0; c < k; c++)
      ret.set(r, c, get(rows_to_keep[r], c));
  return ret;
}

Puz Puz::getMinorRow(int row) const {
  vector<bool> rows_to_keep(s, true);
  rows_to_keep[row] = false;
  return getMinor(&rows_to_keep, NULL);
}

Puz Puz::getMinorCol(int col) const {
  vector<bool> cols_to_keep(k, true);
  cols_to_keep[col] = false;
  return getMinor(NULL, &cols_to_keep);
}

/**
 * Extend or constract the puzzle
 * @param rows: number of rows to add
 * @return Result of resizing the puzzle
 */
Puz Puz::resizePuz(int rows) const {
  if (rows == 0)
    return *this;

  assert(s + rows >= 1);

  Puz ret(s + rows, k);
  if (rows > 0)
    memcpy(ret.data, data, sizeof(e_type) * s * entries_per_row);
  else
    memcpy(ret.data, data, sizeof(e_type) * (s + rows) * entries_per_row);

  return ret;
}

/**
 * Set this puzzle to a random puzzle of the same size
 */
void Puz::setRandom(double prob) {
  uniform_int_distribution<int> dist(1, 3);
  uniform_real_distribution<double> dist2(0, 1);
  FORALL_RC(if (dist2(global_prg) <= prob) {
    int val = dist(global_prg);
    set(r, c, val);
  })
}

/**
 * Sorts the puzzle using radix sort.  Takes O(s*k) time.
 */
void Puz::sort() {
  int perm[s];
  FORALL_R(perm[r] = r;);

  // Use counting sort as the base sort.
  for (int c = k - 1; c >= 0; c--) {
    int counts[3] = {0, 0, 0};
    int perm2[s];

    FORALL_R(counts[get(perm[r], c) - 1]++;);

    counts[1] += counts[0];
    counts[2] += counts[1];

    for (int r = s - 1; r >= 0; r--) {
      int val = get(perm[r], c);
      counts[val - 1]--;
      perm2[counts[val - 1]] = perm[r];
    }

    memcpy(perm, perm2, s * sizeof(int));
  }

  Puz ret(s, k);
  FORALL_RC(ret.set(r, c, get(perm[r], c)););

  *this = ret;
}

/**
 * Functions to set puzzle or row to all 1's and to replace with next
 * row or puzzle in lex order.  The next functions return true iff the
 * puzzle or row is not the last one (all 3's), and the changes it
 * back to the first.  Useful for iterating over puzzles or rows.
 */
void Puz::init() {
  FORALL_RC(set(r, c, 1););
}

void Puz::initRow(unsigned int r) {
  assert(r < s);
  FORALL_C(set(r, c, 1););
}

bool Puz::next() {
  unsigned int r = 0;

  while (r < s && !nextRow(r))
    r++;

  return !(r == s);
}

bool Puz::nextRow(unsigned int r) {
  assert(r < s);

  int c = 0;

  while (c < k && get(r, c) == 3) {
    set(r, c, 1);
    c++;
  }

  if (c == k) {
    return false;
  } else {
    set(r, c, get(r, c) + 1);
    return true;
  }
}

/*
 * Returns true iff a length-k row that permutations map to u1, u2,
 * u3, respectively, satisfies the inner condition of strong USPs.
 * It is false if this length-k row mapping does not witness that
 * the puzzle is a strong USP.  Runtime is O(k).
 */
bool Puz::isWitnessInner(e_type u1, e_type u2, e_type u3, int n, bool strong) {
  for (int i = 0; i < n; i++) {
    int count = ((u1 & 3) == 0) + ((u2 & 3) == 1) + ((u3 & 3) == 2);
    // & 3 is the same as % 4, using 2 bits to represent each entry
    u1 = u1 >> 2;
    u2 = u2 >> 2;
    u3 = u3 >> 2;
    if ((strong && count == 2) || (!strong && count >= 2))
      return true;
  }
  return false;
}

/**
 * Returns true iff a length-k row that permutations map to u_1, u_2,
 * u_3, respectively, satisfies the inner condition of strong USPs.
 * It is false if this length-k row this mapping does not witness that
 * the puzzle is a strong USP.  Runtime is O(k).
 */
/*
  Explaination:
  For each rows, we use bitmask to encode the row. We use 2 bits to encode each entry.
  We use 00 to represent 1, 01 to represent 2, 10 to represent 3.
  We use 32 bits to represent a row. If the number of entries is larger than 32,
  we use multiple integers to represent the row.
*/
bool Puz::isWitness(int r1, int r2, int r3, bool strong) const {
  bool res = false;
  int i1 = r1 * entries_per_row;
  int i2 = r2 * entries_per_row;
  int i3 = r3 * entries_per_row;
  for (int i = 0; i < entries_per_row && !res; i++) {
    res = res || isWitnessInner(data[i1 + i], data[i2 + i], data[i3 + i],
                                (i < entries_per_row - 1
                                     // if i is less than entries_per_row - 1, then we take 32
                                     ? ELTS_PER_ENTRY
                                     : ((k & MOD_MASK) ? (k & MOD_MASK) : ELTS_PER_ENTRY)),
                                // if k is not a multiple of 32, then we take the remainder
                                // otherwise, we take 32
                                strong);
  }
  return res;
}

/**
 * Checks the puzzle against several possible properties.
 *
 * mode = CHECK_FULL: checks against full [CKSU05] definition.
 *      = CHECK_OBVIOUS: checks against 2D matching simplification.
 *      = CHECK_LOCAL: checks whether 3DM is just diagonal.
 *
 * strong = true: checks for strong USP condition.
 *        = false: checks for USP condition.
 */
bool Puz::checkPuz(check_mode_t mode, bool strong) const {
  return check(*this, mode, strong);
}

/**
 * Overidding operator == to compare two puzzle to see if they are equal
 * @param puzz: the puzzle to compare
 *        puzz2: the other puzzle to compare
 * @return True if they are equal, false otherwise
 */
bool operator==(const Puz& puz, const Puz& puz2) {
  if (puz.s != puz2.s || puz.k != puz2.k)
    return false;
  for (int r = 0; r < puz.s; r++)
    for (int c = 0; c < puz.k; c++)
      if (puz.get(r, c) != puz2.get(r, c))
        return false;
  return true;
}

/**
 * Returns a string representation of the puzzle.
 */
string Puz::toString(string separator) const {
  string ret = "";
  for (int r = 0; r < s; r++) {
    for (int c = 0; c < k; c++) {
      ret += to_string(get(r, c));
    }
    ret += separator;
  }
  return ret;
}
