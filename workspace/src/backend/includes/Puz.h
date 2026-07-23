/**
 * @file Puz.h
 * @brief Puzzle representation for Strong Uniquely Solvable Puzzles (SUSPs)
 *
 * This file contains the Puz class, which represents an (s,k)-puzzle - a 2D array
 * with s rows and k columns where each cell contains a value from {1, 2, 3}.
 *
 * @author Jordan An, Jerry Ji, Emma Diep, Hung Duong, Matt Anderson
 */

#pragma once

#include <vector>
#include "constants.h"
#include "checker.h"
#include "3DM_to_SAT.h"

using namespace std;

/**
 * @defgroup MacroHelpers Iteration Macros
 * @brief Helper macros for iterating over puzzle cells
 * @{
 */

/** @brief Iterate over all rows and columns */
#define FORALL_RC(__x)            \
  for (int r = 0; r < s; r++)     \
    for (int c = 0; c < k; c++) { \
      __x                         \
    }

/** @brief Iterate over all rows */
#define FORALL_R(__x)           \
  for (int r = 0; r < s; r++) { \
    __x                         \
  }

/** @brief Iterate over all columns */
#define FORALL_C(__x)           \
  for (int c = 0; c < k; c++) { \
    __x                         \
  }

/** @} */  // end of MacroHelpers

/**
 * @defgroup BitmaskConstants Bitmask Encoding Constants
 * @brief Constants for efficient puzzle storage using bitmasks
 *
 * Puzzles are stored using 2 bits per cell in 64-bit integers.
 * This allows storing 32 entries per integer, reducing memory footprint by 16x.
 * @{
 */

using e_type = unsigned long long;  ///< Type for storing encoded puzzle rows
#define ELTS_PER_ENTRY 32           ///< Number of puzzle entries per 64-bit integer
#define DIV_SHIFT 5                 ///< Bit shift for division by 32
#define BITS_PER_ENTRY 64           ///< Total bits in e_type
#define BITS_RIGHT_SHIFT 60         ///< Shift for accessing high bits
#define MOD_MASK 0x1F               ///< Mask for modulo 32 operation

/** @} */  // end of BitmaskConstants

/**
 * @class Puz
 * @brief Represents an (s,k)-puzzle for matrix multiplication research
 *
 * A Puz object represents a 2D array with s rows and k columns, where each cell
 * contains a value from {1, 2, 3}. These puzzles are used to discover fast
 * matrix multiplication algorithms via the Strong Uniquely Solvable Puzzle (SUSP)
 * property.
 *
 * **Memory Efficiency**: Uses bitmask encoding (2 bits per cell) for compact storage.
 * Each row is stored in ceil(k/32) 64-bit integers.
 *
 * **Thread Safety**: Not thread-safe. Use separate instances per thread.
 *
 * Example usage:
 * @code
 * // Create a 14x6 puzzle
 * Puz p(14, 6);
 * p.set(0, 0, 2);
 * p.set(0, 1, 3);
 *
 * // Verify if it's a Strong USP
 * if (p.checkPuz(CHECK_FULL, true)) {
 *     cout << "Found a Strong USP!" << endl;
 * }
 *
 * // Load from file
 * Puz loaded("puzzles/example.puz");
 * @endcode
 */
class Puz {
 private:
  unsigned int s = 1;
  unsigned int k = 1;
  e_type* data = NULL;
  unsigned int entries_per_row = 1;
  vector<tuple<int, int> > wildcards;
  double probability = 0.0;
  double fitness = -1.0;
  bool isSimplied = false;
  bool* tdm = NULL;
  int results = 0;
  /**
   * Set s, k, data, and entries_per_row according to s, k.  Sets to
   * all 1's.
   */
  void setSize(unsigned int s, unsigned int k);

  /**
   * Readed an extended puzzle from file and returns the wildcard coords.
   */
  void readFromFile(const char* filename);

  /**
   * Readed an puzzle from string.
   */
  void readFromString(const char* str);
  /**
   * Returns true iff a length-k row that permutations map to u1, u2,
   * u3, respectively, satisfies the inner condition of strong USPs.
   * It is false if this length-k row mapping does not witness that
   * the puzzle is a strong USP.  Runtime is O(k).
   */
  bool static isWitnessInner(e_type u1, e_type u2, e_type u3, int n, bool strong);

  void doCopy(const Puz& other);

 public:
  /**
   * Constructs a 1,1 puzzle with 1 as entry.
   */
  Puz();

  /**
   * Constructor to create a puzzle. When created, all cells are of value 1.
   * @param s1: number of rows in the puzzle
   *        k1: number of columns in the puzzle
   */
  Puz(unsigned int s, unsigned int k);

  /**
   * Constructor to create a Puzzle from a file or a string description.
   * @param str: The file's name or string of puzzle.
   */
  Puz(const string& str);
  Puz(char* str) : Puz(string(str)){};

  /**
   * Constructor to create a Puzzle from a 2D vector of ints.
   * @param m: 2D vector of {1,2,3}'s.
   */
  Puz(const vector<vector<int> >& m);

  /**
   * Constructor to create a Puzzle from two Puzzles by unioning the
   * rows sets together.  Fills empty space with 1's.
   * @param p1, p2: Two puzzles.
   * XXX - Not tested
   */
  Puz(const Puz& p1, const Puz& p2);

  /**
   * Constructor to create a Puz from the output of serialize().
   * Doesn't do full error checking.
   */
  Puz(void* buf);

  /**
   * Copy constructor to create a puzzle.
   * @param other: puzzle to copy.
   */
  Puz(const Puz& other);
  Puz(Puz& other);
  Puz& operator=(Puz const& other);
  Puz(Puz&& other) noexcept;
  Puz& operator=(Puz&& other) noexcept;

  /**
   * Destructor.
   */
  ~Puz();

  /**
   * Accessors
   */

  /**
   * Returns the size of this Puz object in bytes
   */
  size_t getSize() const;

  /**
   * Returns a serialized version of a Puz.  Returns buffer that must
   * be deallocated by caller.
   */
  void* serialize(size_t& bytes) const;

  /*
   *  Serializes the puzzle into the given buffer. You must insure
   *  yourself that the buffer has sufficient capacity.
   * */
  void serializeInPlace(void*) const;

  /**
   * Serializes the puzzle to ostream
   */
  void serialize(ostream& os) const {
    os.write(reinterpret_cast<const char*>(&s), sizeof(s));
    os.write(reinterpret_cast<const char*>(&k), sizeof(k));
    os.write(reinterpret_cast<const char*>(data), s * entries_per_row * sizeof(e_type));
  }

  /**
   * @brief Deserialize puzzle from input stream
   *
   * Reads puzzle data from an input stream in binary format.
   *
   * @param is Input stream to read from
   *
   * @note This replaces the current puzzle data
   * @see serialize(ostream&)
   */
  void deserialize(istream& is) {
    is.read(reinterpret_cast<char*>(&s), sizeof(s));
    is.read(reinterpret_cast<char*>(&k), sizeof(k));
    entries_per_row = (unsigned int) (k + (ELTS_PER_ENTRY - 1)) / ELTS_PER_ENTRY;
    if (data != NULL)
      delete[] data;
    data = new e_type[s * entries_per_row];
    is.read(reinterpret_cast<char*>(data), s * entries_per_row * sizeof(e_type));
  }

  /**
   * @brief Get raw data at index (for advanced use)
   * @param idx Index into the internal data array
   * @return Raw encoded data at the specified index
   * @note This is a low-level function. Use get(r,c) for normal access.
   */
  inline e_type getData(int idx) const {
    return data[idx];
  }

  /**
   * @brief Set the value of a cell in the puzzle
   *
   * Sets the cell at position (r,c) to the specified value.
   * Uses bitmask encoding for efficient storage.
   *
   * @param r Row index (0-indexed, must be < s)
   * @param c Column index (0-indexed, must be < k)
   * @param val Value to set (must be 1, 2, or 3)
   *
   * @pre r < s && c < k
   * @pre 1 <= val <= 3
   *
   * Example:
   * @code
   * Puz p(10, 5);
   * p.set(0, 0, 2);  // Set top-left cell to 2
   * p.set(0, 1, 3);  // Set cell (0,1) to 3
   * @endcode
   */

  inline void set(unsigned int r, unsigned int c, unsigned int val) {
    assert(r < s && c < k);
    assert(1 <= val && val <= 3);
    unsigned int i = r * entries_per_row + (c >> DIV_SHIFT);
    unsigned int offset = c & MOD_MASK;
    e_type mask = 0x3L << (offset << 1);
    e_type e_val = val;
    data[i] = (~mask & data[i]) | ((e_val - 1) << (offset << 1));
    // cout << "set: "
    // 	 << "r = " << r << ", c = " << c
    // 	 << ", i = " << i << ", offset = " << offset
    // 	 << ", mask = " << mask
    // 	 << ", new = " << ((val - 1) << (offset << 1))
    //      << ", data[i] = " << data[i]
    // 	 << endl;
  }

  /**
   * @brief Check if puzzle has been simplified
   * @return true if puzzle has been marked as simplified
   */
  bool getSimplified() const {
    return isSimplied;
  }

  /**
   * @brief Mark puzzle as simplified or not
   * @param simplified Simplified status to set
   */
  void setSimplified(bool simplified) {
    isSimplied = simplified;
  }

  /**
   * @brief Resize internal TDM storage
   * @note Allocates space for s³ boolean values
   */
  void resizeTDM() {
    if (tdm != NULL)
      delete[] tdm;
    tdm = new bool[s * s * s];
  }

  /**
   * @brief Clear TDM storage
   */
  void clearTDM() {
    if (tdm != NULL)
      delete[] tdm;
  }

  /**
   * @brief Set a TDM entry
   * @param r1 First row index
   * @param r2 Second row index
   * @param r3 Third row index
   * @param val Value to set
   */
  void setTDM(int r1, int r2, int r3, bool val) {
    tdm[r1 * s * s + r2 * s + r3] = val;
  }

  /**
   * @brief Get a TDM entry
   * @param r1 First row index
   * @param r2 Second row index
   * @param r3 Third row index
   * @return TDM value at (r1, r2, r3)
   */
  bool getTDM(int r1, int r2, int r3) const {
    return tdm[r1 * s * s + r2 * s + r3];
  }

  /**
   * @brief Set verification results
   * @param res Results value to store
   */
  void setResults(int res) {
    results = res;
  }

  /**
   * @brief Get verification results
   * @return Stored results value
   */
  int getResults() const {
    return results;
  }

  /**
   * @brief Get the value of a cell in the puzzle
   *
   * Retrieves the value at position (r,c) using bitmask decoding.
   *
   * @param r Row index (0-indexed, must be < s)
   * @param c Column index (0-indexed, must be < k)
   * @return Value at (r,c), will be 1, 2, or 3
   *
   * @pre r < s && c < k
   *
   * @note This is an inline function for performance
   *
   * Example:
   * @code
   * Puz p(10, 5);
   * p.set(0, 0, 2);
   * int val = p.get(0, 0);  // Returns 2
   * @endcode
   */
  inline unsigned int get(unsigned int r, unsigned int c) const {
    assert(r < s && c < k);
    unsigned int i = r * entries_per_row + (c >> DIV_SHIFT);
    unsigned int offset = c & MOD_MASK;
    // cout << "get: "
    // 	 << "r = " << r << ", c = " << c
    // 	 << ", i = " << i << ", offset = " << offset
    // 	 << ", data[i] = " << data[i]
    // 	 << endl;
    return ((data[i] >> (offset << 1)) & 0x3) + 1;
  }

  inline void setData(int idx, e_type val) {
    data[idx] = val;
  }

  inline e_type getData(int idx) {
    return data[idx];
  }

  /**
   * @brief Get the number of rows in the puzzle
   * @return Number of rows (s)
   */
  inline int getHeight() const {
    return s;
  }

  /**
   * @brief Get the number of columns in the puzzle
   * @return Number of columns (k)
   */
  inline int getWidth() const {
    return k;
  }

  /**
   * @brief Get wildcard positions (for extended puzzles)
   * @return Vector of (row, col) tuples indicating wildcard positions
   */
  vector<tuple<int, int> > getWildcards() const {
    return wildcards;
  }

  /**
   * @brief Copy a row from another puzzle
   *
   * Copies row other_row from puzzle 'other' into row 'row' of this puzzle.
   *
   * @param row Destination row index in this puzzle
   * @param other Source puzzle
   * @param other_row Source row index in other puzzle
   *
   * @pre Both puzzles must have the same width (k)
   */
  void setRow(unsigned int row, const Puz& other, unsigned int other_row);

  /**
   * @brief Convert puzzle to canonical form using Nauty
   *
   * Replaces the current puzzle with its canonical (lexicographically smallest)
   * isomorphic form. This is useful for detecting duplicate puzzles that are
   * equivalent under row/column permutations.
   *
   * @note Uses the Nauty library for graph canonization
   * @note Modifies the puzzle in-place
   *
   * Example:
   * @code
   * Puz p(5, 3);
   * // ... set puzzle values ...
   * p.canonize();  // Convert to canonical form
   * // Now p is in its canonical representation
   * @endcode
   */
  void canonize();

  /**
   * @brief Extract a subpuzzle (minor)
   *
   * Creates a new puzzle by selecting specific rows and columns.
   *
   * @param rows_to_keep Boolean vector indicating which rows to keep
   * @param cols_to_keep Boolean vector indicating which columns to keep
   * @return New puzzle with selected rows and columns
   */
  Puz getMinor(const vector<bool>* rows_to_keep, const vector<bool>* cols_to_keep) const;

  /**
   * @brief Extract a subpuzzle with specified rows
   * @param rows_to_keep Vector of row indices to keep
   * @return New puzzle with selected rows
   */
  Puz getMinor(const vector<int>& rows_to_keep) const;

  /**
   * @brief Get puzzle with one row removed
   * @param row Row index to remove
   * @return New puzzle without the specified row
   */
  Puz getMinorRow(int row) const;

  /**
   * @brief Get puzzle with one column removed
   * @param col Column index to remove
   * @return New puzzle without the specified column
   */
  Puz getMinorCol(int col) const;

  /**
   * @brief Resize puzzle by adding or removing rows
   *
   * Creates a new puzzle with the specified number of rows added or removed.
   * New rows are initialized to all 1's.
   *
   * @param rows Number of rows to add (positive) or remove (negative)
   * @return New resized puzzle
   *
   * Example:
   * @code
   * Puz p(10, 5);
   * Puz larger = p.resizePuz(3);   // Now 13x5
   * Puz smaller = p.resizePuz(-2); // Now 8x5
   * @endcode
   */
  Puz resizePuz(int rows) const;

  /**
   * @brief Randomize puzzle values
   *
   * Sets each cell to a random value from {1, 2, 3} with given probability.
   *
   * @param prob Probability of changing each cell (default 1.0 = change all)
   *
   * Example:
   * @code
   * Puz p(10, 5);
   * p.setRandom(0.5);  // 50% chance each cell changes from default 1
   * @endcode
   */
  void setRandom(double prob = 1.0);

  /**
   * @brief Sort puzzle rows lexicographically
   *
   * Sorts the rows of the puzzle in lexicographic order using radix sort.
   * This is useful for normalizing puzzles and detecting duplicates.
   *
   * @note Time complexity: O(s*k)
   * @note Uses counting sort as the base sort (stable)
   * @note Modifies the puzzle in-place
   *
   * Example:
   * @code
   * Puz p(5, 3);
   * // ... set puzzle values ...
   * p.sort();  // Rows now in lexicographic order
   * @endcode
   */
  void sort();

  /**
   * @defgroup Iteration Puzzle Iteration Methods
   * @brief Methods for iterating through all possible puzzles/rows
   *
   * These methods allow systematic enumeration of all (s,k)-puzzles or
   * all possible rows. Useful for exhaustive search on small puzzle spaces.
   * @{
   */

  /**
   * @brief Initialize all cells to 1
   *
   * Sets every cell in the puzzle to 1 (the first puzzle in lex order).
   */
  void init();

  /**
   * @brief Initialize a specific row to all 1's
   * @param r Row index to initialize
   */
  void initRow(unsigned int r);

  /**
   * @brief Advance to next puzzle in lexicographic order
   *
   * Increments the puzzle to the next one in lexicographic order.
   * When the last puzzle (all 3's) is reached, wraps to first (all 1's).
   *
   * @return true if advanced to next puzzle, false if wrapped to first
   *
   * Example:
   * @code
   * Puz p(2, 2);  // Small puzzle for iteration
   * p.init();
   * do {
   *     // Process puzzle
   *     cout << p.toString() << endl;
   * } while (p.next());
   * @endcode
   */
  bool next();

  /**
   * @brief Advance a specific row to next in lexicographic order
   * @param r Row index to advance
   * @return true if advanced, false if row wrapped to all 1's
   */
  bool nextRow(unsigned int r);

  /** @} */  // end of Iteration group

  /**
   * Returns true iff a length-k row that permutations map to u_1, u_2,
   * u_3, respectively, satisfies the inner condition of strong USPs.
   * It is false if this length-k row this mapping does not witness that
   * the puzzle is a strong USP.  Runtime is O(k).
   */
  bool isWitness(int r1, int r2, int r3, bool strong = true) const;

  /**
   * @brief Verify if puzzle is a (Strong) Uniquely Solvable Puzzle
   *
   * Checks whether the puzzle satisfies the USP or Strong USP property
   * using various modified verification methods.
   *
   * @param mode Verification mode:
   *   - CHECK_FULL: Complete verification using 3D matching (slowest, strongest property)
   *   - CHECK_OBVIOUS: Check for simplifiable/obvious SUSPs (faster, weaker property)
   *   - CHECK_LOCAL: Check local property only (fastest, weakest property)
   * @param strong If true, check Strong USP; if false, check regular USP
   *
   * @return true if puzzle satisfies the property, false otherwise
   *
   * @note For large puzzles (s ≥ 20), this may use GPU acceleration
   * @note This repository uses the older term "obvious" USP, but this is the same as "simplifiable"
   * USP in our paper.
   *
   * Example:
   * @code
   * Puz p(14, 6);
   * // ... set puzzle values ...
   *
   * // Quick check
   * if (p.checkPuz(CHECK_OBVIOUS, true)) {
   *     // Likely a Strong USP, do full verification
   *     if (p.checkPuz(CHECK_FULL, true)) {
   *         cout << "Verified Strong USP!" << endl;
   *     }
   * }
   * @endcode
   */
  bool checkPuz(check_mode_t mode = CHECK_FULL, bool strong = true) const;

  /**
   * Check to see if the Puz is a strong USP or not
   */
  bool checkStrongUSP() const {
    return checkPuz(CHECK_FULL, true);
  }

  /**
   * Outputs SAT instances in DIMACS format but does not check.
   */
  void checkStrongUSPToDimacs(const string filename) const {
    check_SAT_to_Dimacs(*this, filename);
  }

  /**
   * Check to see if the Puz is an obvious strong USP or not
   */
  bool checkObviousStrongUSP() const {
    return checkPuz(CHECK_OBVIOUS, true);
  }

  /**
   * @brief Cartesian product of two puzzles
   *
   * Creates a new puzzle that is the Cartesian product of this puzzle and another.
   * The result has s1*s2 rows and k1+k2 columns.
   *
   * @param other The other puzzle
   * @return Cartesian product puzzle
   *
   * @note Result dimensions: (s1*s2) × (k1+k2)
   */
  Puz operator*(const Puz& other) const;

  /**
   * Overidding operator == to compare two puzzle to see if they are equal
   * @param puzz: the puzzle to compare
   *        puzz2: the other puzzle to compare
   * @return True if they are equal, false otherwise
   */
  friend bool operator==(const Puz& puz, const Puz& puz2);

  /**
   * Overidding operator != to compare two puzzle to see if they are equal
   * @param puzz: the puzzle to compare
   *        puzz2: the other puzzle to compare
   * @return False if they are equal, true otherwise
   */
  friend bool operator!=(const Puz& puz, const Puz& puz2) {
    return !(puz == puz2);
  }

  /**
   * @brief Convert puzzle to string representation
   *
   * Creates a string representation of the puzzle with customizable separator.
   *
   * @param separator String to use between rows (default: newline)
   * @return String representation of the puzzle
   *
   * Example:
   * @code
   * Puz p(3, 2);
   * cout << p.toString();      // Rows separated by newlines
   * cout << p.toString("|");   // Rows separated by |
   * @endcode
   */
  string toString(string separator = "\n") const;

  void setProbability(double prob) {
    probability = prob;
  }

  /**
   * Set the fitness of this puzzle
   */
  void setFitness(double fit) {
    fitness = fit;
  }

  /**
   * Get the probability of being picked of this puzzle
   */
  double getProbability() const {
    return probability;
  }

  /**
   * Get the fitness of this puzzle
   */
  double getFitness() const {
    // cout << fitness << endl;
    assert(fitness >= 0);
    return fitness;
  }

  /**
   * << operator for streaming class.  Use get(r,c) function to
   * determine results.
   */
  template <class charT, class traits>
  friend basic_ostream<charT, traits>& operator<<(basic_ostream<charT, traits>& os, const Puz& p);
};

/**
 * << operator for streaming class.  Use get(r,c) function to
 * determine results.
 */
template <class charT, class traits>
basic_ostream<charT, traits>& operator<<(basic_ostream<charT, traits>& os, const Puz& p) {
  return os << ("Size: " + to_string(p.s) + " x " + to_string(p.k) + "\n" + p.toString());
}
