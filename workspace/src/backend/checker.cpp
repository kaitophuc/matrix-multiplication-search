/*
 * Module implementing several non-trivial methods for checking
 * whether a given puzzle is a strong USP.
 *
 * 1. check_usp_uni() - Undirectional search for a witness that puzzle is not a strong USP.
 * 2. check_usp_bi() - Bidirectional search for a witness that puzzle is not a strong USP.
 * 3. check_SAT_MIP() - Thread parallel search using SAT and MIP solvers.
 * 4. check() - A combinations of the above checks and various heuristics.
 * 5. A mechanism for caching (that isn't turned on and may be broken).
 *
 * Uses it's own representation of sets.
 *
 * Author: Matt & Jerry.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <syscall.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#include "checker.h"
#include "Puz.h"
#include "3DM_to_SAT.h"
#ifdef __GUROBI_INSTALLED__
#include "3DM_to_MIP.h"
#endif
#include "heuristic.h"
#include "canonization.h"
#include "omp.h"

using namespace std;

/*
 *=============================================================
 *
 *  Helper Functions
 *
 *=============================================================
 */

/*
 * Checks whether the given vector pi represents the identity
 * permutation of length s.  Runtime is O(s).
 */
bool ident(vector<int> pi, int s) {
  for (int i = 0; i < s; i++) {
    if (pi[i] != i)
      return false;
  }
  return true;
}

/*
 *=============================================================
 *
 *  Checking for Strong USPs
 *
 *=============================================================
 *
 * A puzzle U is a set of length-k strings over {1,2,3}.  We say that
 * U is a strong uniquely solvable puzzle (USP) if for all pi_1, pi_2,
 * pi_3 in Symmetric(U), either pi_1 = pi_2 = pi_3 or there exists u
 * in U and i in [k] such that exactly two of (pi_1(u))_i = 1,
 * (pi_2(u))_i = 2, and (pi_3(u))_i = 3 hold.
 *
 * While checking this condition One can assume without loss of
 * generality that pi_1 is the identity permutation, because only the
 * relative difference in the permutations is relevant.  In
 * particular, if the inner condition holds (or doesn't) for pi_1,
 * pi_2, pi_3, u, and i, it also holds for 1, pi_1^-1 o pi_2, pi_1^-1 o
 * pi_3, pi_1^-1(u), and i.
 */

/*
 *-------------------------------------------------------------
 *  Unidirectional Test
 *-------------------------------------------------------------
 *
 * Determines whether the given s-by-k puzzle U is a strong USP.  Uses
 * a unidirectional algorithm that tests all permutations pi_2 and
 * pi_3 to see whether they witness that U is not a strong USP.
 *
 * This is an alternative implementation of check_usp from usp.c.  It
 * uses c++ iterators with the built-in function next_permutation to
 * loop over all permutations.
 */
check_t check_usp_uni_full(const Puz& p) {
  TDM tdm(p, true, TDM::NO_GPU);
  return check_usp_uni(&tdm);
}

check_t check_usp_uni(TDM* tdm) {
  int s = tdm->size();

  // The TDM precomputes s * s * s memoization table storing the
  // mapping of rows that witness that a partial mapping is consistent
  // with U being a strong USP.  For U to not be a strong USP there
  // must a way to select s false entries from the table whose indexes
  // are row, column, and slice disjoint.

  // Create two vectors for storing permutations over the set of s
  // rows.  pi_1 is not explicitly represented, because it will always
  // be the identity.
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
      if (ident(pi_2, s) && ident(pi_3, s))
        continue;

      // Check whether pi_1, pi_2, pi_3 touch only trues.
      bool found = false;
      for (int i = 0; i < s; i++) {
        if (!tdm->get(i, pi_2[i], pi_3[i])) {
          found = true;
          break;
        }
      }

      if (!found)
        // pi_2 and pi_3 are a witness that U is not a strong USP.
        return NO_CHECK;
    }
  }

  return YES_CHECK;
}

/*
 *=============================================================
 *
 *  Bidirectional Test
 *
 *=============================================================
 *
 * Idea: Perform a bidirectional search for a non-strong USP witness
 * in the puzzle U that starts from both the first row and the last
 * row of U.  It builds up partial functions for pi_2 and pi_3 and
 * stores the rows which have already been mapped to for each.  This
 * allows the subproblems to be specified by sets rather than by
 * permutations.  We use the fast implementation of sets as 64-bit
 * long below to encode the sets.
 *
 */

/*
 * Function that displays the contents of a pair of sets to the console.
 */
void print_sets(set_long sets) {
  printf("s2 = { ");
  for (int i = 0; i < 31; i++) {
    if (MEMBER_S2(sets, i))
      printf("%d ", i);
  }
  printf("}, s3 = { ");
  for (int i = 0; i < 31; i++) {
    if (MEMBER_S3(sets, i))
      printf("%d ", i);
  }
  printf("}\n");
}

// Function signatures -- so it will compile.
void find_witness_forward(int w, map<set_long, bool>* memo, int i1, set_long sets, TDM* tdm,
                          bool is_id);
bool find_witness_reverse(int w, map<set_long, bool>* memo, int i1, set_long sets, TDM* tdm,
                          bool is_id, map<set_long, bool>* opposite);

/*
 * Determines whether the given s-by-k puzzle U is a strong USP.  Uses
 * a bidirectional algorithm.
 */
check_t check_usp_bi_full(const Puz& p) {
  TDM tdm(p, true, TDM::NO_GPU);
  return check_usp_bi(&tdm);
}

check_t check_usp_bi(TDM* tdm) {
  // Precompute s * s * s memoization table storing the mapping of
  // rows that witness that a partial mapping is consistent with U
  // being a strong USP.  For U to not be a strong USP there must a
  // way to select s false entries from the table whose indexes are
  // row, column, and slice disjoint.  This is the same first step as
  // in the unidirectional verison.

  // Create two maps that will store partial witnesses of U not being
  // a strong USP.
  //
  // The membership of a set pair p = (S2, S3) in forward_memo with t
  // = |S2| = |S3| means that there exists a 1-1 map from the first t
  // rows of U to S2 and S3 such that hits only false entries of
  // row_witness.  The membership of a set pair in reverse_memo is
  // similar, but for the _last_ t rows of U.
  //
  // Note that the value in forward_memo is not used, but the value in
  // reverse_memo indicates whether a witness has been found for that
  // subproblem.
  int s = tdm->size();

  assert(s <= 31);

  map<set_long, bool> forward_memo;
  map<set_long, bool> reverse_memo;

  // Perform the first half of the search by filling in forward_memo
  // for the first s/2 rows of U.
  find_witness_forward(s, &forward_memo, 0, SET_ID(0L), tdm, true);

  // Perform the second half of the search by filling in reverse_memo
  // for the second s/2 rows of U.  These partial mappings are
  // complemented and then looked up in forward_memo to check whether
  // they can be combined into a complete witness for U not being a
  // strong USP.

  bool res = !find_witness_reverse(s, &reverse_memo, s - 1, SET_ID(0L), tdm, true, &forward_memo);

  return (res ? YES_CHECK : NO_CHECK);
}

/*
 * A memoized recursive function contructs partial witnesses of
 * non-strong USP for a s-by-k puzzle with given row_witness, from row
 * i1 to s/2 with sets storing the indexes already used by pi_2 and
 * pi_3.  is_id indicates whether the partial assignment to this point
 * is identity for all three permutations.
 */
void find_witness_forward(int s, map<set_long, bool>* memo, int i1, set_long sets, TDM* tdm,
                          bool is_id) {
  // Update the identity flag in sets, if the partial map is no longer
  // consistent with identity on all permutations.
  if (!is_id)
    sets = UNSET_ID(sets);

  // Base Case 1: Look to see whether we're already computed and
  // stored this answer in the memo table.  If so, return.
  map<set_long, bool>::const_iterator iter = memo->find(sets);
  if (iter != memo->end()) {
    return;
  }

  // Base Case 2: We have processed enough rows, insert the final sets
  // of size floor(s/2) into the memo table and return.  Note that
  // value false stored in the memo table doesn't mean anything.
  if (i1 == (int) (floorf(s / 2.0))) {
    memo->insert(pair<set_long, bool>(sets, false));
    return;
  }

  // Recursive Case: There is more work to do.  Loop over all pairs
  // i2, i3 that are not already present in sets and check whether
  // tdm[i1][i2][i3] is true.  If so, recurse on the subproblem that
  // results from inserting i2 and i3 into sets, incrementing i1, and
  // updating the identity flag.
  for (int i2 = 0; i2 < s; i2++) {
    if (MEMBER_S2(sets, i2))
      continue;
    for (int i3 = 0; i3 < s; i3++) {
      if (MEMBER_S3(sets, i3))
        continue;
      if ((tdm->get(i1, i2, i3)) == false)
        continue;

      set_long sets2 = INSERT_S3(INSERT_S2(sets, i2), i3);

      find_witness_forward(s, memo, i1 + 1, sets2, tdm, is_id && i1 == i2 && i2 == i3);
    }
  }

  // Insert the subproblem we just completed into the memo table, so
  // the computation will not be repeated.
  memo->insert(pair<set_long, bool>(sets, false));
  return;
}

/*
 * A memoized recursive function contructs partial witnesses of
 * non-strong USP for a s-by-k puzzle with given row_witness, from row
 * i1 to s/2 with sets storing the indexes already used by pi_2 and
 * pi_3.  is_id indicates whether the partial assignment to this point
 * is identity for all three permutations.  Takes in a map containing
 * the memo table opposite from the forward search.  Return true iff
 * an witness that U is not a strong USP has been found.
 */
bool find_witness_reverse(int s, map<set_long, bool>* memo, int i1, set_long sets, TDM* tdm,
                          bool is_id, map<set_long, bool>* opposite) {
  // Update the identity flag in sets, if the partial map is no longer
  // consistent with identity on all permutations.
  if (!is_id)
    sets = UNSET_ID(sets);

  // Base Case 1: Look to see whether we're already computed and
  // stored this answer in the memo table.  If so, return that answer.
  map<set_long, bool>::const_iterator iter = memo->find(sets);
  if (iter != memo->end()) {
    return iter->second;
  }

  // Base Case 2: We have processed enough rows, look for a complement
  // in the opposite table to form a complete witness.  If sets was
  // produced by identities, then the complement cannot be an identity
  // and still be a witness, because strong USP ignores the case that
  // all permutations are the same.
  if (i1 < (int) (floorf((s - 1) / 2.0))) {
    set_long sets_comp = COMPLEMENT(sets, s);
    bool found_pair = false;

    if (is_id) {
      // If sets is identity, it can only pair with non-identities.
      sets_comp = UNSET_ID(sets_comp);
    } else {
      // If sets is not identity, it pair with either identities or non-identites.
      iter = opposite->find(sets_comp);
      if (iter != opposite->end()) {
        found_pair = true;
      }
      sets_comp = SET_ID(sets_comp);
    }

    iter = opposite->find(sets_comp);
    if (iter != opposite->end()) {
      found_pair = true;
    }

    // Insert result in memo table.  If we didn't find a witness
    // looking at this subproblem, make a note of this in the memo
    // table so the computation is not repeated.  If we did find a
    // witness the computation will immediately return from each
    // recursive call in this case (and it wasn't really necessary to
    // insert it).
    memo->insert(pair<set_long, bool>(sets, found_pair));

    return found_pair;
  }

  // Recursive Case: There is more work to do.  Loop over all pairs
  // i2, i3 that are not already present in sets and check whether
  // tdm[i1][i2][i3] is true.  If so, recurse on the
  // subproblem that results from inserting i2 and i3 into sets,
  // incrementing i1, and updating the identity flag.
  for (int i2 = 0; i2 < s; i2++) {
    if (MEMBER_S2(sets, i2))
      continue;
    for (int i3 = 0; i3 < s; i3++) {
      if (MEMBER_S3(sets, i3))
        continue;
      if ((tdm->get(i1, i2, i3)) == false)
        continue;

      set_long sets2 = INSERT_S3(INSERT_S2(sets, i2), i3);

      if (find_witness_reverse(s, memo, i1 - 1, sets2, tdm, is_id && i1 == i2 && i2 == i3,
                               opposite)) {
        // We've found a witness that U is not a strong USP.  Note
        // that we don't actually need to insert it in the memo table
        // as the computation will immediately return from each
        // recursive call in this case.
        memo->insert(pair<set_long, bool>(sets, true));
        return true;
      }
    }
  }

  // We didn't find a witness looking at this subproblem, make a note
  // of this in the memo table so the computation is not repeated.
  memo->insert(pair<set_long, bool>(sets, false));
  return false;
}

#ifdef __GUROBI_INSTALLED__

/*
 * Determines whether the given s-by-k puzzle U is a strong USP.
 * Checks using SAT and MIP solvers in parallel threads.
 */
check_t check_SAT_MIP_full(const Puz& p) {
  TDM tdm(p, true);
  return check_SAT_MIP(&tdm);
}

check_t check_SAT_MIP(TDM* tdm) {
  check_t res = MAYBE_CHECK;
  bool interrupt = false;

  // Need to allow nesting because check_SAT using omp.
  omp_set_nested(1);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 2; i++) {
    if (i == 0) {
      check_t res_local = check_SAT(tdm, &interrupt);

#pragma omp critical(SAT_MIP_interrupt)
      if (!interrupt) {
        res = res_local;
        interrupt = true;
#pragma omp flush(interrupt)
      }
    } else {
      check_t res_local = check_MIP(tdm, &interrupt);

#pragma omp critical(SAT_MIP_interrupt)
      if (!interrupt && res_local != MAYBE_CHECK) {
        res = res_local;
        interrupt = true;
#pragma omp flush(interrupt)
      }
    }
  }

  assert(res != MAYBE_CHECK);

  return res;
}

#endif

/*
 * Determines whether the given s-by-k puzzle U is a strong USP.
 * Tries to pick the most efficient method.  Uses the bidirectional
 * search if s is large enough, and the unidirectional search
 * otherwise.
 */
check_t check_full(const Puz& p) {
  return check(p, CHECK_FULL, true, NULL);
}

check_t check(const Puz& p, check_mode_t mode, bool strong, TDM* tdm_ptr) {
  int s = p.getHeight();
  int k = p.getWidth();
  check_t res = MAYBE_CHECK;

  TDM* tdm = tdm_ptr;

  if (s <= 1)
    return YES_CHECK;

  if (s <= 2) {
    return check2(p, 0, 1, mode, strong);
  }

  res = heuristic_usp_pieces(p);
  if (res != MAYBE_CHECK)
    return res;

  if (mode != CHECK_FULL) {
    if (tdm_ptr == NULL) {
      tdm = new TDM(p, strong, TDM::ALLOW_GPU);
    }
    if (mode == CHECK_OBVIOUS)
      tdm->simplify(TDM::LAZY);
    if (tdm->count() == s)
      res = YES_CHECK;
    else
      res = NO_CHECK;
    if (tdm_ptr == NULL)
      delete tdm;
    return res;
  }

  if (s <= 7) {
    // Local create if not passed in.
    if (tdm_ptr == NULL)
      tdm = new TDM(p, strong, TDM::NO_GPU);
    res = check_usp_bi(tdm);
    if (tdm_ptr == NULL)
      delete tdm;
    return res;
  }

  res = heuristic_row_triples(p, mode, strong);
  if (res != MAYBE_CHECK) {
    return res;
  }

  if (tdm_ptr == NULL)
    tdm = new TDM(p, strong, TDM::ALLOW_GPU);

  tdm->simplify(TDM::EAGER);

  // res = heuristic_random(tdm);
  // if (res != MAYBE_CHECK) {
  //   if (tdm_ptr == NULL) delete tdm;
  //   return res;
  // }

  // res = heuristic_greedy(tdm);
  // if (res != MAYBE_CHECK) {
  //   if (tdm_ptr == NULL) delete tdm;
  //   return res;
  // }

#ifdef __GUROBI_INSTALLED__
  res = check_SAT_MIP(tdm);
#else
  res = check_SAT(tdm);
#endif

  if (tdm_ptr == NULL)
    delete tdm;
  assert(res != MAYBE_CHECK);

  return res;
}

/*
 * A specialized function that determines whether the given 2-by-k
 * puzzle U is a strong USP.
 */
check_t check2(const Puz& p, unsigned int r1, unsigned int r2, check_mode_t mode, bool strong) {
  if (mode != CHECK_LOCAL) {
    bool has_bad_matching =
        ((!p.isWitness(r1, r1, r2, strong) && !p.isWitness(r2, r2, r1, strong)) ||
         (!p.isWitness(r1, r2, r1, strong) && !p.isWitness(r2, r1, r2, strong)) ||
         (!p.isWitness(r2, r1, r1, strong) && !p.isWitness(r1, r2, r2, strong)));
    return has_bad_matching ? NO_CHECK : YES_CHECK;
  } else {
    unsigned int rows[2] = {r1, r2};
    int count = 0;
    for (int a = 0; a < 2; a++) {
      for (int b = 0; b < 2; b++) {
        for (int c = 0; c < 2; c++) {
          count += p.isWitness(rows[a], rows[b], rows[c], strong) ? 0 : 1;
        }
      }
    }
    return count == 2 ? YES_CHECK : NO_CHECK;
  }
}

/*
 * A specialized function that determines whether the given 3-by-k
 * puzzle U is a strong USP.
 */
check_t check3(const Puz& p, unsigned int r1, unsigned int r2, unsigned int r3, check_mode_t mode,
               bool strong) {
  vector<int> rows;
  rows.push_back(r1);
  rows.push_back(r2);
  rows.push_back(r3);
  Puz p2 = p.getMinor(rows);
  check_t res = check(p2, mode, strong);
  return res;
}

/*
 * A specialized function that determines whether the given 4-by-k
 * puzzle U is a strong USP.
 */
check_t check4(const Puz& p, unsigned int r1, unsigned int r2, unsigned int r3, unsigned int r4,
               check_mode_t mode, bool strong) {
  vector<int> rows;
  rows.push_back(r1);
  rows.push_back(r2);
  rows.push_back(r3);
  rows.push_back(r4);
  Puz p2 = p.getMinor(rows);
  check_t res = check(p2, mode, strong);
  return res;
}
