/*
 * Module implementing several non-trivial heursitics for checking
 * whether a given puzzle is a strong USP.
 *
 * Heuristics MUST match the checker_t type and semantics, and should
 * be named with a prefix "heuristic_".
 *
 * Author: Matt.
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

#include "heuristic.h"
#include "Puz.h"
#include "3DM_to_SAT.h"
#include "3DM_to_MIP.h"
#include "checker.h"
#include "set.h"

/*
 * A specialized function that determines whether any pair of rows
 * prevent an s-by-k from being a strong USP.  Returns NO_CHECK or
 * MAYBE_CHECK.
 */
check_t heuristic_row_pairs(const Puz& p, check_mode_t mode, bool strong) {
  int s = p.getHeight();
  for (unsigned int r1 = 0; r1 < s - 1; r1++) {
    for (unsigned int r2 = r1 + 1; r2 < s; r2++) {
      check_t res = check2(p, r1, r2, mode, strong);
      if (res == NO_CHECK)
        return NO_CHECK;
    }
  }
  return MAYBE_CHECK;
}

check_t heuristic_row_pairs_full(const Puz& p) {
  return heuristic_row_pairs(p, CHECK_FULL, true);
}

/*
 * A specialized function that determines whether any triple of rows
 * prevent an s-by-k from being a strong USP.  Return NO_CHECK or
 * MAYBE_CHECK.
 */
check_t heuristic_row_triples(const Puz& p, check_mode_t mode, bool strong) {
  int s = p.getHeight();
  for (int r1 = 0; r1 < s - 2; r1++) {
    for (int r2 = r1 + 1; r2 < s - 1; r2++) {
      for (int r3 = r2 + 1; r3 < s; r3++) {
        check_t res = check3(p, r1, r2, r3, mode, strong);
        if (res == NO_CHECK)
          return NO_CHECK;
      }
    }
  }
  return MAYBE_CHECK;
}

check_t heuristic_row_triples_full(const Puz& p) {
  return heuristic_row_triples(p, CHECK_FULL, true);
}

/**
 * A heuristic for SUSP, that checks whether the puzzle can be a USP.
 * By checking whether the pieces of any type are duplicated.  Returns
 * NO_CHECK iff it fails to be a USP, and MAYBE_CHECK otherwise.
 *
 * Note: this is not a sufficient condition for the puzzle to be a
 * USP, because 112/223/331 ~ 113/221/332 is not a USP, but has the
 * same pieces.
 */
check_t heuristic_usp_pieces(const Puz& p) {
  if (p.getHeight() <= 1)
    return YES_CHECK;

  map<set, bool> pieces[3];

  int s = p.getHeight();
  int k = p.getWidth();

  for (int r = 0; r < s; r++) {
    set piece[3];
    for (int val = 1; val <= 3; val++)
      piece[val - 1] = create_empty_set();

    for (int c = 0; c < k; c++) {
      int e = p.get(r, c);
      piece[e - 1] = set_union(piece[e - 1], create_one_element_set(c));
    }

    for (int val = 1; val <= 3; val++) {
      if (pieces[val - 1].find(piece[val - 1]) != pieces[val - 1].end()) {
        return NO_CHECK;
      } else {
        pieces[val - 1][piece[val - 1]] = true;
      }
    }
  }

  return MAYBE_CHECK;
}

// XXX - This is commented out because it would corrupt the new TDM
// /*
//  *  Reorders rows in an attempt to make fewer entries be generated.
//  *  They can be arranged in increasing or decreasing order, or outside
//  *  -> inside, or inside -> outside.
//  *  XXX - Make it also permute the puzzle?
//  */
// void reorder_witnesses(puzzle * p, bool increasing, bool sorted){

//   int s = p -> s;
//   bool * tdm = p -> tdm;

//   // Count the witnesses in each layer.
//   int layer_counts[s];
//   for (int i = 0; i < s; i++){
//     layer_counts[i] = 0;
//     for (int j = 0; j < s; j++){
//       for (int r = 0; r < s; r++){
// 	if (get_tdm_entry(p, i, j, r))
// 	  layer_counts[i]++;
//       }
//     }
//     //printf("layer_count[%d] = %d\n", i, layer_counts[i]);
//   }

//   // Controls whether order is increasing or decreasing.
//   int increase = (increasing ? 1 : -1);

//   // Compute a permutate that orders layers as desired.
//   int perm[s];
//   bool tdm2[s * s * s];
//   for (int i = 0; i < s; i++){
//     int opt_index = -1;
//     int opt = increase * (s*s + 1);

//     for (int j = 0; j < s; j++){
//       if (increase * layer_counts[j] < increase * opt) {
// 	opt = layer_counts[j];
// 	opt_index = j;
//       }
//     }

//     int to_index = i;

//     if (sorted) {
//       // Layers will appear in a sorted order.
//       to_index = i;
//     } else {
//       // Layers will appear order from outside to inside.
//       if (i % 2 == 0)
// 	to_index = (int)(i / 2.0);
//       else
// 	to_index = s - (int) ((i + 1) / 2.0);
//     }

//     //printf("to_index = %d, from_index = %d\n", to_index, minIndex);

//     layer_counts[opt_index] = increase * (s * s + 2);
//     perm[opt_index] = to_index;

//   }

//   // Reorder the layers in a new matrix.
//   for (int i = 0; i < s; i++){
//     for (int j = 0; j < s; j++){
//       for (int r = 0; r < s; r++){
// 	int ix = perm[i] * s * s + perm[j] * s + perm[r];
// 	tdm2[ix] = get_tdm_entry(p, i, j, r);
//       }
//     }
//   }

//   /*
//   // Check that layers are arranged.
//   for (int i = 0; i < s; i++){
//     layer_counts[i] = 0;
//     for (int j = 0; j < s; j++){
//       for (int r = 0; r < s; r++){
// 	if (tdm2[i * s * s + j * s + r])
// 	  layer_counts[i]++;
//       }
//       printf("layer_count[%d] = %d\n", i, layer_counts[i]);
//     }
//   }
//   */

//   // Copy back to original matrix.
//   memcpy(tdm, tdm2, s * s * s * sizeof(bool));

// }

/*
 * Returns NO_CHECK iff it finds a witness that puzzle is not a strong USP.
 * MAYBE_CHECK indicates the search was inconclusive.  Requires s <= 31.
 */
check_t heuristic_random_witness(TDM* tdm, int repeats) {
  int s = tdm->size();

  if (s > 31)
    return MAYBE_CHECK;

  bool failed = false;
  int best = 0;
  int total = 0;
  for (int n = 0; n < repeats && !failed; n++) {
    set_long curr = CREATE_EMPTY();
    bool is_ident = true;
    for (int i = 0; i < s && !failed; i++) {
      int offset_j = lrand48() % s;
      int offset_k = lrand48() % s;
      bool found = false;

      for (int j = 0; j < s && !found; j++) {
        int shift_j = (j + offset_j) % s;
        if (MEMBER_S2(curr, shift_j))
          continue;
        for (int k = 0; k < s && !found; k++) {
          int shift_k = (k + offset_k) % s;
          if (MEMBER_S3(curr, shift_k))
            continue;

          if (tdm->get(i, shift_j, shift_k)) {
            if (shift_k != shift_j || shift_k != i)
              is_ident = false;
            curr = INSERT_S2(INSERT_S3(curr, shift_k), shift_j);
            found = true;
          }
        }
      }

      if (!found) {
        best = (best < i ? i : best);
        total += i;
        failed = true;
        // printf("Failed to locate witness in random. Acheived: %d / %d, id: %d\n", i, s,is_ident);
      }
    }
    if (!failed && !is_ident)
      return NO_CHECK;

    // Not sure if this is a good heuristic stopping condition, it is
    // to prevent a lot of time being waste on small puzzle sizes.  It
    // makes longer puzzles slower.  XXX - Improve parameters.
    if ((log(n + 1) / log(2)) > best)
      return MAYBE_CHECK;
    failed = false;
  }

  return MAYBE_CHECK;
}

int default_heuristic_iteration(int s) {
  return s * s * s;
}

/*
 * Returns NO_CHECK iff it finds a witness that puzzle is not a strong
 * USP.  MAYBE_CHECK indicates the search was inconclusive.  Attempts to
 * generate a witness by greedily selecting a layer from among those
 * with fewest remaining edges and then uniformly selects an edge from
 * that layer.  This repeats until a witness is found, or no progress
 * can be made.  This process repeats for some specified number of
 * iterations.  There is no benefit to reorder_witnesses() be called
 * before this.  Requires s <= 31.
 */
check_t heuristic_greedy(TDM* tdm, int repeats) {
  int s = tdm->size();

  if (s > 31)
    return MAYBE_CHECK;

  bool failed = false;
  int best_depth = 0;
  int total = 0;
  for (int n = 0; n < repeats && !failed; n++) {
    set_long curr_set = CREATE_EMPTY();

    bool is_ident = true;

    int layer_counts[s];

    for (int i = 0; i < s; i++) {
      layer_counts[i] = 0;
      for (int j = 0; j < s; j++) {
        for (int k = 0; k < s; k++) {
          if (tdm->get(i, j, k))
            layer_counts[i]++;
        }
      }
      // printf("layer_counts[%d] = %d\n",i, layer_counts[i]);
    }
    int too_big = s * s + 1;

    for (int i = 0; i < s; i++) {
      // Randomly select a layer with the least choices.
      int best = too_big;
      int num_best = 0;
      int layer = 0;
      for (layer = 0; layer < s; layer++) {
        if (layer_counts[layer] < best) {
          best = layer_counts[layer];
          num_best = 1;
        } else if (layer_counts[layer] == best) {
          num_best++;
        }
      }

      if (best == 0) {
        // There is an empty layer.
        failed = true;
        best_depth = (best_depth < i ? i : best_depth);
        total += i;
        break;
      }

      assert(best < too_big);
      assert(num_best >= 1);
      int choice = lrand48() % num_best;
      int found = -1;
      for (layer = 0; layer < s && found < choice; layer++) {
        if (layer_counts[layer] == best)
          found++;
      }
      // Correct for final increment.
      layer--;  // This the layer to select something in.

      // Randomly select a choice from this layer uniformly at random.
      found = -1;
      choice = lrand48() % layer_counts[layer];
      // printf("choice = %d\n", choice);

      int j = 0;
      int k = 0;
      for (j = 0; j < s && found < choice; j++) {
        if (MEMBER_S2(curr_set, j))
          continue;
        for (k = 0; k < s && found < choice; k++) {
          if (MEMBER_S3(curr_set, k))
            continue;

          if (tdm->get(layer, j, k)) {
            found++;
            // printf("found!\n");
            if (found == choice) {
              layer_counts[layer] = too_big;  // Means layer is already processed.
              curr_set = INSERT_S2(curr_set, j);
              curr_set = INSERT_S3(curr_set, k);
              if (layer != j || layer != k)
                is_ident = false;
            }
          }
        }
      }
      // Correct for final increment.
      j--;
      k--;

      // printf("(j, k) = (%d,%d)\n", j, k);

      // printf("i = %d, layer = %d  %d\n", i, layer, layer_counts[layer]);
      assert(layer_counts[layer] == too_big);
      // Update layer_counts.
      for (int layer2 = 0; layer2 < s; layer2++) {
        if (layer_counts[layer2] != too_big) {
          for (int j2 = 0; j2 < s; j2++) {
            if (!MEMBER_S2(curr_set, j2) && tdm->get(layer2, j2, k))
              layer_counts[layer2]--;
          }
          for (int k2 = 0; k2 < s; k2++) {
            if (!MEMBER_S3(curr_set, k2) && tdm->get(layer2, j, k2))
              layer_counts[layer2]--;
          }
          if (tdm->get(layer2, j, k))
            layer_counts[layer2]--;
          assert(layer_counts[layer2] >= 0);
        }
        // printf("layer_counts[%d] = %d\n",layer2, layer_counts[layer2]);
      }
      // printf("\n");
    }

    if (!failed && !is_ident)
      return NO_CHECK;

    // Not sure if this is a good heuristic stopping condition, it is
    // to prevent a lot of time being waste on small puzzle sizes.  It
    // makes longer puzzles slower.  XXX - Improve parameters.
    // if ((log(n + 1) / log(2)) > best_depth)
    //  return false;
    failed = false;
  }

  return MAYBE_CHECK;
}

check_t heuristic_greedy(TDM* tdm) {
  if (tdm->size() > 31)
    return MAYBE_CHECK;

  return heuristic_greedy(tdm, default_heuristic_iteration(tdm->size()));
}

check_t heuristic_greedy_full(const Puz& p) {
  TDM tdm(p, true, TDM::NO_GPU);
  return heuristic_greedy(&tdm);
}

// Some variables measuring performance.
int size_forward = 0;
int last_layer_forward = 0;
int size_backward = 0;
int checks_backward = 0;

/*
 * Heuristically precheck puzzle via random and greedy approaches
 * Returns IS_USP if puzzle is a strong USP.  Returns NO_CHECK if
 * puzzle is not a strong USP.  Returns MAYBE_CHECK if the function has
 * not determined the puzzle is a strong USP.
 */
check_t heuristic_random(TDM* tdm, int iter) {
  // Rearrange row_witness in the hope it makes the search faster.
  // Then randomly attempt to build a witness that U is not a strong
  // USP.  The number of iterations is a bit ad hoc; s*s*s also seeme
  // to work well in the domain of parameters I profiled.  XXX -
  // Improve parameters.  Doing it twice for two different ordering of
  // the puzzle was the most effective.

  // This reorder is aimed to make the randomize search more likely to
  // succeed.

  check_t res = MAYBE_CHECK;

  // XXX - Random reordering is commented out because it corrupts the
  // tdm.
  //  reorder_witnesses(p, true, true);

  if (heuristic_random_witness(tdm, iter) == NO_CHECK) {
    res = NO_CHECK;
  }
  // else {

  //   reorder_witnesses(p, false, false);
  //   if (heuristic_random_witness(p, iter) == NO_CHECK){
  //     res = NO_CHECK;
  //   } else {

  //     // This reorder is aimed to make the forward and backward search
  //     // balanced.
  //     reorder_witnesses(p, true, false);
  //     if (heuristic_random_witness(p, iter) == NO_CHECK){
  // 	res = NO_CHECK;
  //     }
  //   }
  // }

  return res;
}

check_t heuristic_random(TDM* tdm) {
  if (tdm->size() > 31)
    return MAYBE_CHECK;

  return heuristic_random(tdm, default_heuristic_iteration(tdm->size()));
}

check_t heuristic_random_full(const Puz& p) {
  TDM tdm(p, true, TDM::NO_GPU);
  return heuristic_random(&tdm);
}
