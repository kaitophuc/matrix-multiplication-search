/*
 * Module implementing several non-trivial heursitics for checking
 * whether a given puzzle is a strong USP.
 *
 * Heuristics MUST match the checker_t type and semantics, and should
 * be named with a prefix "heuristic_".
 *
 * Author: Matt.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <math.h>
#include <assert.h>
#include <semaphore.h>
#include "pthread.h"
#include "Puz.h"
#include "checker.h"
using namespace std;

/*
 * A specialized function that determines whether any pair of rows
 * prevent an s-by-k from being a strong USP.  Returns NOT_USP or
 * UNDET_USP.
 */
check_t heuristic_row_pairs(const Puz& p, check_mode_t mode, bool strong);
check_t heuristic_row_pairs_full(const Puz& p);

/*
 * A specialized function that determines whether any triple of rows
 * prevent an s-by-k from being a strong USP.  Returns NOT_USP or
 * UNDET_USP.
 */
check_t heuristic_row_triples(const Puz& p, check_mode_t mode, bool strong);
check_t heuristic_row_triples_full(const Puz& p);

/*
 * A heuristic for SUSP, that checks whether the puzzle can be a USP.
 * By checking whether the pieces of any type are duplicated.  Returns
 * NOT_USP iff it fails to be a USP, and UNDET_USP otherwise.
 *
 * Note: this is not a sufficient condition for the puzzle to be a
 * USP, because 112/223/331 ~ 113/221/332 is not a USP, but has the
 * same pieces.
 */
check_t heuristic_usp_pieces(const Puz& p);

/*
 * Heuristically precheck puzzle via random and greedy approaches
 * Returns IS_USP if puzzle is a strong USP.  Returns NOT_USP if
 * puzzle is not a strong USP.  Returns UNDET_USP if the function has
 * not determined the puzzle is a strong USP.
 */
check_t heuristic_random(TDM* tdm);
check_t heuristic_random_full(const Puz& p);

/*
 * Returns NOT_USP iff it finds a witness that puzzle is not a strong
 * USP.  UNDET_USP indicates the search was inconclusive.  Attempts to
 * generate a witness by greedily selecting a layer from among those
 * with fewest remaining edges and then uniformly selects an edge from
 * that layer.  This repeats until a witness is found, or no progress
 * can be made.  This process repeats for some specified number of
 * iterations.  There is no benefit to reorder_witnesses() be called
 * before this.  Requires s <= 31.
 */
check_t heuristic_greedy(TDM* tdm);
check_t heuristic_greedy_full(const Puz& p);
