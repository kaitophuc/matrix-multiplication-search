/*
 * Module implementing puzzle canonization using Nauty.
 *
 * Author: Matt.
 */

#pragma once

#include "Puz.h"
#include "bloom_filtering.h"

#define MAX_ISOMORPHS -1

// Returns true iff no isomorphs of p have been previously seen.
bool have_seen_isomorph(const Puz& p, bool remember = true, int* index = NULL);

// Reset the set of previously seen isomorphs.
void reset_isomorphs();

// Returns the number of previously seen isomorphs.
size_t get_num_isomorphs();

// Replace p with its canonical isomorph.
void canonize_puzzle(Puz& p);

// Returns true iff p1 and p2 are isomorphs.
bool are_isomorphs(const Puz& p1, const Puz& p2);

// Swaps the set of seen isomorphs and the set of stored isomorphs
void swap_stored_state();

// Save/restore isomorph cache state to/from external maps
void save_isomorph_state(bool& ext_initilized, std::map<std::string, int>& ext_seen_isomorphs,
                         int& ext_num_seen, std::map<std::string, int>& ext_stored_isomorphs,
                         int& ext_stored_seen);
void restore_isomorph_state(const bool& ext_initilized,
                            const std::map<std::string, int>& ext_seen_isomorphs,
                            const int& ext_num_seen,
                            const std::map<std::string, int>& ext_stored_isomorphs,
                            const int& ext_stored_seen);
