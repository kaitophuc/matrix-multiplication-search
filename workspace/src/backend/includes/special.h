/*
 * Author: Matt Anderson
 */

#pragma once

#include "Puz.h"
#include <stdlib.h>

using namespace std;

Puz minor_puzzle(const Puz& p, int c, int r1, int r2);

bool is_special_col(const Puz& p, int c, int r1, int r2);

bool is_special_col(const Puz& p, int c);

int count_special(const Puz& p);

unsigned int hierarchically_special(const Puz& p, int r1, int r2, int c, int required = 1);

unsigned int hierarchically_special(const Puz& p, int required = 1);

unsigned int hierarchically_special_unordered(const Puz& p, int required = 1,
                                              vector<Puz>* ps = NULL);
