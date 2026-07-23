#pragma once

#include <chrono>
#include <iostream>
#include "elt_type.h"
#include "TDM.h"

enum ils_mode_t { ILS_INACTIVE, ILS_SEARCH, ILS_FEED };

typedef struct _stats_t {
  long long evals = 0;
  long long repeated = 0;
  long long proc = 0;
  long long dropped = 0;
  long long since_progress = 0;
  long long restarts = 0;
  double best_gap = 1;
  long long max_found = 0;
  int working = 0;
  ils_mode_t mode = ILS_INACTIVE;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
  int searched_types[NUM_ELT_TYPE];

  void serialize(std::ostream& os) const;
  void deserialize(std::istream& is);
} stats_t;
