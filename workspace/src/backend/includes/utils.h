#pragma once

#include <random>

// Thread-safe random generator in [0,1)
inline double safe_rand() {
  double d = (double) (random() % 1000000) / 1000000.0;
  return d;
}
