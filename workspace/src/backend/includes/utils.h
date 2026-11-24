#pragma once

#include <random>

// Use the process seed once per worker, then generate priority jitter locally.
inline double safe_rand() {
  static thread_local std::mt19937 generator(static_cast<unsigned int>(random()));
  return static_cast<double>((generator() >> 1) % 1000000) / 1000000.0;
}
