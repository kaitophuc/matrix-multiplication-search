#include "prg.h"

std::random_device _rd;
std::mt19937 global_prg(_rd());
