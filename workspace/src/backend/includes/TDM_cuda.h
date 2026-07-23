#pragma once

#include "Puz.h"

int simplify_cuda_inner(const Puz& p, bool* tdm = NULL, bool strong = true);
