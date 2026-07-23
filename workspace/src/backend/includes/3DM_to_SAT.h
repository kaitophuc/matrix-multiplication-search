/* Reduction from 3D-matching to 3SAT
   The goal is to use the result from check_usp_rows to deside whether
   witness is allowed to occur on that specific coordinate on a 3D cube.
   Then use the result to reduce this 3D-perfect matching problem to a 3SAT
   problem and print out the reduction in 3cnf-form (dimacs) for MapleSAT solver
   to give a final answer.

   Author: Jerry & Matt.
*/

#pragma once

#include <stdlib.h>
#include "Puz.h"
#include "checker.h"
#include "TDM.h"

// Checks whether p is a strong USP, using a SAT solver.  Returns
// IS_USP if p is a strong USP, NOT_USP if p is not a strong USP and
// UNDET_USP if it was unable to determine whether p is a strong USP.
// UNDET_USP will only be returned if there was an error.  A
// synchronized pthread version of check_MIP may be asynchronously
// interrupted by setting *interrupt_ptr = true.  Return value only
// meaningful if not interrupted.

check_t check_SAT_full(const Puz& p);
check_t check_SAT(TDM* tdm, bool* interrupt_ptr = NULL);
void check_SAT_to_Dimacs(const Puz& p, const string filename);
