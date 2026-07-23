#ifndef __3DM_to_MIP_H__
#define __3DM_to_MIP_H__

/*
  This module checks whether a puzzle is a strong USP by reducing the
  3DM matching instance implicit in the puzzle to a mixed integer
  program and then solve the mixed integer program using Gurobi.  This
  module requires that Gurobi have been installed on the machine which
  this source is compiled on.

  Author: Anthony Yang Xu & Matt.
  Summer 2017.
*/

#include "TDM.h"

// Checks whether p is a strong USP, using a MIP reduction.  Returns
// IS_USP if p is a strong USP, NOT_USP if p is not a strong USP and
// UNDET_USP if it was unable to determine whether p is a strong USP.
// UNDET_USP will only be returned if there was an error.
// A synchronized pthread version of check_MIP that takes it's
// parameters using an argument struct.  May be asynchronously
// interrupted by setting *interrupt_ptr = true.  Return value only
// meaningful if not interrupted.
check_t check_MIP(TDM* tdm, bool* interrupt_ptr = NULL);
check_t check_MIP_full(const Puz& p);

// Deallocates internal static datastructures.
void finalize_check_MIP();

#endif
