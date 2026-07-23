#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "Puz.h"
#include "fitness.h"
#include "ils_stage.h"
#include "checker.h"

#define NUM_STAGES 4

struct Search {
  /* Search parameters */
  check_mode_t mode;
  bool strong;
  int s_target;
  int s;
  int k;
  Puz p;

  /* Isomorph cache state */
  bool initilized;
  std::map<std::string, int> seen_isomorphs;
  int num_seen;
  std::map<std::string, int> stored_isomorphs;
  int stored_seen;

  /* Search algorithm state */
  FitTester* ft;
  int num_stages;
  int curr;
  std::vector<std::shared_ptr<ILSStage>> stages;

  /* TDM global count */
  unsigned long long num_tdm_simplify;
  unsigned long long num_tdm_gpu_simplify;

  Search();

  ~Search();

  void store_isomorph_state();

  void restore_isomorph_state();

  void serialize(std::ostream& os) const;
  void deserialize(std::istream& is);
};
