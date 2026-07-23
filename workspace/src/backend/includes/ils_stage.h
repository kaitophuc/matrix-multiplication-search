#pragma once

#include <vector>
#include <memory>
#include <string>
#include "Puz.h"
#include "BoundedPuzPriorityQueue.h"
#include "fitness.h"
#include "stats.h"
#include "elt_type.h"
#include "checker.h"

using BPMFQ = BoundedPuzMultilevelFeedbackQueue;

class ILSStage {
 public:
  stats_t stats;
  BPMFQ* work_queues;
  std::shared_ptr<ILSStage> prev;
  FitTester* ft = NULL;

  // Constructor for an initial stage.
  ILSStage(const Puz& p, FitTester* ft);

  // Constructor for a stage starting from a previous stage.
  ILSStage(unsigned int s, unsigned int k, FitTester* ft, std::shared_ptr<ILSStage> prev);

  ~ILSStage();

  unsigned int getHeight();

  // Perform one unit of work on this stage.
  bool doWork(check_mode_t mode, bool strong);

  // Called by the next stage: returns the next SUSP.
  Puz* takeResult(bool pop = true);

  unsigned int numResult();

  // Safely deactivates the previous stage.
  void deactivatePrev();

  void try_restart();

  void display_stats(std::string counts_str);

  void serialize(std::ostream& os) const;

  void deserialize(std::istream& is);

  void setPrev(std::shared_ptr<ILSStage> p);

 private:
  unsigned int s;
  unsigned int k;

  std::vector<Puz> results;

  void init_stats();

  bool is_active(int r, const std::vector<int>& idxs);

  void update_queue(BPMFQ* q, BPPQ** lists, unsigned int q_level, stats_t& stats);

  unsigned int crop_prior(int q_prior);

  bool add_to_frontier(const Puz& p, FitTester* ft, double gap, BPPQ** new_frontiers,
                       unsigned int q_prior, int type);

  void expand_frontier_GPU(std::vector<Puz>& puzzles, check_mode_t mode, bool strong, FitTester* ft,
                           BPPQ** new_frontiers, int q_prior, double gap, int num, elt_type type);
  void expand_frontier_GPU_row_replacements(const Puz& base, int row_to_replace,
                                            check_mode_t mode, bool strong, FitTester* ft,
                                            BPPQ** new_frontiers, int q_prior,
                                            double gap, elt_type type);

  void expand_frontier(Puz& p, check_mode_t mode, bool strong, FitTester* ft, BPPQ** new_frontiers,
                       int q_prior, double gap, bool brute);
};
