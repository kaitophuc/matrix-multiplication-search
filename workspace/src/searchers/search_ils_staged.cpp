/*
 * Iterated local search of USP and SUSP.
 * Currently fastest search algorithm as of Spring '22.
 *
 * Author: Matt Anderson
 */

#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <tuple>
#include <set>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <strings.h>
#include <time.h>
#include <curl/curl.h>
#include <omp.h>
#include <iostream>
#include <fstream>

#include "Puz.h"
#include "BoundedPuzPriorityQueue.h"
#include "special.h"
#include "checker.h"
#include "fitness.h"
#include "canonization.h"

using namespace std;

enum elt_type { NO_TYPE = 0, INIT, DERV, RAND, PERM_ELT, PERM_ROW, ALL, COL_SWAP, REPL_ROW };
#define NUM_ELT_TYPE 9

#define RESTART_THRESH (s * 500)
#define MAX_RESTARTS 1000
#define MAX_RESULTS 20
#define Q_SIZE (s * 1000)
#define Q_LEVELS 5
#define Q_PRIORS 5
#define REQUIRE_SPECIAL 0
#define RANDOM_FRONTIER 10 * (((double) stats.since_progress / 1000.0 + 1))

#define NUM_STAGES 4

// Loading Probabilities
#define PROB_PREV 0.3
#define PROB_GET_PREV 0.005
#define PROB_CURR (1.0 - PROB_GET_PREV - PROB_PREV)

using BPPQ = BoundedPuzPriorityQueue;
using BPMFQ = BoundedPuzMultilevelFeedbackQueue;

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
  chrono::high_resolution_clock::time_point start;
  int searched_types[NUM_ELT_TYPE];

} stats_t;

// Thread-safe prg in [0..1).
double safe_rand() {
  double d = (double) (random() % 1000000) / 1000000.0;
  return d;
}

// A part of a producer-consumer system for generating SUSP.  Each
// stage is responsible for specific number of puzzle rows.
class ILSStage {
 public:
  stats_t stats;

 private:
  unsigned int s;
  unsigned int k;
  ILSStage* prev = NULL;
  FitTester* ft = NULL;

  BPMFQ* work_queues;
  vector<Puz> results;

  void init_stats() {
    stats = {0, 0, 0, 0, 0, 0, 1, 0, 0, ILS_INACTIVE, chrono::high_resolution_clock::now()};
    TDM::setNumSimplify(0);
    TDM::setNumGPUSimplify(0);
    bzero(stats.searched_types, NUM_ELT_TYPE * sizeof(int));
  }

  bool is_active(int r, const vector<int>& idxs) {
    if (idxs.size() == 0)
      return true;  // Used when CHECK_FULL.
    for (int i = 0; i < idxs.size(); i++)
      if (idxs[i] == r)
        return true;
    return false;
  }

  void update_queue(BPMFQ* q, BPPQ** lists, unsigned int q_level, stats_t& stats) {
    size_t size = 0;
    for (int i = 0; i < Q_PRIORS; i++) {
      size += lists[i]->size();
    }

    size_t enqueued = q->mass_enqueue(lists, q_level);
    stats.dropped += size - enqueued;
    stats.evals += size;
  }

  unsigned int crop_prior(int q_prior) {
    if (q_prior < 0)
      return 0;
    if (q_prior >= Q_PRIORS)
      return Q_PRIORS - 1;
    return (unsigned int) q_prior;
  }

  bool add_to_frontier(const Puz& p, FitTester* ft, double gap, BPPQ** new_frontiers,
                       unsigned int q_prior, int type) {
    double curr_fit, max_fit;
    tie(curr_fit, max_fit) = ft->getFit(p);

    if (curr_fit - max_fit < gap) {
      // Got worse, reset to 0 priority.
      new_frontiers[crop_prior(q_prior - 2)]->enqueue(
          q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
    } else if (curr_fit - max_fit == gap) {
      // Stayed same, keep same priority.
      new_frontiers[crop_prior(q_prior - 1)]->enqueue(
          q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
    } else {
      // Got better, improve priority.
      new_frontiers[crop_prior(q_prior + 1)]->enqueue(
          q_elt_t{(curr_fit - max_fit) - safe_rand(), p, type});
    }

    return (prev != NULL && curr_fit == max_fit);
  }

  void expand_frontier(Puz& p, check_mode_t mode, bool strong, FitTester* ft, BPPQ** new_frontiers,
                       int q_prior, double gap, bool brute) {
    int s = p.getHeight();
    int k = p.getWidth();

    if (brute) {
      // Try all possibilities for the last row.
      unsigned long long num_rows = pow(3, k);

      for (unsigned long long row = 0; row < num_rows; row++) {
        Puz p_curr = p;
        unsigned long long curr = row;
        for (int c = 0; c < k; c++) {
          p_curr.set(s - 1, c, curr % 3 + 1);
          curr /= 3;
        }

        if (add_to_frontier(p_curr, ft, gap, new_frontiers, q_prior, ALL))
          return;

        // Break early if already have a result and are current.
        if (prev != NULL && numResult() != 0) {
          return;
        }
      }

      return;
    }

    // The puzzle hasn't been examined, calculate the active
    // indexes when mode isn't CHECK_FULL.
    vector<int> idxs;
    if (mode != CHECK_FULL) {
      TDM tdm(p, strong, TDM::ALLOW_GPU);
      if (mode == CHECK_OBVIOUS)
        tdm.simplify(TDM::EAGER);
      tdm.count(&idxs);
    }

    // Try all possibilities for a random active row.
    // Theta(3^k * Check(s,k))
    if (idxs.size() > 0) {
      int r = idxs[lrand48() % idxs.size()];
      if (r < s - 1) {
        unsigned long long num_rows = pow(3, k);

        for (unsigned long long row = 0; row < num_rows; row++) {
          Puz p_curr = p;
          unsigned long long curr = row;
          for (int c = 0; c < k; c++) {
            p_curr.set(r, c, curr % 3 + 1);
            curr /= 3;
          }

          if (add_to_frontier(p_curr, ft, gap, new_frontiers, q_prior, REPL_ROW))
            return;

          // Break early if already have a result and are current.
          if (prev != NULL && numResult() != 0) {
            return;
          }
        }
      }
    }

    // Explore Derivatives
    // Loop over rows.
    // Theta(s * k * Check(s,k))
    for (int r = 0; r < s; r++) {
      // Only consider active rows
      bool found = is_active(r, idxs);  // true;
      if (!found)
        continue;

      // Loop over columns.
      for (int c = 0; c < k; c++) {
        // Store current value of r, c.
        unsigned int curr = p.get(r, c);

        // Consider alternatives values.
        for (int dv = 1; dv <= 2; dv++) {
          int val = ((curr - 1 + dv) % 3) + 1;
          p.set(r, c, val);

          // Calculate fitness of new puzzle and add to
          // the local search frontier.
          if (add_to_frontier(p, ft, gap, new_frontiers, q_prior, DERV))
            return;
        }
        // Reset value to original.
        p.set(r, c, curr);
      }
    }

    // Explore permuting element values in row or column.
    // Theta((s+k) * Check(s,k))
    int perms[6][3] = {{0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

    for (int pi = 1; pi < 6; pi++) {
      for (int i = 0; i < s + k; i++) {
        bool is_col = (i >= s);
        int idx = (is_col ? i - s : i);
        Puz p_curr = p;

        for (int j = 0; j < (is_col ? s : k); j++) {
          int r = (is_col ? j : idx);
          int c = (is_col ? idx : j);

          int val = p_curr.get(r, c);
          p_curr.set(r, c, perms[pi][val - 1] + 1);
        }

        // Calculate fitness of new puzzle and add to
        // the local search frontier.
        if (add_to_frontier(p_curr, ft, gap, new_frontiers, q_prior, PERM_ELT))
          return;
      }
    }

    // int num_rand = RANDOM_FRONTIER;

    // // Explore randomly replace each row / column.
    // // Theta((s+k) * num_rand * Check(s,k)).
    // for (int n = 1; n < num_rand; n++){
    //   for (int i = 0; i < s + k; i++){
    //     bool is_col = (i >= s);
    //     int idx = (is_col ? i - s : i);
    //     Puz p_curr = p;

    //     for (int j = (is_col ? 0 : REQUIRE_SPECIAL);
    //          j < (is_col ? s : k); j++){
    //       int r = (is_col ? j : idx);
    //       int c = (is_col ? idx : j);

    //       p_curr.set(r, c, ((int) random()) % 3 + 1);
    //     }

    //     // Calculate fitness of new puzzle and add to
    //     // the local search frontier.
    //     if (add_to_frontier(p_curr, ft, gap, new_frontiers,
    //                         q_prior, RAND)) return;
    //   }
    // }

    /*
    // Explore swapping columns
    // Theta(k^2*s).
    // XXX - This never triggers because they're isomorphic.
    for (int c1 = 0; c1 < k - 1; c1++){
      for (int c2 = c1 + 1; c2 < k; c2++){
  for (int r = 0; r < s; r++){
    int tmp = p.get(r, c1);
    p.set(r, c1, p.get(r, c2));
    p.set(r, c2, tmp);
  }
  if (add_to_frontier(p, ft, gap, new_frontiers,
      q_prior, COL_SWAP)) return;
  for (int r = 0; r < s; r++){
    int tmp = p.get(r, c1);
    p.set(r, c1, p.get(r, c2));
    p.set(r, c2, tmp);
  }
      }
    }
    */

    // // Replace rows with a permutations of other rows.
    // for (int r1 = 0; r1 < s; r1++){

    //   // Only consider active rows
    //   bool found = is_active(r1, idxs); // true;
    //   if (!found) continue;

    //   for (int r2 = 0; r2 < s; r2++){

    //     if (r2 == r1) continue;

    //     for (int pi = 0; pi < 6; pi++){

    //       Puz p_curr = p;

    //       for (int c = 0; //REQUIRE_SPECIAL;
    //            c < k; c++){
    //         int val = p_curr.get(r2, c);
    //         p_curr.set(r1, c, perms[pi][val-1] + 1);
    //       }

    //       if (add_to_frontier(p_curr, ft, gap, new_frontiers,
    //     		      q_prior, PERM_ROW)) return;
    //     }
    //   }
    // }
  }

 public:
  // Creates an ILS Stage searching from the results of a previous stage.
  ILSStage(unsigned int s, unsigned int k, FitTester* ft, ILSStage* prev)
      : s(s), k(k), ft(ft), prev(prev) {
    init_stats();
    stats.mode = ILS_SEARCH;
    assert(prev != NULL && prev->stats.mode == ILS_FEED);
    work_queues = new BPMFQ(Q_SIZE, Q_LEVELS, Q_PRIORS);
  }

  // Creates an ILS Stage searching from an initial puzzle.
  ILSStage(const Puz p, FitTester* ft) : ft(ft) {
    s = p.getHeight();
    k = p.getWidth();
    prev = NULL;
    init_stats();
    stats.mode = ILS_SEARCH;
    work_queues = new BPMFQ(Q_SIZE, Q_LEVELS, Q_PRIORS);

    double fit, max_fit;
    tie(fit, max_fit) = ft->getFit(p);
    stats.evals++;
    work_queues->enqueue(q_elt_t{(fit - max_fit) - safe_rand(), p, INIT}, 0, 0);
  }

  // Default copy constructor should work.
  // ILSStage(const ILStage other){
  // }

  ~ILSStage() {
    // Pointers are deleted by caller.
    delete work_queues;
  }

  unsigned int getHeight() {
    return s;
  }

  // Do one unit of work on this stage, must be reentrant.
  bool doWork(check_mode_t mode, bool strong) {
    assert(stats.mode != ILS_INACTIVE);

    bool working = false;

    Puz p;
    double gap;
    q_elt_t e;
    int q_level;
    int q_prior;

// If queue(s) is empty, return.
// Pop next element from queue(s).
#pragma omp critical(queue)
    {
      bool success1 = false;
      if (prev != NULL &&
          (work_queues->is_empty() || safe_rand() <= PROB_GET_PREV / (PROB_GET_PREV + PROB_CURR))) {
        Puz* p_ptr = prev->takeResult();
        if (p_ptr != NULL) {
          working = true;
          success1 = true;
          p = *p_ptr;
          p = p.resizePuz(1);
          delete p_ptr;
          e.elt_type = INIT;
          q_level = -1;
          q_prior = 0;
          double fit, max_fit;
          tie(fit, max_fit) = ft->getFit(p);
          gap = fit - max_fit;
        }
      }

      bool success2 = false;
      if (!success1 && !work_queues->is_empty()) {
        working = true;
#pragma omp critical(stats)
        stats.working++;
        work_queues->dequeue(e, q_level, q_prior);
        gap = e.gap;
        p = e.p;
        success2 = true;
      }
    }

    if (!working)
      return false;

    // If the puzzle has already been examined -- its in the
    // isomorph cache, skip it.
    bool seen = false;
#pragma omp critical(isomorph)
    seen = have_seen_isomorph(p, true);
    if (seen) {
#pragma omp critical(stats)
      {
        stats.repeated++;
        stats.working--;
      }
      return true;
    }

    // Update best result.
    gap = ceil(gap);
#pragma omp critical(stats)
    if (stats.best_gap > 0 || stats.best_gap < gap) {
      stats.best_gap = gap;
      stats.since_progress = 0;
      // printf("\nnew best_gap = %6.4e", stats.best_gap);
    }

    // If the puzzle has maximum fitness, we've found a new solution.
    // Update mode, push to results.
    if (gap == 0) {
#pragma omp critical(stats)
      {
        stats.max_found++;
        stats.mode = ILS_FEED;
      }
#pragma omp critical(results)
      results.push_back(p);
      // if (prev != NULL)
      //  return true;
    }

    // Expand frontier on element.

    // Local variable for collecting next frontier.
    BPPQ* local_qs[Q_PRIORS];
    for (int i = 0; i < Q_PRIORS; i++)
      local_qs[i] = new BPPQ(Q_SIZE);

#pragma omp critical(stats)
    {
      stats.proc++;
      stats.since_progress++;
      stats.searched_types[e.elt_type]++;
    }

    expand_frontier(p, mode, strong, ft, local_qs, q_prior, gap, e.elt_type == INIT);

// Add the local search frontier to the global search
// frontier in a thread safe way.
#pragma omp critical(queue)
#pragma omp critical(stats)
    {
      update_queue(work_queues, local_qs, (q_level < Q_LEVELS - 1 ? q_level + 1 : Q_LEVELS - 1),
                   stats);
      stats.working--;
    }

    for (int i = 0; i < Q_PRIORS; i++)
      delete local_qs[i];

    return true;
  }

  // Called by next stage, returns next SUSP. Must be reentrant and
  // non-blocking.  NULL is no available SUSP. Must be deallocated by
  // caller.
  Puz* takeResult(bool pop = true) {
    Puz* res = NULL;
#pragma omp critical(results)
    {
      if (results.size() > 0) {
        res = new Puz(results.back());
        if (pop)
          results.pop_back();
      }
    }
    return res;
  }

  // Returns whether there are results.
  unsigned int numResult() {
    unsigned int res = 0;
#pragma omp critical(results)
    res = results.size();
    return res;
  }

  void deactivatePrev() {
#pragma omp critical(queue)
    prev = NULL;
  }

  void try_restart() {
    work_queues->restart();
    stats.since_progress = 0;
    stats.best_gap = 1;
    stats.restarts++;
    cout << endl << "------- Restart #" << stats.restarts << "-------" << endl;
  }

  void display_stats(string counts_str) {
    usleep(200000);
#pragma omp critical(stats)
    {
      if (stats.proc != 0) {
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration<double>(stop - stats.start);
        double time_secs = duration.count();

        printf("\033[150A\r\033[J");
        work_queues->display();

        double eval_rate = 0;
        // if (duration.count() > 0)
        eval_rate = stats.evals / time_secs;
        printf("\ns: %3d, #: %6lld (%6lld, r: %3lld), Gap: %6.4e, ", s, stats.proc,
               stats.since_progress, stats.restarts, stats.best_gap);

        printf("\nQ: %7ld [%6.4e .. %6.4e]", work_queues->size(), work_queues->best(),
               work_queues->worst());
        cout << " (Prev Stages: " << counts_str << "), ";

        printf("\nDistinct: %6ld, Repeated: %6lld ", get_num_isomorphs(), stats.repeated);
        printf("\nEval: %6.4e (per sec: %6.0f), ", (double) (stats.evals), eval_rate);
        auto num_gpu = TDM::getNumGPUSimplify();
        auto num_cpu = TDM::getNumSimplify();
        auto tot_simplify = num_gpu + num_cpu;
        printf("\nGPU Simplify: %8llu (%3.0f%%) (per sec: %8.2f)", num_gpu,
               (tot_simplify ? 100.0 * num_gpu / tot_simplify : 0.0),
               (time_secs ? num_gpu / time_secs : 0.0));
        printf("\nCPU Simplify: %8llu (%3.0f%%) (per sec: %8.2f)", num_cpu,
               (tot_simplify ? 100.0 * num_cpu / tot_simplify : 0.0),
               (time_secs ? num_cpu / time_secs : 0.0));
        printf("\nTot Simplify: %8llu (100%%) (per sec: %8.2f)", tot_simplify,
               (time_secs ? tot_simplify / time_secs : 0.0));
        assert(stats.searched_types[0] == 0);
        printf("\nST: [INIT:%4d, DERV:%4d, RAND:%4d, PERM_ELT:%4d, PERM_ROW:%4d, ALL:%4d, "
               "COL_SWAP:%4d, REPL_ROW: %4d]",
               stats.searched_types[1], stats.searched_types[2], stats.searched_types[3],
               stats.searched_types[4], stats.searched_types[5], stats.searched_types[6],
               stats.searched_types[7], stats.searched_types[8]);

        fflush(stdout);
      }
    }
  }
};

bool search_ils(Puz& p, int s_target, check_mode_t mode, bool strong) {
  int s = p.getHeight();
  int k = p.getWidth();

  // Clear isomorphs, looking at new size.
  reset_isomorphs();

  // Initialize fitness tester.
  FitTester* ft = NULL;
  if (mode == CHECK_FULL)
    ft = new SubsetFitTester(strong);
  else
    ft = new TDMSizeFitTester(mode, strong, REQUIRE_SPECIAL);

  int num_stages = NUM_STAGES;
  assert(num_stages >= 2);  // Must be at least 2.
  int curr = num_stages - 1;
  ILSStage* stages[num_stages];
  for (int i = 0; i < num_stages; i++)
    stages[i] = NULL;
  stages[curr] = new ILSStage(p, ft);

  for (; s <= s_target; s++) {
    cout << endl << endl << ">>>>>>>> NEW STAGE (s = " << s << ") <<<<<<<<<" << endl;

    // Work in parallel on the current stage.

    while (stages[curr]->numResult() == 0) {
      bool restart = false;
#pragma omp parallel
      while (stages[curr]->numResult() == 0 && !restart) {
        // Some kind of load balancing.
        int thread_num = omp_get_thread_num();
        if (thread_num == 0) {
#pragma omp critical(stats)
          if (stages[curr]->stats.since_progress > RESTART_THRESH) {
            restart = true;
#pragma omp flush(restart)
          }

          string counts_str = "";

          for (int i = curr - 1; i >= 0; i--) {
            if (stages[i] != NULL)
              counts_str += to_string(stages[i]->numResult());
            else
              counts_str += "X";
            if (i != 0)
              counts_str += " <- ";
          }
          stages[curr]->display_stats(counts_str);

        } else {
          for (int i = curr; i >= 0; i--)
            if ((i == 0 || stages[i - 1] == NULL) ||
                (i < curr && safe_rand() >= PROB_PREV && stages[i]->numResult() < MAX_RESULTS) ||
                (i == curr && safe_rand() >= PROB_PREV)) {
              stages[i]->doWork(mode, strong);
              break;
            }
        }
      }
      if (restart) {
        stages[curr]->try_restart();
      }
    }

    Puz* result = stages[curr]->takeResult(false);
    assert(result != NULL);
    p = *result;
    delete result;
    cout << endl << p << endl;

    for (int i = 0; i < 100; i++)
      cout << endl;
    ofstream puzzle_log("puzzle_log.txt", ios::app);
    puzzle_log << p << endl;
    puzzle_log.close();
    assert(p.checkPuz(mode, strong));

    // Create a next stage and remove old stages if necesssary.
    if (stages[0] != NULL) {
      assert(stages[1] != NULL);
      stages[1]->deactivatePrev();
      delete stages[0];
    }

    for (int i = 0; i < num_stages - 1; i++)
      stages[i] = stages[i + 1];
    stages[curr] = new ILSStage(s + 1, k, ft, stages[curr - 1]);
  }

  // The loop exits only when expected result is found.

  // Clean up.
  delete ft;
  for (int i = 0; i < num_stages; i++)
    if (stages[i] != NULL)
      delete stages[i];

  return true;
}

// Robert Jenkins' 96 bit Mix Function
unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
  a = a - b;
  a = a - c;
  a = a ^ (c >> 13);
  b = b - c;
  b = b - a;
  b = b ^ (a << 8);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 13);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 12);
  b = b - c;
  b = b - a;
  b = b ^ (a << 16);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 5);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 3);
  b = b - c;
  b = b - a;
  b = b ^ (a << 10);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 15);
  return c;
}

int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 6) {
    cerr << "usage: search_ils_staged <mode> <strong> <s_target> <k> [<start>]" << endl;
    cerr << endl;
    cerr << "- <mode> = 0 does check" << endl;
    cerr << "         = 1 does simplifiable (obvious) check" << endl;
    cerr << "         = 2 does local check" << endl;
    cerr << endl;
    cerr << "- <strong> = 0 does check" << endl;
    cerr << "           = 1 does strong check" << endl;
    cerr << endl;
    cerr << "- <s_target> is number of rows to search up to." << endl;
    cerr << endl;
    cerr << "- <k> is puzzle width to search for." << endl;
    cerr << endl;
    cerr << "- [<start>] is an optional argument which is a file in .puz format containing a "
            "puzzle to start from."
         << endl;
    cerr << endl;
    cerr << "Common usage:" << endl;
    cerr << "  search_ils_staged 1 1 14 6" << endl;
    cerr << "  searches for a Simplifiable Strong USP with 6 columns and up to 14 rows." << endl;

    return -1;
  }

  int mode_i = atoi(argv[1]);
  check_mode_t mode = (mode_i == 0 ? CHECK_FULL : (mode_i == 1 ? CHECK_OBVIOUS : CHECK_LOCAL));
  bool strong = atoi(argv[2]) == 1;
  int s_target = atoi(argv[3]);
  int s = 1;
  int k = atoi(argv[4]);
  Puz p(s, k);

  if (argc == 6) {
    p = Puz(argv[5]);
    assert(k == p.getWidth() && s_target >= p.getHeight());
    s = p.getHeight();
  }

  cout << p << endl;

  srandom(mix(clock(), time(NULL), getpid()));

  // omp_set_num_threads(40);

  search_ils(p, s_target, mode, strong);
  assert(p.checkPuz(mode, strong));
  p.sort();
  cout << endl << endl << p << endl;
  cout << "is an " << (mode == CHECK_OBVIOUS ? "obvious " : (mode == CHECK_LOCAL ? "local " : ""))
       << (strong ? "strong " : "") << "(" << p.getHeight() << "," << k << ")-USP." << endl;

  bool hier = hierarchically_special_unordered(p) == 0;
  if (hier) {
    cout << "is hierarchically special with reorder" << endl;
  } else {
    cout << "is NOT hierarchically special with reorder" << endl;
  }

  reset_isomorphs();  // For valgrinds sake.

  return 0;
}
