#pragma once

#include <set>
#include "Puz.h"
// Missing these includes at the top:
#include <iostream>
#include <iomanip>
#include <fstream>
#include <set>
#include <cassert>
#include <cstdio>

#define QUEUE_LIMIT 100000
#define QUEUE_LEVELS 3
#define QUEUE_PRIORS 3

typedef struct _q_elt_t {
  double gap;
  Puz p;
  int elt_type = 0;

  void serialize(ostream& os) const {
    os.write(reinterpret_cast<const char*>(&gap), sizeof(gap));
    p.serialize(os);
    os.write(reinterpret_cast<const char*>(&elt_type), sizeof(elt_type));
  }

  void deserialize(istream& is) {
    is.read(reinterpret_cast<char*>(&gap), sizeof(gap));
    p.deserialize(is);
    is.read(reinterpret_cast<char*>(&elt_type), sizeof(elt_type));
  }

} q_elt_t;

struct cmp {
  bool operator()(q_elt_t left, q_elt_t right) const {
    return left.gap > right.gap;
  };
};

typedef std::multiset<q_elt_t, cmp> set_t;

class BoundedPuzPriorityQueue {
 private:
  set_t q;
  double best_prior;  // Bounds may not be valid if empty.
  double worst_prior;
  size_t limit;
  Puz dummy_p;

 public:
  void serialize(ostream& os) const {
    size_t size = q.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& elt : q) {
      elt.serialize(os);
    }
    os.write(reinterpret_cast<const char*>(&best_prior), sizeof(best_prior));
    os.write(reinterpret_cast<const char*>(&worst_prior), sizeof(worst_prior));
    os.write(reinterpret_cast<const char*>(&limit), sizeof(limit));
  }

  void deserialize(istream& is) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    q.clear();
    for (int i = 0; i < size; i++) {
      q_elt_t elt;
      elt.deserialize(is);
      q.insert(elt);
    }
    is.read(reinterpret_cast<char*>(&best_prior), sizeof(best_prior));
    is.read(reinterpret_cast<char*>(&worst_prior), sizeof(worst_prior));
    is.read(reinterpret_cast<char*>(&limit), sizeof(limit));
  }

  BoundedPuzPriorityQueue(size_t limit = QUEUE_LIMIT) : limit(limit) {
    assert(limit > 0);
    dummy_p = Puz(1, 1);
  }

  bool enqueue(const q_elt_t& t) {
    double prior = t.gap;

    // At limit and no better than worst, reject.
    if (size() == limit && prior <= worst_prior) {
      return false;
    }

    //  At limit, remove worst first.
    if (size() == limit) {
      // auto worst = q.lower_bound(tuple<double, Puz>{
      //	  max_prior, dummy_p});
      q_elt_t elt = *(q.rbegin());
      q.erase(elt);
      worst_prior = (*q.rbegin()).gap;
    }

    // Insert new element.
    q.insert(t);

    if (size() == 1) {
      best_prior = prior;
      worst_prior = prior;
    } else {
      best_prior = (prior > best_prior ? prior : best_prior);
      worst_prior = (prior < worst_prior ? prior : worst_prior);
    }

    return true;
  }

  size_t mass_enqueue(BoundedPuzPriorityQueue* list) {
    size_t enqueued = 0;
    size_t l_size = list->size();
    for (int i = 0; i < l_size; i++) {
      q_elt_t e;
      list->dequeue(e);
      assert(e.p.getHeight() > 0);
      enqueued += (enqueue(e) ? 1 : 0);
    }

    return enqueued;
  }

  bool dequeue(q_elt_t& elt) {
    if (is_empty())
      return false;

    auto best = *(q.begin());

    elt = best;
    q.erase(best);

    if (size() > 1)
      best_prior = q.begin()->gap;

    return true;
  }

  size_t size() const {
    return q.size();
  }

  double best() {
    return best_prior;
  }

  double worst() {
    return worst_prior;
  }

  bool is_empty() {
    return size() == 0;
  }

  void clear() {
    q.clear();
  }

  void display() {
    printf("%6ld, [%6.4e..%6.4e]\n", size(), best_prior, worst_prior);
    // printf("q.size(): %6ld\n", q.size());
  }

  void display(std::ostream& os) const {
    os << std::setw(6) << size() << ", [" << std::setprecision(4) << std::scientific << best_prior
       << ".." << std::setprecision(4) << std::scientific << worst_prior << "]\n";
  }
};

using BPPQ = BoundedPuzPriorityQueue;

class BoundedPuzMultilevelFeedbackQueue {
 private:
  size_t limit;
  unsigned int levels;
  unsigned int priors;
  BPPQ** queues;

 public:
  void serialize(ostream& os) const {
    os.write(reinterpret_cast<const char*>(&limit), sizeof(limit));
    os.write(reinterpret_cast<const char*>(&levels), sizeof(levels));
    os.write(reinterpret_cast<const char*>(&priors), sizeof(priors));
    for (int i = 0; i < levels * priors; i++)
      queues[i]->serialize(os);
  }

  void deserialize(istream& is) {
    is.read(reinterpret_cast<char*>(&limit), sizeof(limit));
    is.read(reinterpret_cast<char*>(&levels), sizeof(levels));
    is.read(reinterpret_cast<char*>(&priors), sizeof(priors));
    if (queues != NULL) {
      for (int i = 0; i < levels * priors; i++) {
        delete queues[i];
        queues[i] = NULL;
      }
      delete[] queues;
    }
    queues = new BPPQ*[levels * priors];
    for (int i = 0; i < levels * priors; i++) {
      queues[i] = new BPPQ(limit);
      queues[i]->deserialize(is);
    }
  }

  BoundedPuzMultilevelFeedbackQueue(size_t limit = QUEUE_LIMIT, unsigned int levels = QUEUE_LEVELS,
                                    unsigned int priors = QUEUE_PRIORS)
      : limit(limit), levels(levels), priors(priors) {
    queues = new BPPQ*[levels * priors];
    for (int i = 0; i < levels * priors; i++)
      queues[i] = new BPPQ(limit);
  }

  // XXX - Default copy constructor is not correct, but I don't think
  // we want to copy it anyway.

  ~BoundedPuzMultilevelFeedbackQueue() {
    for (int i = 0; i < levels * priors; i++) {
      delete queues[i];
      queues[i] = NULL;
    }
    delete[] queues;
  }

  size_t mass_enqueue(BPPQ** lists, int q_level) {
    assert(q_level < levels && q_level >= 0);
    size_t ret = 0;
    for (int i = 0; i < priors; i++) {
      ret += queues[q_level * priors + i]->mass_enqueue(lists[i]);
    }
    return ret;
  }

  bool enqueue(const q_elt_t& t, int q_level, int q_prior) {
    assert(q_level < levels && q_level >= 0 && q_prior < priors && q_prior >= 0);
    return queues[q_level * priors + q_prior]->enqueue(t);
  }

  // Picks at random from the non-empty queues.  Returns false if all
  // empty, otherwise set elt and idx on return.
  bool dequeue(q_elt_t& elt, int& q_level, int& q_prior) {
    if (is_empty())
      return false;

    int dequeue_mode = 2;
    int n = levels * priors;
    int i;

    if (dequeue_mode == 0) {
      // Uniform random among all non-empty queues.
      i = random() % n;
      while (queues[i]->is_empty())
        i = random() % n;
      q_level = i / priors;
      q_prior = i % priors;
      queues[i]->dequeue(elt);
    } else if (dequeue_mode == 1) {
      // Quadratically weighted towards higher levels, uniform in level.
      q_level = (int) (floor(sqrt(random()))) % levels;
      q_prior = random() % priors;  //(int)(floor(sqrt(random()))) % priors;
      while (queues[q_level * priors + q_prior]->is_empty()) {
        q_level = (int) (floor(sqrt(random()))) % levels;
        q_prior = random() % priors;  //(int)(floor(sqrt(random()))) % priors;
      }
      queues[q_level * priors + q_prior]->dequeue(elt);
    } else if (dequeue_mode == 2) {
      // Quadratically weighted towards higher levels,
      // Weight distribution in level by best.

      bool found = false;
      while (!found) {
        q_level = (int) (floor(sqrt(random()))) % levels;
        for (int i = 0; i < priors && !found; i++)
          if (!(queues[q_level * priors + i]->is_empty()))
            found = true;
      }

      double total_weight = 0;
      for (int i = 0; i < priors; i++)
        if (!(queues[q_level * priors + i]->is_empty()))
          total_weight += 1.0 / queues[q_level * priors + i]->best();

      q_prior = -1;
      double choice = (double) (random() % 1000000) / 1000000.0;
      double cumulative_weight = -0.000001;
      for (int i = 0; i < priors; i++)
        if (!(queues[q_level * priors + i]->is_empty())) {
          if (q_prior == -1)
            q_prior = i;
          cumulative_weight += (1.0 / queues[q_level * priors + i]->best()) / total_weight;
          if (cumulative_weight >= choice) {
            q_prior = i;
            break;
          }
        }
      assert(q_prior >= 0);
      queues[q_level * priors + q_prior]->dequeue(elt);
    }

    return true;
  }

  size_t size() {
    size_t ret = 0;
    for (int i = 0; i < levels * priors; i++)
      ret += queues[i]->size();
    return ret;
  }

  double best() {
    double ret = queues[0]->best();
    for (int i = 0; i < levels * priors; i++)
      if (!queues[i]->is_empty())
        ret = (queues[i]->best() > ret ? queues[i]->best() : ret);
    return ret;
  }

  double worst() {
    double ret = queues[0]->worst();
    for (int i = 0; i < levels * priors; i++)
      if (!queues[i]->is_empty())
        ret = (queues[i]->worst() < ret ? queues[i]->worst() : ret);
    return ret;
  }

  bool is_empty() {
    return size() == 0;
  }

  void clear() {
    for (int i = 0; i < levels * priors; i++)
      queues[i]->clear();
  }

  void display() {
    for (int i = 0; i < levels; i++) {
      cout << "Level: " << i << '\n';
      for (int j = 0; j < priors; j++) {
        if (!(queues[i * priors + j]->is_empty())) {
          printf("p: %2d, ", j);
          queues[i * priors + j]->display();
        }
      }
    }
  }

  void display(std::ostream& os) const {
    for (int i = 0; i < levels; i++) {
      os << "Level: " << i << '\n';
      for (int j = 0; j < priors; j++) {
        if (!(queues[i * priors + j]->is_empty())) {
          os << "p: " << std::setw(2) << j << ", ";
          queues[i * priors + j]->display(os);
        }
      }
    }
  }

  void restart() {
    // Clear everything.
    for (int i = 0; i < levels; i++)
      for (int j = 0; j < priors; j++)
        queues[i * priors + j]->clear();

    // Clear everything except last and swap to mid.
    // for (int i = 0; i < levels - 1; i++)
    //   for (int j = 0; j < priors; j++) {
    // 	queues[i * priors + j] -> clear();
    //   }

    // for (int j = 0; j < priors; j++) {
    //   auto tmp = queues[(levels-1) * priors + j];
    //   queues[(levels-1) * priors + j] = queues[levels/2 * priors + j];
    //   queues[levels/2 * priors + j] = tmp;
    // }
  }
};
