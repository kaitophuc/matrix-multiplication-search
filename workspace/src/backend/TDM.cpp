#include <stdio.h>
#include <strings.h>
#include "TDM.h"
#include "TDM_cuda.h"
#include "matching.h"
#include "Puz.h"

unsigned long long TDM::concurrent_on_gpu = 0;
std::atomic<unsigned long long> TDM::num_gpu_simplify{0};
std::atomic<unsigned long long> TDM::num_simplify{0};

/*
 * Constructs a (strong) TDM instance from given puzzle.
 * - if strong is set, it will use the strong version of the TDM instance.
 * - use_gpu is either TDM::NO_GPU (default), TDM::ALLOW_GPU, or TDM::FORCE_GPU.
 */
TDM::TDM(const Puz& _p, bool strong, gpuUsage use_gpu)
    : s(_p.getHeight()), strong(strong), use_gpu(use_gpu) {
  tdm = new bool[s * s * s];
  simplified = false;
  count_result = 0;
  computed = false;

  p = new Puz(_p);

  if (p->getSimplified()) {
    simplified = true;
    computed = true;
    count_result = p->getResults();
    for (int r1 = 0; r1 < s; r1++)
      for (int r2 = 0; r2 < s; r2++)
        for (int r3 = 0; r3 < s; r3++)
          set(r1, r2, r3, p->getTDM(r1, r2, r3));
  }

  if (use_gpu == NO_GPU)
    compute();
}

TDM::~TDM() {
  delete[] tdm;
  delete p;
}

bool TDM::operator==(const TDM& tr) const {
  if (s != tr.s)
    return false;

  for (int i = 0; i < s * s * s; i++)
    if (tdm[i] != tr.tdm[i])
      return false;

  return true;
}

/*
 * Calculates the 3D matching instance from the given puzzle.
 */
void TDM::compute() {
  if (!computed) {
    Puz p_local = *p;
    for (int r1 = 0; r1 < s; r1++)
      for (int r2 = 0; r2 < s; r2++)
        for (int r3 = 0; r3 < s; r3++)
          // set the value of the edge in the 3DM instance to
          // the value of the witness in the puzzle instance
          set(r1, r2, r3, !p_local.isWitness(r1, r2, r3, strong));

    computed = true;
  }
}

/*
 * Removes some edges from the 3DM instance for columns that contain
 * only two of the three values {1, 2, 3}.  Shouldn't effect whether
 * the puzzle is determined to be a SUSP or not, only effects
 * performance of checking.
 *
 * 1. Projects the 3DM instance to the three faces of the cube.
 * 2. Removed any entries in 3DM that are not present in some
 *    2D matching of face.
 * 3. Repeat Steps 1 & 2 until there are no changes.
 *
 * Modes:
 *   TDM::EAGER - Fully simplifies the puzzle (on the CPU or GPU)
 *                and updates the TDM instance to contain the
 *                simplified puzzle.
 *   TDM::LAZY  - Fully simplifies the puzzle (on the CPU or GPU),
 *                but doesn't necessarily update the TDM instance.
 *                May only update number of edges in the simplified
 *                instance.
 *   TDM::SLOW  - Like TDM::EAGER, but uses the slower CPU-only
 *                implementation, regardless of how use_gpu was set.
 */
void TDM::simplify(simplifyMode mode) {
  if (mode == TDM::EAGER || mode == TDM::LAZY) {
    bool res = simplify_gpu(mode);
    if (!res)
      simplify_cpu();

  } else if (mode == TDM::SLOW) {
    simplify_cpu_slow();

  } else {
    assert(false);
  }
}

/*
 * Private gpu implementation of simplify.
 * Unless TDM::FORCE_GPU, may opt not to simplify on the GPU.
 * Depending on mode, make not update the cpu TDM instance.
 */
bool TDM::simplify_gpu(simplifyMode mode) {
  if (simplified)
    return true;
  if (use_gpu == TDM::NO_GPU)
    return false;
  if (use_gpu == TDM::ALLOW_GPU && s < GPU_SIZE_THRESH)
    return false;

  bool stop = false;

#pragma omp critical(cuda_streams)
  {
    if (use_gpu != TDM::FORCE_GPU && TDM::concurrent_on_gpu >= MAX_ON_GPU) {
      // cout << "TOO BUSY" << '\n';
      stop = true;
    } else
      TDM::concurrent_on_gpu++;
  }

  if (stop)
    return false;

  int result = simplify_cuda_inner(*p, (mode == TDM::EAGER ? tdm : NULL), strong);
  assert(result >= 0);
  count_result = result;

#pragma omp critical(cuda_streams)
  {
    TDM::num_gpu_simplify.fetch_add(1, std::memory_order_relaxed);
    TDM::concurrent_on_gpu--;
  }

  if (mode == TDM::EAGER) {
    simplified = true;
    computed = true;
  }

  return true;
}

/*
 * Private cpu implementation of simplify.
 * Always updates the cpu TDM instance.
 */
void TDM::simplify_cpu() {
  if (simplified)
    return;

  if (!computed)
    compute();

  simplified = true;

  TDM::num_simplify.fetch_add(1, std::memory_order_relaxed);

  int no_change = 0;
  int iter = 0;

  // Calculate the projection to the current face.
  // This is \Theta(s^3).
  adj_node projections[3][s * s];
  // 0 -> (y, z); 1 -> (x, z); 2 -> (x, y).
  bzero(projections, 3 * s * s * sizeof(adj_node));
  // initialize to 0
  for (int face = 0; face < 3; face++) {
    for (int x = 0; x < s; x++) {
      for (int y = 0; y < s; y++) {
        projections[face][x * s + y].count = 0;
        projections[face][x * s + y].x = x;
        projections[face][x * s + y].y = y;
        for (int z = 0; z < s; z++) {
          bool entry = false;
          if (face == 0)
            entry = get(z, x, y);
          // face 0 - (y, z)
          if (face == 1)
            entry = get(x, z, y);
          // face 1 - (x, z)
          if (face == 2)
            entry = get(x, y, z);
          // face 2 - (x, y)
          projections[face][x * s + y].count += (entry ? 1 : 0);
          // count the number of edges in the projection
        }
      }
    }
    init_adj_nodes(projections[face], s);
    // create the adjacency list
  }

  // Loop until no face produces a change in the 3DM instace.
  // Loops at most O(s^2) times, because an entry in the projections
  // must change each iteration.  O(s^4) overall.
  while (no_change < 3) {
    bool changed = false;
    int face = iter % 3;
    // 0 -> (2, 3); 1 -> (1, 3); 2 -> (1, 2).

    // This is O(s + s^2) per iteration.
    vector<adj_node*> to_remove;
    allowed_matching_edges(projections[face], s, &to_remove);
    // get the edges that are allowed to be removed

    // This is O(|E| s) overall (not per iteration) because each edge
    // can only be removed once from a projection.
    for (int i = 0; i < to_remove.size(); i++) {
      adj_node* u = to_remove[i];
      int x = u->x;
      int y = u->y;

      del_adj_node(&(projections[face][x * s + y]));
      // remove the edge from the projection
      for (int z = 0; z < s; z++) {
        int a, b, c;
        if (face == 0) {
          a = z;
          b = x;
          c = y;
        }
        if (face == 1) {
          a = x;
          b = z;
          c = y;
        }
        if (face == 2) {
          a = x;
          b = y;
          c = z;
        }

        bool entry = get(a, b, c);
        // check if the edge is in the 3DM instance
        set(a, b, c, false);
        // remove the edge from the 3DM instance by setting it to false

        if (entry) {
          // if the edge was in the 3DM instance
          // Update projections.
          if (face == 0) {
            dec_adj_node(&(projections[1][z * s + y]));
            dec_adj_node(&(projections[2][z * s + x]));
          }
          if (face == 1) {
            dec_adj_node(&(projections[0][z * s + y]));
            dec_adj_node(&(projections[2][x * s + z]));
          }
          if (face == 2) {
            dec_adj_node(&(projections[0][y * s + z]));
            dec_adj_node(&(projections[1][x * s + z]));
          }
        }
      }
      changed = true;
      // The instance must have changed.
    }

    if (!changed)
      // Update counters.
      no_change++;
    else
      no_change = 0;
    iter++;
  }
}

/*
 * Slow, private cpu implementation of simplify.
 * Always updates the cpu TDM instance.
 */
void TDM::simplify_cpu_slow() {
  if (simplified)
    return;

  if (!computed)
    compute();

  simplified = true;

  int no_change = 0;
  int iter = 0;

  // Loop until no face produces a change in the 3DM instace.
  while (no_change < 3) {
    bool changed = false;
    int face = iter % 3;
    // 0 -> (2, 3); 1 -> (1, 3); 2 -> (1, 2).

    // Calculate the projection to the current face
    bool projection[s][s];
    for (int x = 0; x < s; x++) {
      for (int y = 0; y < s; y++) {
        projection[x][y] = false;
        for (int z = 0; z < s && !projection[x][y]; z++) {
          bool entry = false;
          if (face == 0)
            entry = get(z, x, y);
          if (face == 1)
            entry = get(x, z, y);
          if (face == 2)
            entry = get(x, y, z);
          projection[x][y] = entry;
        }
      }
    }

    // For each off-diagonal edge test whether it appears in
    // some matching of the face.  If not, delete it from the 3DM.
    for (int x = 0; x < s; x++) {
      for (int y = 0; y < s; y++) {
        // Only test cells with edges.
        if (x == y || !projection[x][y])
          continue;

        // Determine whether edge x, y is in some 2D matching of face.
        // Calculate minor of 2D matching instance with the edge x,y
        // present then evaluated 2D matching.
        bool minor[(s - 1) * (s - 1)];

        for (int x2 = 0; x2 < s; x2++) {
          if (x2 == x)
            continue;
          for (int y2 = 0; y2 < s; y2++) {
            if (y2 == y)
              continue;
            minor[(x2 - (x2 > x ? 1 : 0)) * (s - 1) + (y2 - (y2 > y ? 1 : 0))] = projection[x2][y2];
          }
        }

        bool is_present = has_perfect_bipartite_matching(minor, s - 1);

        // If the edge cannot be used, remove it from the projection
        // and the 3DM instance.
        if (!is_present) {
          projection[x][y] = false;
          for (int z = 0; z < s; z++) {
            if (face == 0)
              set(z, x, y, false);
            if (face == 1)
              set(x, z, y, false);
            if (face == 2)
              set(x, y, z, false);
          }
          changed = true;  // The instance must have changed.
        }
      }
    }

    // Update counters.
    if (!changed)
      no_change++;
    else
      no_change = 0;
    iter++;
  }
}

/*
 * Returns the number of edges in the 3DM instance.
 */
unsigned int TDM::count(vector<int>* active_idx, unsigned int diag_weight,
                        unsigned int near_diag_weight, unsigned int off_diag_weight) {
  if (active_idx == NULL && diag_weight == 1 && near_diag_weight == 1 && off_diag_weight == 1 &&
      count_result != 0)
    return count_result;

  // cout << computed << ", " << simplified << ", " << count_result << '\n';

  if (!computed)
    compute();

  // This function isn't general.  If the TDM has been simplified on
  // the gpu with TDM::LAZY, the instance hasn't been copied back, so
  // count will fail, so we resimplify it.
  // assert(count_result == 0 || simplified);
  if (count_result > 0 && !simplified) {
    simplify(TDM::EAGER);
  }

  int count = 0;
  for (int r1 = 0; r1 < s; r1++)
    for (int r2 = 0; r2 < s; r2++)
      for (int r3 = 0; r3 < s; r3++) {
        if (get(r1, r2, r3)) {
          if (r1 == r2 && r2 == r3) {
            count += diag_weight;
          } else if (r1 == r2 || r1 == r3 || r2 == r3) {
            count += near_diag_weight;
          } else {
            count += off_diag_weight;
          }

          if (!(r1 == r2 && r2 == r3) && active_idx != NULL) {
            active_idx->push_back(r1);
            active_idx->push_back(r2);
            active_idx->push_back(r3);
          }
        }
      }

  return count;
}

/*
 * Print the 3DM instance to the specified open file.
 */
void TDM::fprint(FILE* f, bool verbose) {
  if (!computed)
    compute();

  if (verbose) {
    if (strong)
      fprintf(f, "Strong TDM:\n");
    else
      fprintf(f, "TDM:\n");
  }

  for (int r1 = 0; r1 < s; r1++) {
    for (int r2 = 0; r2 < s; r2++) {
      for (int r3 = 0; r3 < s; r3++) {
        fprintf(f, "%d", get(r1, r2, r3));
      }
      fprintf(f, "\n");
    }
    fprintf(f, "\n");
  }
}

/*
 * Print the 3DM instance to the console.
 */
void TDM::print(bool verbose) {
  fprint(stdout, verbose);
}
