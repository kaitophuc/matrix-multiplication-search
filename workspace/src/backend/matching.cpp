#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include "constants.h"
#include "matching.h"
#include <vector>
#include <assert.h>
#include <strings.h>

using namespace std;

// Iterative depth-first search for augmenting path.  O(n^2)
bool augment_path(bool* G, int s, int t, int n) {
  int prev[n];
  for (int i = 0; i < n; i++) {
    prev[i] = -1;
  }

  prev[s] = s;

  int curr = s;
  int next = 0;

  while (curr != s || next != n) {
    //    printf("n = %d, curr = %d, next = %d\n", n, curr, next);

    for (; next < n; next++) {
      // printf("G[%d * n + %d] = %d\n",curr, next, G[curr * n + next]);
      if (curr != next && G[curr * n + next] && prev[next] == -1) {
        // printf("Found edge (%d, %d)\n", curr, next);
        if (next == t) {
          // Loop and update augmenting path.
          // printf("Found destination\n");
          while (next != s) {
            G[curr * n + next] = false;
            G[next * n + curr] = true;
            next = curr;
            curr = prev[curr];
          }
          return true;
        }
        prev[next] = curr;
        curr = next;
        next = 0;
        break;
      }
    }

    if (next == n && curr != s) {
      next = curr + 1;
      curr = prev[curr];
    }
  }

  return false;
}

void print_G(bool* G, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (G[i * n + j])
        printf("1");
      else
        printf("0");
    }
    printf("\n");
  }
}

// Implement Bipartite matching using Ford-Faulkerson flow algorithm.
// O(n^3) implementation.
bool has_perfect_bipartite_matching(bool* M, int n) {
  int N = 2 * n + 2;
  bool G[N * N];
  int s = N - 2;
  int t = N - 1;

  // 1. Initialize data structures
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      G[i * N + j] = false;

  for (int i = 0; i < n; i++) {
    G[s * N + i] = true;
    G[(n + i) * N + t] = true;
    for (int j = 0; j < n; j++) {
      G[i * N + (n + j)] = M[i * n + j];
    }
  }

  // print_G(G, N);

  // 2. Locate augmenting path.
  // 3. If path exists, put flow on path and repeat from Step 2.
  int matching_size = 0;
  while (augment_path(G, s, t, N)) {
    // print_G(G, N);
    matching_size++;
  }

  // printf("maximum matching = %d\n", matching_size);
  //  4. No path exists, return
  return matching_size == n;
}

// ==============================================================
//
// Calculates allowed matching edges
//
// Implements DFS, Strongly Connected Components as in Chapter 22 of
// CLRS and then uses Algorithm 2 of "Finding all maximally-matchable
// edges in a bipartite graph" to find all edges in some maximum 2D
// matching.  Implemented in time O(n + n^2) though could be
// implemented in time O(n + m) if using adj lists.
//
// ==============================================================

void init_adj_nodes(adj_node* projection, int n) {
  for (int i = 0; i < n; i++) {
    adj_node* prev[2] = {NULL, NULL};
    adj_node* start[2] = {&(projection[i * n + 0]), &(projection[0 * n + i])};

    for (int j = 0; j < n; j++) {
      adj_node* curr[2] = {&(projection[i * n + j]), &(projection[j * n + i])};
      for (int dir = 0; dir < 2; dir++) {
        curr[dir]->prev[dir] = NULL;
        curr[dir]->next[dir] = NULL;
        if (curr[dir]->count > 0) {
          if (prev[dir] == NULL) {  // No previous edge in row / column.
            if (j == 0) {           // Is the first edge, point at NULL.
              curr[dir]->prev[dir] = NULL;
            } else {  // Is not the first edge, point at first.
              curr[dir]->prev[dir] = start[dir];
            }
          } else {  // Point to previous edge in row / column.
            curr[dir]->prev[dir] = prev[dir];
          }
          if (curr[dir]->prev[dir] != NULL)  // If prev, set their next.
            curr[dir]->prev[dir]->next[dir] = curr[dir];
          // Update previous to the current edge.
          prev[dir] = curr[dir];
        }
      }
    }
    assert(prev[0] != NULL && prev[1] != NULL);
  }

  /*
  cout << "End of init" << endl;

  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      display_adj_node(&(projection[i*n + j]));
      assert_adj_node(&(projection[i*n + j]));
    }
  }
  */
}

typedef enum { WHITE, GRAY, BLACK } node_color;

typedef struct _node {
  node_color color;
  unsigned int discovery;
  unsigned int finish;

} node;

void dfs_visit(int i, int n, node* nodes, adj_node* M, int dir, vector<int>* new_order,
               vector<adj_node*>* to_remove, int& dfs_time, int dfs_tree_time) {
  dfs_time++;
  nodes[i].discovery = dfs_time;
  nodes[i].color = GRAY;

  int idx = (dir == 0 ? (i * n + 0) : (0 * n + i));
  adj_node* curr = &(M[idx]);
  if (curr->count <= 0)
    curr = curr->next[dir];

  assert(curr != NULL);

  while (curr != NULL) {
    // display_adj_node(curr);
    assert(curr->count > 0);
    int j = (dir == 0 ? curr->y : curr->x);
    // cout << j << endl;
    if (nodes[j].color == WHITE)
      dfs_visit(j, n, nodes, M, dir, new_order, to_remove, dfs_time, dfs_tree_time);
    if (to_remove != NULL && dir == 1 && nodes[j].color == BLACK &&
        nodes[j].finish <= dfs_tree_time)
      to_remove->push_back(curr);

    curr = curr->next[dir];
  }

  nodes[i].color = BLACK;
  dfs_time++;
  nodes[i].finish = dfs_time;

  if (new_order != NULL)
    new_order->push_back(i);
}

void dfs(int n, node* nodes, adj_node* M, vector<int>* order, vector<int>* new_order, int dir,
         vector<adj_node*>* to_remove) {
  int dfs_time = 0;
  bzero(nodes, n * sizeof(node));

  // dfs_init(n, nodes);

  // assert(order == NULL || order -> size() == n);

  for (int idx = n - 1; idx >= 0; idx--) {
    int i = idx;
    if (order != NULL)
      i = order->at(idx);
    if (nodes[i].color == WHITE) {
      // if (direction == 1)
      //   cout << "-- Tree start --" << endl;
      int dfs_tree_time = dfs_time;
      dfs_visit(i, n, nodes, M, dir, new_order, to_remove, dfs_time, dfs_tree_time);
      // if (direction == 1)
      //   cout << "-- Tree end --" << endl;
    }
  }
}

void allowed_matching_edges(adj_node* M, int n, vector<adj_node*>* to_remove) {
  node nodes[n];
  vector<int> new_order;

  // cout << "FORWARD" << endl;
  dfs(n, nodes, M, NULL, &new_order, 0, NULL);
  assert(new_order.size() == n);
  // cout << "BACKWARD" << endl;
  dfs(n, nodes, M, &new_order, NULL, 1, to_remove);
}
