#ifndef __MATCHING_H__
#define __MATCHING_H__

#include <vector>
#include <iostream>
#include <assert.h>
using namespace std;

typedef struct _adj_node {
  int count;
  int x, y;
  struct _adj_node* prev[2];
  struct _adj_node* next[2];

} adj_node;

inline void assert_adj_node(adj_node* u) {
  if (u != NULL) {
    if (u->x == 0) {
      assert(u->next[1] != NULL || u->count > 0);
    }
    if (u->y == 0) {
      assert(u->next[0] != NULL || u->count > 0);
    }

    if (u->count == 0 && u->x > 0 && u->y > 0) {
      assert(u->next[0] == NULL && u->next[1] == NULL && u->prev[0] == NULL && u->prev[1] == NULL);
    }

    for (int dir = 0; dir < 2; dir++) {
      assert(u->prev[dir] == NULL || u->prev[dir]->next[dir] == u);
      assert(u->next[dir] == NULL || u->next[dir]->prev[dir] == u);
    }

    assert(u->prev[0] == NULL || u->prev[0]->x == u->x);
    assert(u->prev[0] == NULL || u->prev[0]->y < u->y);

    assert(u->prev[1] == NULL || u->prev[1]->x < u->x);
    assert(u->prev[1] == NULL || u->prev[1]->y == u->y);

    assert(u->next[0] == NULL || u->next[0]->x == u->x);
    assert(u->next[0] == NULL || u->next[0]->y > u->y);

    assert(u->next[1] == NULL || u->next[1]->x > u->x);
    assert(u->next[1] == NULL || u->next[1]->y == u->y);
  }
}

inline void display_adj_node(adj_node* u) {
  if (u != NULL) {
    cout << "addr: " << u;
    cout << " count: " << u->count;
    cout << " x: " << u->x;
    cout << " y: " << u->y;
    cout << " prev: " << u->prev[0] << ", " << u->prev[1];
    cout << " next: " << u->next[0] << ", " << u->next[1];
    cout << endl;

    // assert_adj_node(u);
  }
}

inline void display_adj_nodes(adj_node* M, int n, bool verbose) {
  for (int x = 0; x < n; x++) {
    for (int y = 0; y < n; y++) {
      adj_node* curr = &(M[x * n + y]);
      if (verbose) {
        printf(" (%3d) %p: {%p, %p} <%p, %p>  |", curr->count, curr, curr->prev[0], curr->prev[1],
               curr->next[0], curr->next[1]);
      } else {
        printf(" %3d", curr->count);
      }
    }
    cout << endl;
  }
  cout << endl;
}

inline void dec_adj_node(adj_node* u) {
  u->count--;
  if (u->count == 0) {
    // cout << "Deleting: " << u << endl;
    assert(u->x != u->y);

    // cout << "before: ";
    // display_adj_node(u);

    for (int dir = 0; dir < 2; dir++) {
      // Only remove next if not on the outer edge.
      if ((dir == 1 && u->x > 0) || (dir == 0 && u->y > 0)) {
        if (u->prev[dir] != NULL)
          u->prev[dir]->next[dir] = u->next[dir];
        if (u->next[dir] != NULL)
          u->next[dir]->prev[dir] = u->prev[dir];

        u->prev[dir] = NULL;
        u->next[dir] = NULL;
      }
    }
    // cout << "after: ";
    // display_adj_node(u);
  }

  assert_adj_node(u);
}

inline void del_adj_node(adj_node* u) {
  u->count = 1;
  dec_adj_node(u);
}

void init_adj_nodes(adj_node* projection, int n);

// Implement Bipartite matchin using Ford-Faulkerson flow algorithm.
bool has_perfect_bipartite_matching(bool* M, int n);

void allowed_matching_edges(adj_node* M, int n, vector<adj_node*>* to_remove);

void print_G(bool* G, int n);

#endif
