/**
 *
 *  Author: Vu Le, Matt Anderson.
 */

#include "matching.h"
#include <vector>
#include "cuda_runtime.h"
#include <iostream>
#include <algorithm>
#include <TDM.h>
#include <Puz.h>
#include <chrono>
#include <fstream>
#include <cooperative_groups.h>
#include "TDM_cuda.h"

namespace cg = cooperative_groups;

#define HASH_CONST 1023 
// TODO: Prevent ranges from overflowing somehow.

#define MAX_S 500


__host__
void check_cuda_error(std::string c = "default"){
    cudaError_t err= cudaGetLastError();
    if (err != cudaSuccess){
        std::cout << c << '\n';
        printf("Cuda Error: %s \n", cudaGetErrorString(err));
        exit(-1);
    }
}

/** initialize projections directly from tdm, in parallel */
__global__ 
void calc_projection(int s, bool* dev_tdm, bool* dev_projections, int face){
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  if (x < s && y < s && z < s && dev_tdm[x * s * s + y * s + z]) {
    int i =
      (  (face == 0) * (y * s + z)
       + (face == 1) * (x * s + z)
       + (face == 2) * (x * s + y));
    dev_projections[i] = true;
  }
}

//__global__
__device__
void calc_graph(int u, int s, bool* d_projection, unsigned int &num_neigh, unsigned int* edges,
		unsigned int &num_neigh_r, unsigned int * edges_r){  
  num_neigh = 0;
  for (int i = 0; i < s; i++)
    if(i != u && d_projection[u * s + i]){
      num_neigh++;
      edges[(num_neigh - 1)] = i;
    }

  num_neigh_r = 0;
  for (int i = 0; i < s; i++)
    if(i != u && d_projection[i * s + u]) {
      num_neigh_r++;
      edges_r[(num_neigh_r - 1)] = i;
    }
}


// Direct calculation of adj list from TDM, too 
__device__
void calc_graph_tdm(int u, int s, bool* d_tdm, int face, unsigned int &num_neigh, unsigned int* edges,
		unsigned int &num_neigh_r, unsigned int * edges_r){

  num_neigh = 0;
  num_neigh_r = 0;
  
  if (face == 0) { // (y, z, x)

    int y = u;
    for (int z = 0; z < s; z++) {
      bool found = false;
      for (int x = 0; x < s && !found; x++) found = found || d_tdm[x * s * s + y * s + z];
      if(z != y && found){
	num_neigh++;
	edges[(num_neigh - 1)] = z;
      }
    }

    for (int z = 0; z < s; z++) {
      bool found = false;
      for (int x = 0; x < s && !found; x++) found = found || d_tdm[x * s * s + z * s + y];
      if(z != y && found){
	num_neigh_r++;
	edges_r[(num_neigh_r - 1)] = z;
      }
    }
    
  } else if (face == 1){  // (x, z, y)

    int x = u;
    for (int z = 0; z < s; z++) {
      bool found = false;
      for (int y = 0; y < s && !found; y++) found = found || d_tdm[x * s * s + y * s + z];
      if(z != x && found){
	num_neigh++;
	edges[(num_neigh - 1)] = z;
      }
    }

    for (int z = 0; z < s; z++) {
      bool found = false;
      for (int y = 0; y < s && !found; y++) found = found || d_tdm[z * s * s + y * s + x];
      if(z != x && found){
	num_neigh_r++;
	edges_r[(num_neigh_r - 1)] = z;
      }
    }
  } else { // (x, y, z)

    int x = u;
    for (int y = 0; y < s; y++) {
      bool found = false;
      for (int z = 0; z < s && !found; z++) found = found || d_tdm[x * s * s + y * s + z];
      if(x != y && found){
	num_neigh++;
	edges[(num_neigh - 1)] = y;
      }
    }

    for (int y = 0; y < s; y++) {
      bool found = false;
      for (int z = 0; z < s && !found; z++) found = found || d_tdm[y * s * s + x * s + z];
      if(x != y && found){
	num_neigh_r++;
	edges_r[(num_neigh_r - 1)] = y;
      }
    }
    

  }
}


/* allowed matching in cuda starts here */

/** 
 * Trim trivial components (leading or terminal SCC), perform on transposed graph for efficiency
 * Leading T_SCC ---> [SCC] ---> Terminal T_SCC
 * Condition explained: for leading tscc, check if no predecessors, for terminal tscc, check if no 
 * immediate predeccessors that site on the same subgraph
*/
__device__
void trim(int u, int s, unsigned int num_neigh_r, unsigned int* edges_r, int* ranges, bool* elim){
  
  auto g = cg::this_thread_block();

  __shared__ bool term; 
  // shared memory, shared among all threads in the block. term is a flag to terminate the loop

  if (g.thread_rank() == 0) term = false;
  g.sync();
  
  while (!term){
    g.sync();
    if (g.thread_rank() == 0) term = true;
    g.sync();
    
    if (!elim[u]){
      bool trimable = true;
      for (int i = 0; i < num_neigh_r && trimable; i++) {
	int v = edges_r[i];
	trimable = trimable && !(ranges[u] == ranges[v] && !elim[v] && u != v);
      }
      if (trimable){
	elim[u] = true;
	term = false;
      }
    }
    g.sync();
  }
}

__device__
void forward_reach(int u, int s, unsigned int num_neigh, unsigned int* edges,
		   bool* visited, int* ranges){

  auto g = cg::this_thread_block();

  __shared__ bool term; 
  // flag to terminate the loop
  if (g.thread_rank() == 0) term = false;
  g.sync();
  
  while (!term){
    g.sync();
    if (g.thread_rank() == 0) term = true;
    g.sync();
    
    if (visited[u]){
      for (int i = 0; i < num_neigh; i++){
	int v = edges[i];
	if (!visited[v] && ranges[u] == ranges[v]){
	  visited[v] = true;
	  term = false;
	}
      }
    }
    g.sync();
  }
}

__device__
void update(int u, int s, int* ranges, bool* visitedF, bool* visitedB,
	    bool* eliminated, unsigned int* pivots, unsigned int iter, bool* terminate){

  if (eliminated[u]){
    ranges[u] = -u;
    return;
  }
  else if (visitedF[u] && visitedB[u]){
    ranges[u] = ranges[u] * 4;
    return;
  }
  else if (visitedF[u]){
    ranges[u] = 4 * ranges[u] + 2;
  }
  else if (visitedB[u]){
    ranges[u] = 4 * ranges[u] + 3;
  }
  else {
    ranges[u] = 4 * ranges[u] + 1;
  }
  visitedF[u] = false; 
  // reset the visited flag
  visitedB[u] = false; 
  // reset the visited flag

  int hash = ranges[u] & HASH_CONST;
  if (pivots[hash] < iter && atomicMax(&pivots[hash], iter) < iter){
    visitedF[u] = true;
    visitedB[u] = true;
    *terminate = false;
  }
}


__global__
void modify_TDM(bool* dev_tdm, int s, int* dev_ranges, int face, bool* d_isChanged){
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  if (x < s && y < s && z < s){
    int idx = x * s * s + y * s + z;
    if (dev_tdm[idx]) {
      bool toDelete =
	(  (face == 0) && (dev_ranges[y] != dev_ranges[z])
	   || (face == 1) && (dev_ranges[x] != dev_ranges[z])
	   || (face == 2) && (dev_ranges[x] != dev_ranges[y]));

      dev_tdm[idx] = !toDelete;
      if (toDelete) *d_isChanged = true;
    }
  }
}


__device__
void zero_1d_uint(int u, unsigned int * buf, size_t len){
  
  auto gg = cg::this_grid();
  
  int stride = (len / (gg.size())) + 1;
  int idx = (stride * u);
  int end = (idx + stride <= len ? idx + stride : len);
  for (; idx < end; idx++)
    buf[idx] = 0;
}

__device__
void zero_1d_bool(int u, bool * buf, size_t len){
  
  auto gg = cg::this_grid();
  
  int stride = (len / (gg.size())) + 1;
  int idx = (stride * u);
  int end = (idx + stride <= len ? idx + stride : len);
  for (; idx < end; idx++)
    buf[idx] = 0;
  
}

__global__
void fb_kern(int s, bool * d_projection, int* d_ranges){
  
  auto g = cg::this_thread_block();
  int u = threadIdx.x;
  
  unsigned int num_neigh;
  unsigned int num_neigh_r;
  unsigned int edges[MAX_S];
  unsigned int edges_r[MAX_S];

  __shared__ int iter;
  __shared__ bool term;
  __shared__ unsigned int d_pivots[HASH_CONST+1]; 
  __shared__ bool d_visitedF[MAX_S];  // Per vertex, checked and set by other verts.
  __shared__ bool d_visitedB[MAX_S];  // Per vertex, checked and set by other verts.
  __shared__ bool d_eliminated[MAX_S];// Per vertex, checked by other verts.
  
  // Init pivots to 0 - probably more elegant way to do this.
  zero_1d_uint(u, d_pivots, (HASH_CONST+1));
  d_visitedF[u] = false;
  d_visitedB[u] = false;
  d_eliminated[u] = false;
  calc_graph(u, s, d_projection, num_neigh, edges, num_neigh_r, edges_r);
  
  if (g.thread_rank() == 0) {
    term = false;
    iter = 0;
  }
  g.sync();

  while (!term){
    g.sync();
    if (g.thread_rank() == 0) {
      term = true;
      iter++; 
      // increment the iteration counter
    }
    g.sync();

    forward_reach(u, s, num_neigh, edges, d_visitedF, d_ranges);
    forward_reach(u, s, num_neigh_r, edges_r, d_visitedB, d_ranges);
    trim(u, s, num_neigh_r, edges_r, d_ranges, d_eliminated);    
    update(u, s, d_ranges, d_visitedF, d_visitedB,
	   d_eliminated, d_pivots, iter, &term);
    g.sync();
  }
}


__global__
void sum_y(unsigned int * ress, unsigned int * ress2, unsigned int s){
  int r1 = blockDim.x * blockIdx.x + threadIdx.x;
  if (r1 < s) {
    unsigned int res = 0;
    for (int r2 = 0; r2 < s; r2++)
      res += ress2[r1*s + r2];
    ress[r1] = res;
  }
}

__global__
void sum_z(unsigned int * ress, bool * tdm, unsigned int s){
  int r1 = blockDim.x * blockIdx.x + threadIdx.x;
  int r2 = blockDim.y * blockIdx.y + threadIdx.y;
  if (r1 < s && r2 < s) {
    unsigned int res = 0;
    for (int r3 = 0; r3 < s; r3++)
      res += (tdm[r1*s*s + r2*s + r3] == true);
    ress[r1 * s + r2] = res;
  }
}

__device__
bool isWitness(const char * cstr, unsigned int k, unsigned int r1,
	       unsigned int r2, unsigned int r3, bool strong){

  for (unsigned int c = 0; c < k; c++){
    unsigned int count = ((cstr[(k+1)*r1 + c] == '1')
			  + (cstr[(k+1)*r2 + c] == '2')
			  + (cstr[(k+1)*r3 + c] == '3'));
    if ((strong && count == 2) || (!strong && count >= 2))
      return true;
  }  
  return false;
}

__global__
void compute_TDM(char * cstr, bool * tdm, unsigned int s, unsigned int k, bool strong){
  int r1 = blockDim.x * blockIdx.x + threadIdx.x;
  int r2 = blockDim.y * blockIdx.y + threadIdx.y;
  int r3 = blockDim.z * blockIdx.z + threadIdx.z;
  if (r1 < s && r2 < s && r3 < s)
    tdm[r1 * s * s + r2 * s + r3] = !isWitness(cstr, k, r1, r2, r3, strong);
}

  
int simplify_cuda_inner(unsigned int s, bool * tdm, bool * d_tdm){

  assert(s <= MAX_S);
  assert(d_tdm != NULL);

  // Allocate in bulk.  Globally device allocated array for things
  // used in different kernels or that would not fit in shared memory.
  // Bools have be allocated last to avoid issues with byte alignment.
  char * d_buf;
  cudaMalloc(&d_buf, (sizeof(int) * (s)
		      + sizeof(bool) * (1 + s * s)
		      + sizeof(unsigned int) * (s + (s * s))));
    size_t offset = 0;

  unsigned int *d_res1d = (unsigned int *)(d_buf + offset);  offset += sizeof(unsigned int) * s;
  unsigned int *d_res2d = (unsigned int *)(d_buf + offset);  offset += sizeof(unsigned int) * s * s;
  int *d_ranges = (int *)(d_buf + offset);  offset += sizeof(int) * s;
  bool *d_isChanged = (bool *)(d_buf + offset);  offset += sizeof(bool);
  bool *d_projection = (bool *)(d_buf + offset);  offset += sizeof(bool) * s * s;
  
  int no_change = 0;
  for (int iter = 0; no_change < 3; iter++){
    int face = iter % 3;

    // Mem-zero d_ranges, d_isChanged, d_projection en masse.
    cudaMemsetAsync(d_ranges, 0, sizeof(int) * s + sizeof(bool) * (1 + s * s));

    check_cuda_error("start init");
    
    dim3 threadsPerBlock1(8, 8, 8);
    int temp_b = ceil(s * 1.0 / 8);
    dim3 blocksPerGrid1(temp_b, temp_b, temp_b);
    calc_projection<<<blocksPerGrid1, threadsPerBlock1>>>(s, d_tdm, d_projection, face);
    check_cuda_error("after calc_projection");
    
    bool isChanged = false;
    fb_kern<<<1, s>>>(s, d_projection, d_ranges);
    check_cuda_error("after fb");
    
    dim3 threadsPerBlock(8, 8, 8);
    int temp = ceil(s * 1.0 / 8);
    dim3 blocksPerGrid(temp, temp, temp);
    modify_TDM<<<blocksPerGrid, threadsPerBlock>>>(d_tdm, s, d_ranges, face, d_isChanged);
    cudaMemcpy(&isChanged, d_isChanged, sizeof(bool), cudaMemcpyDeviceToHost);

    if (!isChanged)
      no_change++;
    else 
      no_change = 0;
  }

  // Copy back to tdm if one provided by caller.
  if (tdm != NULL)
    cudaMemcpy(tdm, d_tdm, sizeof(bool) * s * s * s, cudaMemcpyDeviceToHost);

  // Calculate sum of tdm.  Not the most elegant or efficient, but not
  // the bottle neck.
  dim3 threadsPerBlock2d(16, 16);
  int temp = ceil(s * 1.0 / 16);
  dim3 blocksPerGrid2d(temp, temp);
  sum_z<<<blocksPerGrid2d, threadsPerBlock2d>>>(d_res2d, d_tdm, s);

  dim3 threadsPerBlock1d(32);
  temp = ceil(s * 1.0 / 32);
  dim3 blocksPerGrid1d(temp);
  sum_y<<<blocksPerGrid1d, threadsPerBlock1d>>>(d_res1d, d_res2d, s);

  unsigned int res1d[s];
  cudaMemcpy(res1d, d_res1d, sizeof(unsigned int) * s, cudaMemcpyDeviceToHost);

  // Final calculate occurs on CPU because its simpler.
  int result = 0;
  for (int r1 = 0; r1 < s; r1++)
    result += res1d[r1];

  cudaFree(d_buf);
  check_cuda_error("after dealloc");
  
  return result;
}


int simplify_cuda_inner(const Puz &p, bool * tdm, bool strong){

  check_cuda_error("Begin cuda_inner(p)");
  
  unsigned int s = p.getHeight();
  unsigned int k = p.getWidth();

  // Copy puzzle string to device.
  std::string str = p.toString();
  const char *cstr = str.c_str();
  assert(strlen(cstr) == str.length());

  char * dev_cstr;
  bool * dev_tdm;
  cudaMalloc(&dev_cstr, sizeof(char) * s * (k+1));
  cudaMemcpyAsync(dev_cstr, cstr, sizeof(char) * s * (k+1), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_tdm, sizeof(bool) * s * s * s);
  check_cuda_error("Before compute_TDM");

  // Compute the 3DM instance in parallel on device.
  dim3 threadsPerBlock(8, 8, 8);
  int temp = ceil(s * 1.0 / 8);
  dim3 blocksPerGrid(temp, temp, temp);
  compute_TDM<<<blocksPerGrid, threadsPerBlock>>>(dev_cstr, dev_tdm, s, k, strong);
  check_cuda_error("After compute_TDM");

  // Compute simplification on device.
  int result = simplify_cuda_inner(s, tdm, dev_tdm);
  
  cudaFree(dev_cstr);
  cudaFree(dev_tdm);
  
  return result;
}
