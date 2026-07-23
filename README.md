# Matrix Multiplication Search: CUDA-Accelerated Strong USP Discovery

This repository searches for large Strong Uniquely Solvable Puzzles (SUSPs),
combinatorial objects over `{1, 2, 3}` that connect to fast matrix
multiplication constructions. Practically, the code generates candidate
`(s, k)` puzzles, converts each candidate into a 3D matching instance, simplifies
that instance, and uses the remaining edge count as a fitness signal for search.

The public snapshot is centered on GPU systems work. The original path is
`TDM_cuda.cu` plus `search_ils_staged.cpp`; the enhanced path is
`TDM_cuda_enhanced.cu`, `GpuTask.*`, `ils_stage.cpp`, `search.cpp`, and
`search_ils_staged_enhance.cpp`.

## What This Repo Demonstrates

- CUDA kernel design for an irregular combinatorial workload.
- Packed puzzle representations and bit-level witness tests.
- Batched GPU execution with pinned memory, streams, and reusable device buffers.
- OpenMP CPU search workers feeding asynchronous GPU coordinator threads.
- Multi-GPU scheduling, launch coalescing, checkpoint/resume, and optional Bloom
  filtering for long-running research searches.

## Core Idea

For an `(s, k)` puzzle, each row is a length-`k` string over `{1, 2, 3}`. The
verification code builds a 3D matching table over all row triples. If the table
can be simplified down to the right structure, the puzzle is a USP/SUSP. Larger
valid puzzles are more interesting because they imply stronger matrix
multiplication constructions.

## Papers

- Original research background:
  [Efficiently-Verifiable Strong Uniquely Solvable Puzzles and Matrix Multiplication](paper/2307.06463v1.pdf)
  by Matthew Anderson and Vu Le introduces simplifiable SUSPs, efficient
  verification through 3D matching simplification, and iterative local search for
  stronger matrix multiplication exponent bounds.
- GPU pipeline paper:
  [GPU Pipeline for Accelerated Search of an Upper Bound on the Matrix Multiplication Exponent](<GPU Pipeline for Accelerated Search of an Upper Bound on the Matrix Multiplication Exponent.pdf>)
  documents the enhanced implementation in this repository: batched CUDA
  simplification, an asynchronous CPU/GPU producer-consumer pipeline, pinned
  buffers, launch coalescing, and multi-GPU execution. The paper reports up to
  5.75x end-to-end throughput speedup and the discovery of a `(79, 10)`-SUSP,
  improving on the baseline `(78, 10)` result.

## CUDA Implementation Comparison

### Baseline: `workspace/src/backend/TDM_cuda.cu`

`TDM_cuda.cu` is the original GPU simplifier. It accelerates one puzzle at a
time:

- Copies a puzzle string to the GPU.
- Builds a boolean `s * s * s` TDM table with `compute_TDM`.
- Repeatedly projects the TDM onto one cube face, computes graph ranges with
  `fb_kern`, deletes inconsistent TDM entries with `modify_TDM`, and stops after
  three unchanged faces.
- Sums the simplified TDM with `sum_z`, `sum_y`, and a final CPU accumulation.

This version is useful as a correctness-oriented CUDA baseline: it keeps the
algorithm close to the CPU simplifier and uses simple launch geometry.

### Enhanced: `workspace/src/backend/TDM_cuda_enhanced.cu`

`TDM_cuda_enhanced.cu` turns the one-puzzle simplifier into a batched GPU
engine. The main entry point is still `outer_Kernel`, but it now operates over
many trials at once through a reusable `GpuResources` object.

| Area | `TDM_cuda.cu` | `TDM_cuda_enhanced.cu` |
| --- | --- | --- |
| Work granularity | One puzzle per simplify call. | Many puzzles per launch, defaulting to `TRIALS = 256` and supporting larger coalesced batches. |
| Input format | Converts `Puz` to a character string before copying. | Uses packed `Puz` row data directly as `e_type` values. |
| Host memory | Regular host allocations and per-call setup. | Pinned host buffers via `cudaMallocHost` for faster async transfer. |
| Device memory | Allocates temporary device memory inside each simplify call. | Allocates one aligned contiguous device buffer per `GpuResources` instance and reuses it. |
| Stream model | Mostly synchronous single-call flow. | Uses a CUDA stream per GPU worker for async copies, memset, kernels, and result transfer. |
| TDM storage | `bool` TDM table. | `uint8_t` TDM table, sized for batched trials. |
| Small-`k` witness test | Character comparisons across columns. | `isWitness16_Kernel_u32` uses bit masks and packed 2-bit entries for `k <= 16`. |
| TDM construction | Generic `8 x 8 x 8` thread blocks over row triples. | Small-`k` cached kernel stores packed rows in shared memory; general fallback handles wider puzzles. |
| Projection | One projection kernel over `x, y, z`. | Face-specific projection kernel over `s * s` cells per trial, skipping completed trials with `no_change`. |
| SCC/range step | One block with cooperative groups and shared arrays. | Warp-level transitive closure for `s <= 32`, shared-memory bitset Floyd-Warshall for larger `s`. |
| TDM modification | One generic 3D delete kernel. | Multi-block-per-trial delete kernels, plus a face-2 fast path that clears contiguous `z` lines. |
| Convergence | Host copies one change flag each iteration. | Per-trial `no_change` and `active_count` keep completed trials out of later iterations. |
| Reduction | Two custom sum kernels plus CPU final sum. | CUB `BlockReduce` with per-trial atomic accumulation. |
| Launch sizing | Fixed/simple geometry. | Uses SM count and occupancy to round launches toward fuller waves. |

The enhanced kernel path is specifically optimized for the searcher's real
workload: evaluate many same-sized candidate puzzles, minimize allocation churn,
reduce host/device synchronization, and keep the GPU fed.

## Searcher Comparison

### Baseline: `workspace/src/searchers/search_ils_staged.cpp`

The original staged ILS searcher is a monolithic OpenMP program:

- Maintains several ILS stages in one source file.
- Uses priority queues to expand candidate frontiers.
- Evaluates row replacements, derivatives, and symbol permutations.
- Uses the original TDM path through normal fitness checks.
- Supports starting from a `.puz` file.
- Logs found puzzles, but has no checkpoint/resume path for long runs.

This is a good reference implementation of the staged search idea.

### Enhanced: `workspace/src/searchers/search_ils_staged_enhance.cpp`

The enhanced searcher keeps the same ILS algorithmic structure but adds a GPU
runtime around it:

- Moves reusable search state into `Search`, `ILSStage`, and `stats_t` so it can
  be serialized.
- Adds `GpuTask` slots with pinned input/output buffers and explicit states:
  `Empty`, `Filling`, `Ready`, `Processing`, and `Done`.
- Gives each OpenMP worker a pool of reusable GPU task slots.
- Uses `GpuTaskIdQueue` and GPU coordinator threads to move ready batches from
  CPU workers to CUDA devices.
- Coalesces multiple task slots into larger GPU launches when dimensions match.
- Supports multiple CUDA devices and multiple coordinator lanes per device.
- Streams row-replacement candidates directly into GPU slots, avoiding a large
  temporary vector for the `3^k` replacement frontier.
- Adds `.dat` checkpoint/resume, auto-start from matching checkpoints in `data/`,
  and SIGINT-triggered checkpoint before exit.
- Adds optional Bloom filtering for the isomorph cache with serialization.
- Reports live CPU/GPU simplify counts and per-device GPU launch statistics.

The enhanced flow is:

```text
OpenMP worker
  -> generate candidate puzzles
  -> fill pinned GpuTask slot
  -> enqueue slot id
  -> GPU coordinator coalesces slots
  -> outer_KernelDeviceInput runs CUDA batch
  -> results copied back to pinned buffers
  -> worker updates ILS frontier
```

This is the main engineering upgrade in the repo: it changes the GPU from an
occasional per-puzzle accelerator into a batched service for the search.

## Build

Requirements:

- CMake 3.28+
- C++17 compiler
- CUDA Toolkit
- OpenMP
- zlib
- Prebuilt `workspace/lib/MapleCOMSPS.a` and `workspace/lib/nauty.a`, or rebuilt
  equivalents available at those paths

Build from the repository root:

```bash
cd workspace
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake writes executables to the top-level `bin/` directory.

## Run

From the repository root:

Baseline search:

```bash
./bin/search_ils_staged 1 1 14 6
```

Enhanced GPU search:

```bash
./bin/search_ils_staged_enhance 1 1 14 6
```

More configurations to test:

```bash
./bin/search_ils_staged_enhance 1 1 40 8
./bin/search_ils_staged_enhance 1 1 60 9
./bin/search_ils_staged_enhance 1 1 100 10
```

Enhanced search with Bloom filtering:

```bash
./bin/search_ils_staged_enhance 1 1 14 6 --bloom=100000000,0.01
```

Resume from a checkpoint:

```bash
./bin/search_ils_staged_enhance 1 1 14 6 data/checkpoint_YYYYMMDD_HHMMSS.dat
```

Argument meanings:

- `mode = 0`: full check
- `mode = 1`: simplifiable/obvious check
- `mode = 2`: local check
- `strong = 1`: search for Strong USPs
- `s_target`: target row count
- `k`: puzzle width

## GPU Runtime Knobs

The enhanced searcher exposes several environment variables for tuning:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ILS_GPU_DEVICES` | all devices | Number of CUDA devices to use. `ILS_GPU_WORKERS` is accepted as a legacy alias. |
| `ILS_GPU_LANES_PER_DEVICE` | `2` | GPU coordinator threads per selected CUDA device. |
| `ILS_GPU_BATCH_SLOTS` | `32` | Requested number of `GpuTask` slots to coalesce per CUDA launch. |
| `ILS_GPU_MAX_BATCH_MB` | `512` | Memory cap used to choose maximum puzzles per coalesced launch. |
| `ILS_GPU_SLOTS_PER_WORKER` | `32` | Reusable pinned task slots owned by each OpenMP worker. |
| `ILS_GPU_COALESCE_US` | `5000` | How long a coordinator waits for more compatible work before launching. |
| `ILS_GPU_MOD_BLOCKS_PER_TRIAL` | size-dependent | Blocks per trial for TDM modification kernels, capped at 16. |
| `ILS_GPU_STREAM_ROW_REPLACEMENTS` | enabled | Stream `3^k` row replacements directly into GPU slots. |
| `ILS_GPU_DONE_DRAIN_LIMIT` | `4` | Limit on opportunistic completed-slot draining. |
| `ILS_AUTO_CHECKPOINT_SECONDS` | `21600` | Auto-checkpoint interval. Set to `0` to disable. |

Example:

```bash
ILS_GPU_DEVICES=2 \
ILS_GPU_LANES_PER_DEVICE=2 \
ILS_GPU_BATCH_SLOTS=64 \
ILS_GPU_MAX_BATCH_MB=1024 \
./bin/search_ils_staged_enhance 1 1 14 6 --bloom=100000000,0.01
```

## Repository Layout

```text
.
|-- README.md
|-- bin/                         # Built search executables
|-- data/                        # Checkpoints are written here
|-- paper/                       # Research background material
`-- workspace/
    |-- CMakeLists.txt
    |-- lib/                     # Imported static libraries
    |-- src/
    |   |-- backend/
    |   |   |-- TDM.cpp          # CPU TDM construction/simplification
    |   |   |-- TDM_cuda.cu      # Baseline CUDA simplifier
    |   |   |-- TDM_cuda_enhanced.cu
    |   |   |-- GpuTask.cpp      # Pinned task slots and GPU task state
    |   |   |-- ils_stage.cpp    # Enhanced ILS stage implementation
    |   |   |-- search.cpp       # Serializable search state
    |   |   `-- includes/
    |   `-- searchers/
    |       |-- search_ils_staged.cpp
    |       `-- search_ils_staged_enhance.cpp
    `-- thirdparty/
```

## Notes

This repository originates from matrix multiplication research code and has been
adapted here to highlight GPU kernel and systems engineering. The most important
files for review are the enhanced CUDA simplifier, the `GpuTask` pipeline, and
the enhanced staged ILS searcher.
