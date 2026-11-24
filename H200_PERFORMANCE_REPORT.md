# Size-specific GPU batching on two H200s

Grouping ready GPU slots by puzzle dimensions increased live
`(51,9) → (52,9)` search throughput from **488,380 to 511,067
simplifications/s** at 64 OpenMP threads on `hpcc03` (+4.6%). In a separate
run that reached stage 52 from `(48,9)`, replaying the same stage-52 checkpoint
increased live throughput by 5.4%. A fixed-input replay with four interleaved
sizes improved by 54.5%. The broader live measurements include candidate
generation, scheduling, GPU execution, and ILS frontier updates.

## System and measurement

The node has two 32-core CPU sockets with two hardware threads per core
(64 physical cores and 128 logical CPUs) and two NVIDIA H200 NVL GPUs. The
64-thread runs used one logical CPU per physical core. The 128-thread runs used
both SMT siblings. Search used two CUDA coordinator lanes per GPU. Each lane
has its own stream and reusable device memory. The coordinator lanes take work
from the same queues.

The A/B reference is the enhanced searcher before the host scheduling and
slot-handling changes. Both builds use the shared-memory bitset
Floyd–Warshall kernel for the SCC/range step at these puzzle heights.
[`8d17638`](https://github.com/kaitophuc/matrix-multiplication-search/commit/8d17638118764df07135855d115bdf5ae6d5da24).

Live throughput is the change in the terminal's cumulative `Tot Simplify`
counter divided by elapsed wall time. Stage 52 can submit candidates for
several active sizes, so the stage-52 rate counts all simplifications in that
interval. Live runs can finish early when they find a solution. Comparisons
use common time windows, with shorter windows identified in the results.
Average logical CPUs used is process CPU time divided by wall time.

## Slot assembly and memory

Previously each coordinator lane had a queue of ready slot IDs. Workers
submitted these IDs round-robin, and coordinators could also steal work.
A coordinator grouped consecutive slots with compatible dimensions and
deferred the first incompatible slot. Interleaving sizes could therefore
split a group even when compatible slots were already queued. The
coordinator also created a temporary vector for each group of slot IDs,
and row-replacement producers materialized candidates before filling slots.

The new [scheduler](workspace/src/backend/GpuBatchQueue.cpp) keeps a bounded
ring of slot IDs per active `(s,k)`. All lanes share these rings. Under a
mutex, a lane takes a homogeneous group and advances a round-robin cursor
among eligible queues. A full group is immediately eligible. A partial group
becomes eligible once its oldest slot reaches the configured coalescing
age, or immediately during retirement or shutdown. The 5 ms setting governs
eligibility. Actual queue residence can be longer when all lanes are busy.
The rings allocate storage when a size opens, each lane reuses its group
vector, and the row-replacement path streams candidates into reusable slots.

Workers fill pinned input buffers in the slot pool. The coordinator copies
packed data once per slot to the corresponding region of a contiguous device
input buffer. After simplification, the coordinator transfers the group's
scores to a host staging buffer and copies each slot's 4-byte scores into its
output buffer. A slot follows `Empty → Filling → Ready → Processing → Done →
Empty`. The coordinator records completion before publishing `Done`, so the
search control thread sees consistent accounting at a stage transition.

The pinned buffers and per-lane device allocations were already persistent
in the reference implementation. The initial CUDA trace showed no runtime
CUDA allocations in the sampled steady-state interval. On that run the
worker slot pool used about 210 MiB of pinned memory and each lane used about
492 MiB of device memory at the target dimensions. The input path issued
12,916 host-to-device copies in a seven-second sample, with about 78 ms of
aggregate on-device copy time. The new grouping reduces the number of GPU
groups for a given candidate count while keeping the slot-level input copy.

The measured configuration used 256 candidates per slot, 32 slots per
worker, two lanes per GPU, and a 512 MiB per-lane batch memory budget. The
requested group cap was 32 slots. At target `(52,9)` the memory budget reduced
the effective cap to 14 slots, or **3,584 candidates per group**.

## Stage retirement and correctness

Starting from `(48,9)` with target `(52,9)` opened queues for heights
48–51. At the transition to stage 52, the search control thread retired
`(48,9)` and opened `(52,9)`, leaving heights 49–52 active. The retired
queue recorded 4,389 submitted and 4,389 completed slots, 1,121,931 processed
candidates, and zero pending or in-flight slots. A stage-52 checkpoint from a
repeat run resumed with queues for heights 49–52. The queues are rebuilt
from active stages when a checkpoint loads.

The live search from height 48 reached a `(52,9)` solution that passed
independent CPU verification. Both starting seeds were verified on the CPU
as well. Across the full instrumented run, the two GPUs processed
53,877,961 candidates in 17,105 groups, averaging **3,149.8 candidates/group**
(87.9% of the 3,584-candidate cap). The two device averages were 3,154.4
and 3,145.3.

The [queue test](workspace/tests/gpu_batch_queue_test.cpp) exercises
retirement while work is pending and in flight, partial-group deadlines,
round-robin fairness, ring wraparound, and delivery of each submitted slot
exactly once to a compatible group under concurrent producers and consumers.
The CPU-only test passed 200 standalone repetitions and the final HPC build
passed 50 consecutive CTest runs. Both staged search binaries built on the
HPC node.

## Fixed-input replay and live checkpoint

Each replay used 393,216 candidates preloaded on the host, with identical
inputs and slot order for both binaries. With three runs per scheduler and
size mixture, all score arrays matched byte for byte. Twenty-nine
representative candidates per mixture also matched independent CPU scores.

| Interleaved sizes | Reference candidates/s | Size-specific candidates/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 465,667 | 468,061 | +0.5% |
| 2 | 345,970 | 470,440 | +36.0% |
| 4 | 317,768 | 490,999 | +54.5% |

The replay supplies a backlog and measures the scheduling effect with
controlled inputs. For an end-to-end comparison with four active stages,
both binaries resumed the same checkpoint from the live `(48,9) → (52,9)`
search. With 64 OpenMP threads and two H200s, the 5–40 s stage-52 window
gave the following rates.

| Scheduler | Run 1 | Run 2 | Run 3 | Median simplify/s |
| --- | ---: | ---: | ---: | ---: |
| Reference | 515,938 | 511,379 | 532,023 | 515,938 |
| Size-specific | 543,894 | 563,157 | 539,355 | 543,894 |

The live median gain was **5.4%**. A separate seven-second trace at the same
checkpoint measured an increase from 1,377.6 to 3,271.4 candidates/group.
Height 52 accounted for 74.5% and 74.7% of the candidates, respectively.
Convergence-counter reads fell from 14,591 to 6,763 per million candidates
in that sample.

## Repeat of the `(51,9) → (52,9)` benchmark

This benchmark uses the original performance-test workload and its
`(51,9)` seed. The main window covers 12–27 s of stage 52. The 48-thread
comparison uses 7–12 s because runs finished before the main window closed.
The 32- and 48-thread reference values come from the previous benchmark.
The remaining pairs used reference and modified binaries from the same
benchmark campaign.

| OpenMP threads | H200s | Window | Reference simplify/s | Size-specific simplify/s | Change | Avg. logical CPUs used, size-specific |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 | 2 | 12–27 s | 313,169 | 323,234 | +3.2% | 30.0 |
| 48 | 2 | 7–12 s | 437,138 | 434,320 | −0.6% | 40.0 |
| 64 | 2 | 12–27 s | 488,380 | **511,067** | **+4.6%** | 47.2 |
| 128 | 2 | 12–27 s | 451,682 | 453,802 | +0.5% | 42.8 |
| 64 | 1 | 12–27 s | 264,302 | 266,720 | +0.9% | 25.4 |

The reported values are medians. Both 64-thread, two-GPU runs completed the
main window per scheduler. The modified runs produced 510,625–511,509
simplifications/s, up from 487,210–489,551. The one-GPU comparison has one
complete main-window sample per scheduler. In the later 27–34 s window,
the 64-thread two-GPU median increased from 482,648 to 519,808 (+7.7%).
At 64 threads the second GPU raised main-window throughput by about 1.92×
in the modified build.

### CPU utilization and backpressure

One diagnostic run per build and thread count instrumented frontier
publication, worker slot waits, and coordinator activity. Wait times sum
over all workers, so they can exceed one thread-second per wall second.

| Threads | Metric | Reference | Size-specific |
| ---: | --- | ---: | ---: |
| 64 | Frontier publication lock wait, thread-s/s | 10.1 | 9.1 |
| 64 | Worker slot wait, thread-s/s | 7.4 | 6.9 |
| 64 | Median ready queue depth, slots | 187 | 108 |
| 64 | Mean candidates per group | 1,444.6 | 2,892.2 |
| 128 | Frontier publication lock wait, thread-s/s | 75.7 | 78.3 |
| 128 | Worker slot wait, thread-s/s | 2.3 | 2.2 |
| 128 | Median ready queue depth, slots | 16 | 9 |
| 128 | Mean candidates per group | 1,641.2 | 2,478.8 |

The 64-thread throughput runs used 46.1 logical CPUs on average before the
change and 47.2 after it. In the diagnostic run the scheduler assembled
larger groups with fewer slots waiting. At 128 threads the dominant measured
worker wait was acquiring the lock to publish accepted results into the ILS
frontier. The GPU queue was shallower than at 64 threads, and coordinators
spent 17.2% of lane time waiting or assembling groups, compared with 8.1%
at 64 threads. A separate baseline CPU sample attributed about 75% of on-CPU OpenMP
worker samples to `Puz::isWitness`.

### GPU utilization and launch groups

In the unprofiled benchmark's 12–27 s window, the two GPUs in the modified
64-thread runs had about 92–99% sampled busy time. Both devices reached
100% sampled busy time in the 27–34 s window. This metric reports whether
kernels are executing over the sampling interval. SM occupancy and achieved
DRAM bandwidth require hardware counters, which were restricted on this
node. The separate seven-second Nsight Systems samples showed at least one
kernel running about 92–93% of wall time on each GPU in both builds.

The following counts use only complete launch groups in those traces.

| Candidate subset | Scheduler | Groups | Candidates | Mean/group | Max/group |
| --- | --- | ---: | ---: | ---: | ---: |
| All sizes | Reference | 2,206 | 3,250,313 | 1,473.4 | 3,584 |
| All sizes | Size-specific | 1,209 | 3,292,978 | 2,723.7 | 3,584 |
| `(52,9)` | Reference | 1,324 | 2,513,082 | 1,898.1 | 3,584 |
| `(52,9)` | Size-specific | 777 | 2,367,621 | 3,047.1 | 3,584 |

During each group, the host repeatedly reads a 4-byte convergence counter.
At height 52, the rate of these device-to-host reads fell from 11,125 to
7,969 per million candidates (−28.4%). The device-side copy itself took
about 1.7 µs. Aggregate host CUDA API time for these reads changed from
6.56 to 6.76 µs per height-52 candidate, summed across lanes. For the full
mix it changed from 7.10 to 7.04 µs per candidate. The API timing includes
waiting for earlier stream work and a lower read count alone does not
establish a reduction in end-to-end synchronization time.

In the modified trace, TDM modification accounted for 47.9% of summed
kernel duration, projection for 31.3%, and the bitset Floyd–Warshall step
for 8.9%. The respective reference shares were 47.5%, 30.4%, and 10.5%.
These duration shares include kernels on both devices and across streams.

## Configuration sweep

The sweep covered 256–2,048 MiB batch budgets in replay and 384–2,048 MiB
in live search, one to four lanes per GPU in replay and two to four in live
search, coalescing waits of 0, 250, 1,000, and 5,000 µs, and 8–64 slots per
worker. Slot capacities of 128, 256, 512, and 1,024 candidates were compared
with a fixed per-worker pinned-memory budget. Two additional experiments
prioritized fuller ready queues with an age limit and used targeted worker
completion notifications.

A 2,048 MiB budget reached roughly 512,000 candidates/s in four-size replay,
up from 491,000 at 512 MiB, but reached about 525,000 simplify/s in its live
sample compared with about 542,000 at 512 MiB. The other alternatives did
not show a repeatable gain over the retained configuration in live search.
The branch retains two lanes per GPU, 256 candidates per slot, 32 slots per
worker, a 512 MiB batch budget, a 5 ms age threshold, and round-robin
selection. The original 10% live throughput target was not reached.

## Reproducing the live tests

Build in a CUDA environment after satisfying the repository's build
dependencies. Build the reference and modified revisions in separate
checkouts so each build has its own `bin/` directory. The
`gpu_batch_queue_test` target and CTest command apply to the modified
revision. See [Build](README.md#build).

```bash
cmake -S workspace -B workspace/build -DCMAKE_BUILD_TYPE=Release
cmake --build workspace/build --target search_ils_staged_enhance gpu_batch_queue_test -j 16
ctest --test-dir workspace/build --output-on-failure --repeat until-fail:50
```

On the HPC cluster, the tests reserved the two H200s and all node CPUs.
Check the allocated CPU IDs and topology before using the affinity mask.
On this allocation, CPUs 0–63 selected one hardware thread per physical
core.

```bash
srun -p gpu -w hpcc03 -N 1 -n 1 -c 128 --gres=gpu:nvidia_h200_nvl:2 \
  --mem=0 -t 00:30:00 --pty bash
lscpu -e=CPU,CORE,SOCKET,NODE

export OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=FALSE
export ILS_GPU_DEVICES=2 ILS_GPU_LANES_PER_DEVICE=2
export ILS_GPU_BATCH_SLOTS=32 ILS_GPU_SLOTS_PER_WORKER=32
export ILS_GPU_MAX_BATCH_MB=512 ILS_GPU_COALESCE_US=5000
export ILS_AUTO_CHECKPOINT_SECONDS=0
taskset -c 0-63 ./bin/search_ils_staged_enhance 1 1 52 9 seed-51-9.puz | tee search.log
```

Use separate log directories for repeat runs. The searcher stores
checkpoints under `data/` relative to the executable's repository root.
Use all allocated logical CPUs for the 128-thread test, and use
`ILS_GPU_DEVICES=1` for the single-GPU control. Compare the cumulative
counter over matching stage-52 intervals. For the four-size A/B, enable
periodic checkpoints for the seed run, keep a checkpoint written during
stage 52, and pass the same checkpoint path in place of the `.puz`
argument to both builds.

For the retirement test, `seed-48-9.puz` is the first 48 rows of the
published `(52,9)` puzzle. Extract it from the appendix, then substitute
that file for `seed-51-9.puz` in the command above. [Anderson and Le,
*Efficiently-Verifiable Strong Uniquely Solvable Puzzles and Matrix
Multiplication*](paper/2307.06463v1.pdf).

```bash
pdftotext -raw paper/2307.06463v1.pdf - |
  awk '/^Simplifiable \(52, 9\)-SUSP:/{p=1;next}
       /^Simplifiable \(78, 10\)-SUSP:/{p=0}
       p && /^[123]+$/ && length == 9 {print}' |
  sed -n '1,48p' > seed-48-9.puz
```

| Seed | SHA-256 |
| --- | --- |
| `seed-48-9.puz` | `d16536a6706cc5881b9b7e835c0267c828ede4df1beb8186c5f352921fe765e4` |
| `seed-51-9.puz` | `99d7c85b9cb6c23a3526c45505106237401fca7a3af8d8b222e504a9e59eb8eb` |

### Benchmark seed `seed-51-9.puz`

The benchmark seed was derived from the published height-52 puzzle and
screened for the absence of a direct single-row extension. Save the rows
below as plain text with a newline after the last row.

```text
123222111
112232111
121223111
111233111
312333111
222222311
122322311
212232311
221223311
121323311
211233311
221221121
212222131
112322131
122232131
211223131
111323131
121233131
212322331
222232331
122332331
211323331
221233331
121333331
112332312
111333312
111332122
312333332
111222113
213222113
121232113
122223113
112233113
211322313
221232313
121332313
222223313
122323313
212233313
223222133
121322133
211232133
212223133
112323133
122233133
221321333
222231333
211332333
222323333
122333333
113333333
```
