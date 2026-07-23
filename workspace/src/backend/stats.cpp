#include "stats.h"
#include <iostream>

void _stats_t::serialize(std::ostream& os) const {
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> current_total_time = total_time + (end_time - start);
  long long total_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(current_total_time).count();
  unsigned long long stat_num_gpu_simplify = TDM::getNumGPUSimplify();
  unsigned long long stat_num_simplify = TDM::getNumSimplify();
  os.write(reinterpret_cast<const char*>(&evals), sizeof(evals));
  os.write(reinterpret_cast<const char*>(&repeated), sizeof(repeated));
  os.write(reinterpret_cast<const char*>(&proc), sizeof(proc));
  os.write(reinterpret_cast<const char*>(&dropped), sizeof(dropped));
  os.write(reinterpret_cast<const char*>(&since_progress), sizeof(since_progress));
  os.write(reinterpret_cast<const char*>(&restarts), sizeof(restarts));
  os.write(reinterpret_cast<const char*>(&best_gap), sizeof(best_gap));
  os.write(reinterpret_cast<const char*>(&max_found), sizeof(max_found));
  os.write(reinterpret_cast<const char*>(&working), sizeof(working));
  os.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
  os.write(reinterpret_cast<const char*>(&total_time_ns), sizeof(total_time_ns));
  os.write(reinterpret_cast<const char*>(&searched_types), sizeof(searched_types));
  os.write(reinterpret_cast<const char*>(&stat_num_gpu_simplify), sizeof(stat_num_gpu_simplify));
  os.write(reinterpret_cast<const char*>(&stat_num_simplify), sizeof(stat_num_simplify));
}

void _stats_t::deserialize(std::istream& is) {
  is.read(reinterpret_cast<char*>(&evals), sizeof(evals));
  is.read(reinterpret_cast<char*>(&repeated), sizeof(repeated));
  is.read(reinterpret_cast<char*>(&proc), sizeof(proc));
  is.read(reinterpret_cast<char*>(&dropped), sizeof(dropped));
  is.read(reinterpret_cast<char*>(&since_progress), sizeof(since_progress));
  is.read(reinterpret_cast<char*>(&restarts), sizeof(restarts));
  is.read(reinterpret_cast<char*>(&best_gap), sizeof(best_gap));
  is.read(reinterpret_cast<char*>(&max_found), sizeof(max_found));
  is.read(reinterpret_cast<char*>(&working), sizeof(working));
  is.read(reinterpret_cast<char*>(&mode), sizeof(mode));
  long long total_time_ns;
  is.read(reinterpret_cast<char*>(&total_time_ns), sizeof(total_time_ns));
  total_time = std::chrono::duration<double>(std::chrono::nanoseconds(total_time_ns));
  is.read(reinterpret_cast<char*>(&searched_types), sizeof(searched_types));
  unsigned long long stat_num_gpu_simplify;
  unsigned long long stat_num_simplify;
  is.read(reinterpret_cast<char*>(&stat_num_gpu_simplify), sizeof(stat_num_gpu_simplify));
  is.read(reinterpret_cast<char*>(&stat_num_simplify), sizeof(stat_num_simplify));
  TDM::setNumGPUSimplify(stat_num_gpu_simplify);
  TDM::setNumSimplify(stat_num_simplify);
}
