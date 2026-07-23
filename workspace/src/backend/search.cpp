#include "search.h"
#include "canonization.h"
#include "bloom_filtering.h"
#include <iostream>

Search::Search()
    : ft(nullptr),
      num_stages(NUM_STAGES),
      curr(NUM_STAGES - 1),
      initilized(false),
      num_seen(0),
      stored_seen(0),
      num_tdm_simplify(0),
      num_tdm_gpu_simplify(0) {}

Search::~Search() {
  if (ft)
    delete ft;
}

void Search::store_isomorph_state() {
  save_isomorph_state(initilized, seen_isomorphs, num_seen, stored_isomorphs, stored_seen);
}

void Search::restore_isomorph_state() {
  ::restore_isomorph_state(initilized, seen_isomorphs, num_seen, stored_isomorphs, stored_seen);
}

void Search::serialize(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
  os.write(reinterpret_cast<const char*>(&strong), sizeof(strong));
  os.write(reinterpret_cast<const char*>(&s_target), sizeof(s_target));
  os.write(reinterpret_cast<const char*>(&s), sizeof(s));
  os.write(reinterpret_cast<const char*>(&k), sizeof(k));
  p.serialize(os);

  os.write(reinterpret_cast<const char*>(&num_stages), sizeof(num_stages));
  os.write(reinterpret_cast<const char*>(&curr), sizeof(curr));

  for (int i = 0; i < num_stages; ++i) {
    bool has_stage = stages[i] != nullptr;
    os.write(reinterpret_cast<const char*>(&has_stage), sizeof(has_stage));
    if (has_stage) {
      os.write(reinterpret_cast<const char*>(&i), sizeof(i));
      stages[i]->serialize(os);
    }
  }

  const_cast<Search*>(this)->store_isomorph_state();

  // Serialize isomorph state directly
  os.write(reinterpret_cast<const char*>(&initilized), sizeof(initilized));

  size_t seen_size = seen_isomorphs.size();
  os.write(reinterpret_cast<const char*>(&seen_size), sizeof(seen_size));
  for (const auto& pair : seen_isomorphs) {
    size_t str_size = pair.first.size();
    os.write(reinterpret_cast<const char*>(&str_size), sizeof(str_size));
    os.write(pair.first.c_str(), str_size);
    os.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
  }

  os.write(reinterpret_cast<const char*>(&num_seen), sizeof(num_seen));

  size_t stored_size = stored_isomorphs.size();
  os.write(reinterpret_cast<const char*>(&stored_size), sizeof(stored_size));
  for (const auto& pair : stored_isomorphs) {
    size_t str_size = pair.first.size();
    os.write(reinterpret_cast<const char*>(&str_size), sizeof(str_size));
    os.write(pair.first.c_str(), str_size);
    os.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
  }

  os.write(reinterpret_cast<const char*>(&stored_seen), sizeof(stored_seen));

  if (ft) {
    std::string ft_name = ft->getName();
    size_t ft_name_size = ft_name.size();
    os.write(reinterpret_cast<const char*>(&ft_name_size), sizeof(ft_name_size));
    os.write(ft_name.c_str(), ft_name_size);
  } else {
    size_t ft_name_size = 0;
    os.write(reinterpret_cast<const char*>(&ft_name_size), sizeof(ft_name_size));
  }

  bloom_filter_serialize(os);
}

void Search::deserialize(std::istream& is) {
  is.read(reinterpret_cast<char*>(&mode), sizeof(mode));
  is.read(reinterpret_cast<char*>(&strong), sizeof(strong));
  is.read(reinterpret_cast<char*>(&s_target), sizeof(s_target));
  is.read(reinterpret_cast<char*>(&s), sizeof(s));
  is.read(reinterpret_cast<char*>(&k), sizeof(k));
  p.deserialize(is);

  is.read(reinterpret_cast<char*>(&num_stages), sizeof(num_stages));
  is.read(reinterpret_cast<char*>(&curr), sizeof(curr));

  stages.clear();
  stages.resize(num_stages, nullptr);

  for (size_t i = 0; i < num_stages; ++i) {
    bool has_stage;
    is.read(reinterpret_cast<char*>(&has_stage), sizeof(has_stage));
    if (has_stage) {
      int stage_index;
      is.read(reinterpret_cast<char*>(&stage_index), sizeof(stage_index));

      stages[stage_index] = std::make_shared<ILSStage>(s, k, ft, nullptr);
      stages[stage_index]->ft = ft;
      stages[stage_index]->deserialize(is);
    }
  }

  int prev_idx = -1;
  for (int i = 0; i < num_stages; ++i) {
    if (stages[i]) {
      if (prev_idx >= 0) {
        stages[i]->setPrev(stages[prev_idx]);
      } else {
        stages[i]->setPrev(nullptr);
      }
      prev_idx = i;
    }
  }

  is.read(reinterpret_cast<char*>(&initilized), sizeof(initilized));

  size_t seen_size;
  is.read(reinterpret_cast<char*>(&seen_size), sizeof(seen_size));
  seen_isomorphs.clear();

  for (size_t i = 0; i < seen_size; ++i) {
    size_t str_size;
    is.read(reinterpret_cast<char*>(&str_size), sizeof(str_size));
    std::string key(str_size, '\0');
    is.read(&key[0], str_size);
    int value;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    seen_isomorphs[key] = value;
  }

  is.read(reinterpret_cast<char*>(&num_seen), sizeof(num_seen));

  size_t stored_size;
  is.read(reinterpret_cast<char*>(&stored_size), sizeof(stored_size));
  stored_isomorphs.clear();
  for (size_t i = 0; i < stored_size; ++i) {
    size_t str_size;
    is.read(reinterpret_cast<char*>(&str_size), sizeof(str_size));
    std::string key(str_size, '\0');
    is.read(&key[0], str_size);
    int value;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    stored_isomorphs[key] = value;
  }

  is.read(reinterpret_cast<char*>(&stored_seen), sizeof(stored_seen));

  restore_isomorph_state();

  size_t ft_name_size;
  is.read(reinterpret_cast<char*>(&ft_name_size), sizeof(ft_name_size));
  if (ft_name_size > 0) {
    std::string ft_name(ft_name_size, '\0');
    is.read(&ft_name[0], ft_name_size);
    if (ft)
      delete ft;
    ft = createFitTesterFromType(ft_name);
    if (!ft) {
      std::cerr << "Error: Unknown FitTester name '" << ft_name << "' during deserialization.\n";
      exit(EXIT_FAILURE);
    }
  } else {
    if (ft) {
      delete ft;
      ft = nullptr;
    }
  }

  bloom_filter_deserialize(is);
}
