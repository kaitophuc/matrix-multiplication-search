/*
 * Bloom filter support for isomorph caching.
 * With support for serialization/deserialization.
 *
 * Author: Matt Anderson, Minh Phuc Nguyen
 * 
 * This file contains AI-generated code. Be careful when editing to ensure correctness.
 */

#include "bloom_filtering.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
constexpr char kBloomMagic[8] = {'B', 'L', 'O', 'O', 'M', 'F', '1', '\0'};
constexpr uint32_t kBloomVersion = 1;
constexpr uint64_t kSeed1 = 0x12345678ULL;
constexpr uint64_t kSeed2 = 0x87654321ULL;

bool g_bloom_enabled = false;
size_t g_num_bits = 0;
int g_num_hashes = 0;
size_t g_num_words = 0;
size_t g_expected_elements = 0;
double g_false_positive_rate = 0.0;
size_t g_inserted = 0;
std::vector<uint64_t> g_bits;

uint64_t hash_bytes(const void* key, size_t len, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;
  uint64_t h = seed ^ (len * m);

  const uint64_t* data = static_cast<const uint64_t*>(key);
  const uint64_t* end = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
  }

  const unsigned char* data2 = reinterpret_cast<const unsigned char*>(data);
  switch (len & 7) {
    case 7:
      h ^= uint64_t(data2[6]) << 48;
    case 6:
      h ^= uint64_t(data2[5]) << 40;
    case 5:
      h ^= uint64_t(data2[4]) << 32;
    case 4:
      h ^= uint64_t(data2[3]) << 24;
    case 3:
      h ^= uint64_t(data2[2]) << 16;
    case 2:
      h ^= uint64_t(data2[1]) << 8;
    case 1:
      h ^= uint64_t(data2[0]);
      h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
}

inline bool get_bit(size_t bit) {
  size_t word = bit >> 6;
  return (g_bits[word] >> (bit & 63)) & 1ULL;
}

inline void set_bit(size_t bit) {
  size_t word = bit >> 6;
  g_bits[word] |= 1ULL << (bit & 63);
}

void reset_state() {
  g_bloom_enabled = false;
  g_num_bits = 0;
  g_num_hashes = 0;
  g_num_words = 0;
  g_expected_elements = 0;
  g_false_positive_rate = 0.0;
  g_inserted = 0;
  g_bits.clear();
}
}  // namespace

void enable_bloom_filtering(size_t expected_elements, double false_positive_rate) {
  if (expected_elements == 0) {
    expected_elements = 1;
  }
  if (false_positive_rate <= 0.0 || false_positive_rate >= 1.0) {
    false_positive_rate = 0.01;
  }

  double ln2 = log(2.0);
  double num_bits_d =
      -static_cast<double>(expected_elements) * log(false_positive_rate) / (ln2 * ln2);
  size_t num_bits = static_cast<size_t>(num_bits_d);
  if (num_bits == 0) {
    num_bits = 1;
  }

  int num_hashes =
      static_cast<int>((static_cast<double>(num_bits) / expected_elements) * ln2);
  if (num_hashes < 1) {
    num_hashes = 1;
  }

  g_num_bits = num_bits;
  g_num_hashes = num_hashes;
  g_num_words = (g_num_bits + 63) / 64;
  g_expected_elements = expected_elements;
  g_false_positive_rate = false_positive_rate;
  g_inserted = 0;
  g_bits.assign(g_num_words, 0);
  g_bloom_enabled = true;

  std::cout << "Bloom Filter Enabled: " << (g_num_bits / 8 / 1024 / 1024) << " MB, "
            << g_num_hashes << " hashes." << std::endl;
}

void disable_bloom_filtering() {
  reset_state();
}

bool bloom_filter_is_enabled() {
  return g_bloom_enabled;
}

void reset_bloom_filter() {
  if (!g_bloom_enabled) {
    return;
  }
  std::fill(g_bits.begin(), g_bits.end(), 0);
  g_inserted = 0;
}

bool bloom_filter_check_and_add(const void* data, size_t len, bool remember) {
  if (!g_bloom_enabled || g_num_bits == 0 || g_num_hashes <= 0) {
    return false;
  }

  uint64_t h1 = hash_bytes(data, len, kSeed1);
  uint64_t h2 = hash_bytes(data, len, kSeed2);
  if (h2 == 0) {
    h2 = 0x9e3779b97f4a7c15ULL;
  }

  bool seen = true;
  bool inserted = false;
  for (int i = 0; i < g_num_hashes; i++) {
    size_t bit = static_cast<size_t>((h1 + static_cast<uint64_t>(i) * h2) % g_num_bits);
    if (!get_bit(bit)) {
      seen = false;
      inserted = true;
      if (remember) {
        set_bit(bit);
      } else {
        return false;
      }
    }
  }

  if (remember && inserted) {
    g_inserted++;
  }
  return seen;
}

size_t bloom_filter_estimate_count() {
  return g_inserted;
}

void bloom_filter_serialize(std::ostream& os) {
  os.write(kBloomMagic, sizeof(kBloomMagic));
  uint32_t version = kBloomVersion;
  os.write(reinterpret_cast<const char*>(&version), sizeof(version));

  uint8_t enabled = g_bloom_enabled ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&enabled), sizeof(enabled));
  uint8_t reserved[3] = {0, 0, 0};
  os.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));

  if (!g_bloom_enabled) {
    return;
  }

  uint64_t num_bits = static_cast<uint64_t>(g_num_bits);
  uint32_t num_hashes = static_cast<uint32_t>(g_num_hashes);
  uint64_t num_words = static_cast<uint64_t>(g_bits.size());
  uint64_t inserted = static_cast<uint64_t>(g_inserted);
  uint64_t expected = static_cast<uint64_t>(g_expected_elements);
  double fpr = g_false_positive_rate;

  os.write(reinterpret_cast<const char*>(&num_bits), sizeof(num_bits));
  os.write(reinterpret_cast<const char*>(&num_hashes), sizeof(num_hashes));
  os.write(reinterpret_cast<const char*>(&num_words), sizeof(num_words));
  os.write(reinterpret_cast<const char*>(&inserted), sizeof(inserted));
  os.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
  os.write(reinterpret_cast<const char*>(&fpr), sizeof(fpr));

  if (!g_bits.empty()) {
    os.write(reinterpret_cast<const char*>(g_bits.data()),
             static_cast<std::streamsize>(g_bits.size() * sizeof(uint64_t)));
  }
}

bool bloom_filter_deserialize(std::istream& is) {
  char magic[sizeof(kBloomMagic)];
  is.read(magic, sizeof(magic));
  if (!is) {
    is.clear();
    reset_state();
    return false;
  }

  if (std::memcmp(magic, kBloomMagic, sizeof(kBloomMagic)) != 0) {
    reset_state();
    return false;
  }

  uint32_t version = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (!is || version != kBloomVersion) {
    reset_state();
    return false;
  }

  uint8_t enabled = 0;
  uint8_t reserved[3] = {0, 0, 0};
  is.read(reinterpret_cast<char*>(&enabled), sizeof(enabled));
  is.read(reinterpret_cast<char*>(reserved), sizeof(reserved));
  if (!is) {
    reset_state();
    return false;
  }

  if (enabled == 0) {
    reset_state();
    return true;
  }

  uint64_t num_bits = 0;
  uint32_t num_hashes = 0;
  uint64_t num_words = 0;
  uint64_t inserted = 0;
  uint64_t expected = 0;
  double fpr = 0.0;

  is.read(reinterpret_cast<char*>(&num_bits), sizeof(num_bits));
  is.read(reinterpret_cast<char*>(&num_hashes), sizeof(num_hashes));
  is.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));
  is.read(reinterpret_cast<char*>(&inserted), sizeof(inserted));
  is.read(reinterpret_cast<char*>(&expected), sizeof(expected));
  is.read(reinterpret_cast<char*>(&fpr), sizeof(fpr));
  if (!is || num_bits == 0 || num_hashes == 0 || num_words == 0) {
    reset_state();
    return false;
  }

  uint64_t expected_words = (num_bits + 63) / 64;
  if (num_words != expected_words) {
    reset_state();
    return false;
  }

  std::vector<uint64_t> bits;
  bits.resize(static_cast<size_t>(num_words));
  is.read(reinterpret_cast<char*>(bits.data()),
          static_cast<std::streamsize>(bits.size() * sizeof(uint64_t)));
  if (!is) {
    reset_state();
    return false;
  }

  g_num_bits = static_cast<size_t>(num_bits);
  g_num_hashes = static_cast<int>(num_hashes);
  g_num_words = static_cast<size_t>(num_words);
  g_expected_elements = static_cast<size_t>(expected);
  g_false_positive_rate = fpr;
  g_inserted = static_cast<size_t>(inserted);
  g_bits = std::move(bits);
  g_bloom_enabled = true;
  return true;
}
