#pragma once

#include <cstddef>
#include <iosfwd>

void enable_bloom_filtering(size_t expected_elements, double false_positive_rate);
void disable_bloom_filtering();
bool bloom_filter_is_enabled();
void reset_bloom_filter();

bool bloom_filter_check_and_add(const void* data, size_t len, bool remember);
size_t bloom_filter_estimate_count();

void bloom_filter_serialize(std::ostream& os);
bool bloom_filter_deserialize(std::istream& is);
