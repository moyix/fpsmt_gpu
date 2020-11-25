#pragma once
#include <stdint.h>

__host__   void expand_key(uint8_t *key, uint8_t *rkey);
__device__ void encrypt_k(uint8_t *data, const uint8_t *rkey, uint32_t numblock);
__device__ void encrypt_one_table(uint8_t *block, const uint8_t *rkey, uint32_t offset);
