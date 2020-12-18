/**
 * The MIT License
 *
 * Copyright (c) 2018-2020 Ilwoong Jeong (https://github.com/ilwoong)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "cham.h"

#include <string.h>

static const size_t BLOCKSIZE_64 = 8;
static const size_t BLOCKSIZE_128 = 16;

static const size_t CHAM_64_128_ROUNDS = 88;
static const size_t CHAM_128_128_ROUNDS = 8;
static const size_t CHAM_128_256_ROUNDS = 120;

__device__ static inline uint16_t rol16(uint16_t value, size_t rot) { return (value << rot) | (value >> (16 - rot)); }

__device__ static inline uint16_t ror16(uint16_t value, size_t rot) { return (value >> rot) | (value << (16 - rot)); }

__device__ static inline uint32_t rol32(uint32_t value, size_t rot) { return (value << rot) | (value >> (32 - rot)); }

__device__ static inline uint32_t ror32(uint32_t value, size_t rot) { return (value >> rot) | (value << (32 - rot)); }

/**
 * CHAM 64-bit block, 128-bit key
 */
__device__ void cham64_keygen(uint8_t *rks, const uint8_t *mk) {
  const uint16_t *key = (uint16_t *)mk;
  uint16_t *rk = (uint16_t *)rks;

  for (size_t i = 0; i < 8; ++i) {
    rk[i] = key[i] ^ rol16(key[i], 1);
    rk[(i + 8) ^ (0x1)] = rk[i] ^ rol16(key[i], 11);
    rk[i] ^= rol16(key[i], 8);
  }
}

__device__ void cham64_encrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint16_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_64);

  const uint16_t *rk = (const uint16_t *)rks;
  uint16_t rc = 0;

  for (size_t round = 0; round < CHAM_64_128_ROUNDS; round += 8) {
    blk[0] = rol16((blk[0] ^ (rc++)) + (rol16(blk[1], 1) ^ rk[0]), 8);
    blk[1] = rol16((blk[1] ^ (rc++)) + (rol16(blk[2], 8) ^ rk[1]), 1);
    blk[2] = rol16((blk[2] ^ (rc++)) + (rol16(blk[3], 1) ^ rk[2]), 8);
    blk[3] = rol16((blk[3] ^ (rc++)) + (rol16(blk[0], 8) ^ rk[3]), 1);

    blk[0] = rol16((blk[0] ^ (rc++)) + (rol16(blk[1], 1) ^ rk[4]), 8);
    blk[1] = rol16((blk[1] ^ (rc++)) + (rol16(blk[2], 8) ^ rk[5]), 1);
    blk[2] = rol16((blk[2] ^ (rc++)) + (rol16(blk[3], 1) ^ rk[6]), 8);
    blk[3] = rol16((blk[3] ^ (rc++)) + (rol16(blk[0], 8) ^ rk[7]), 1);

    rk = (rk == (const uint16_t *)rks) ? rk + 8 : rk - 8;
  }

  memcpy(dst, blk, BLOCKSIZE_64);
}

__device__ void cham64_decrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint16_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_64);

  const uint16_t *rk = (const uint16_t *)rks;
  uint16_t rc = CHAM_64_128_ROUNDS;

  for (size_t round = 0; round < CHAM_64_128_ROUNDS; round += 8) {
    blk[3] = (ror16(blk[3], 1) - (rol16(blk[0], 8) ^ rk[7])) ^ (--rc);
    blk[2] = (ror16(blk[2], 8) - (rol16(blk[3], 1) ^ rk[6])) ^ (--rc);
    blk[1] = (ror16(blk[1], 1) - (rol16(blk[2], 8) ^ rk[5])) ^ (--rc);
    blk[0] = (ror16(blk[0], 8) - (rol16(blk[1], 1) ^ rk[4])) ^ (--rc);

    blk[3] = (ror16(blk[3], 1) - (rol16(blk[0], 8) ^ rk[3])) ^ (--rc);
    blk[2] = (ror16(blk[2], 8) - (rol16(blk[3], 1) ^ rk[2])) ^ (--rc);
    blk[1] = (ror16(blk[1], 1) - (rol16(blk[2], 8) ^ rk[1])) ^ (--rc);
    blk[0] = (ror16(blk[0], 8) - (rol16(blk[1], 1) ^ rk[0])) ^ (--rc);

    rk = (rk == (const uint16_t *)rks) ? rk + 8 : rk - 8;
  }

  memcpy(dst, blk, BLOCKSIZE_64);
}

/**
 * CHAM 128-bit block, 128-bit key
 */
__device__ void cham128_keygen(uint8_t *rks, const uint8_t *mk) {
  const uint32_t *key = (uint32_t *)mk;
  uint32_t *rk = (uint32_t *)rks;

  for (size_t i = 0; i < 4; ++i) {
    rk[i] = key[i] ^ rol32(key[i], 1);
    rk[(i + 4) ^ (0x1)] = rk[i] ^ rol32(key[i], 11);
    rk[i] ^= rol32(key[i], 8);
  }
}

__device__ void cham128_encrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint32_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_128);

  const uint32_t *rk = (const uint32_t *)rks;
  uint32_t rc = 0;

  for (size_t round = 0; round < CHAM_128_128_ROUNDS; round += 8) {
    blk[0] = rol32((blk[0] ^ (rc++)) + (rol32(blk[1], 1) ^ rk[0]), 8);
    blk[1] = rol32((blk[1] ^ (rc++)) + (rol32(blk[2], 8) ^ rk[1]), 1);
    blk[2] = rol32((blk[2] ^ (rc++)) + (rol32(blk[3], 1) ^ rk[2]), 8);
    blk[3] = rol32((blk[3] ^ (rc++)) + (rol32(blk[0], 8) ^ rk[3]), 1);

    blk[0] = rol32((blk[0] ^ (rc++)) + (rol32(blk[1], 1) ^ rk[4]), 8);
    blk[1] = rol32((blk[1] ^ (rc++)) + (rol32(blk[2], 8) ^ rk[5]), 1);
    blk[2] = rol32((blk[2] ^ (rc++)) + (rol32(blk[3], 1) ^ rk[6]), 8);
    blk[3] = rol32((blk[3] ^ (rc++)) + (rol32(blk[0], 8) ^ rk[7]), 1);
  }

  memcpy(dst, blk, BLOCKSIZE_128);
}

__device__ void cham128_decrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint32_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_128);

  const uint32_t *rk = (const uint32_t *)rks;
  uint32_t rc = CHAM_128_128_ROUNDS;

  for (size_t round = 0; round < CHAM_128_128_ROUNDS; round += 8) {
    blk[3] = (ror32(blk[3], 1) - (rol32(blk[0], 8) ^ rk[7])) ^ (--rc);
    blk[2] = (ror32(blk[2], 8) - (rol32(blk[3], 1) ^ rk[6])) ^ (--rc);
    blk[1] = (ror32(blk[1], 1) - (rol32(blk[2], 8) ^ rk[5])) ^ (--rc);
    blk[0] = (ror32(blk[0], 8) - (rol32(blk[1], 1) ^ rk[4])) ^ (--rc);

    blk[3] = (ror32(blk[3], 1) - (rol32(blk[0], 8) ^ rk[3])) ^ (--rc);
    blk[2] = (ror32(blk[2], 8) - (rol32(blk[3], 1) ^ rk[2])) ^ (--rc);
    blk[1] = (ror32(blk[1], 1) - (rol32(blk[2], 8) ^ rk[1])) ^ (--rc);
    blk[0] = (ror32(blk[0], 8) - (rol32(blk[1], 1) ^ rk[0])) ^ (--rc);
  }

  memcpy(dst, blk, BLOCKSIZE_128);
}

/**
 * CHAM 128-bit block, 256-bit key
 */
__device__ void cham256_keygen(uint8_t *rks, const uint8_t *mk) {
  const uint32_t *key = (uint32_t *)mk;
  uint32_t *rk = (uint32_t *)rks;

  for (size_t i = 0; i < 8; ++i) {
    rk[i] = key[i] ^ rol32(key[i], 1);
    rk[(i + 8) ^ (0x1)] = rk[i] ^ rol32(key[i], 11);
    rk[i] ^= rol32(key[i], 8);
  }
}

__device__ void cham256_encrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint32_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_128);

  const uint32_t *rk = (const uint32_t *)rks;
  uint32_t rc = 0;

  for (size_t round = 0; round < CHAM_128_256_ROUNDS; round += 8) {
    blk[0] = rol32((blk[0] ^ (rc++)) + (rol32(blk[1], 1) ^ rk[0]), 8);
    blk[1] = rol32((blk[1] ^ (rc++)) + (rol32(blk[2], 8) ^ rk[1]), 1);
    blk[2] = rol32((blk[2] ^ (rc++)) + (rol32(blk[3], 1) ^ rk[2]), 8);
    blk[3] = rol32((blk[3] ^ (rc++)) + (rol32(blk[0], 8) ^ rk[3]), 1);

    blk[0] = rol32((blk[0] ^ (rc++)) + (rol32(blk[1], 1) ^ rk[4]), 8);
    blk[1] = rol32((blk[1] ^ (rc++)) + (rol32(blk[2], 8) ^ rk[5]), 1);
    blk[2] = rol32((blk[2] ^ (rc++)) + (rol32(blk[3], 1) ^ rk[6]), 8);
    blk[3] = rol32((blk[3] ^ (rc++)) + (rol32(blk[0], 8) ^ rk[7]), 1);

    rk = (rk == (const uint32_t *)rks) ? rk + 8 : rk - 8;
  }

  memcpy(dst, blk, BLOCKSIZE_128);
}

__device__ void cham256_decrypt(uint8_t *dst, const uint8_t *src, const uint8_t *rks) {
  uint32_t blk[4] = {0};
  memcpy(blk, src, BLOCKSIZE_128);

  const uint32_t *rk = (const uint32_t *)rks;
  uint32_t rc = CHAM_128_256_ROUNDS;

  for (size_t round = 0; round < CHAM_128_256_ROUNDS; round += 8) {
    blk[3] = (ror32(blk[3], 1) - (rol32(blk[0], 8) ^ rk[7])) ^ (--rc);
    blk[2] = (ror32(blk[2], 8) - (rol32(blk[3], 1) ^ rk[6])) ^ (--rc);
    blk[1] = (ror32(blk[1], 1) - (rol32(blk[2], 8) ^ rk[5])) ^ (--rc);
    blk[0] = (ror32(blk[0], 8) - (rol32(blk[1], 1) ^ rk[4])) ^ (--rc);

    blk[3] = (ror32(blk[3], 1) - (rol32(blk[0], 8) ^ rk[3])) ^ (--rc);
    blk[2] = (ror32(blk[2], 8) - (rol32(blk[3], 1) ^ rk[2])) ^ (--rc);
    blk[1] = (ror32(blk[1], 1) - (rol32(blk[2], 8) ^ rk[1])) ^ (--rc);
    blk[0] = (ror32(blk[0], 8) - (rol32(blk[1], 1) ^ rk[0])) ^ (--rc);

    rk = (rk == (const uint32_t *)rks) ? rk + 8 : rk - 8;
  }

  memcpy(dst, blk, BLOCKSIZE_128);
}
