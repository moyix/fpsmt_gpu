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

#ifndef __CRYPTO_PRIMITIVES_CHAM_H__
#define __CRYPTO_PRIMITIVES_CHAM_H__

#include <stdint.h>
#include <stddef.h>

__device__ void cham64_keygen(uint8_t* rks, const uint8_t* mk);
__device__ void cham64_encrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);
__device__ void cham64_decrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);

__device__ void cham128_keygen(uint8_t* rks, const uint8_t* mk);
__device__ void cham128_encrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);
__device__ void cham128_decrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);

__device__ void cham256_keygen(uint8_t* rks, const uint8_t* mk);
__device__ void cham256_encrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);
__device__ void cham256_decrypt(uint8_t* dst, const uint8_t* src, const uint8_t* rks);

#endif
