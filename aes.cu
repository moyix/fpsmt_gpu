#include "aes_table.h"

// CUDA AES implementation courtesy of Andrew Pratt
// https://www.andrew.cmu.edu/user/aspratt/accelerated_aes/
// No license specified, so uh, hope he doesn't mind?

//XOR round key with block(1 block per thread)
__device__ void add_round_key(uint8_t *block, const uint8_t *key, uint32_t offset){
  //word size traversal
  uint32_t *b = (uint32_t *)block;
  uint32_t *k = (uint32_t *)key;
  for(int i = 0; i < 4; ++i){
    b[offset/4 + i] = b[offset/4 + i] ^ k[i];
  }  
}

//substitute block int sbox (1 block per thread)
__device__ void sub_bytes(uint8_t *block, uint32_t offset){
  for(int i = 0; i < 16; ++i){
    block[offset + i] = sbox[block[offset + i]];
  }
}

//mix columns by taking linear combinations in the field (1 block per thread)
__device__ void mix_columns(uint8_t *block, uint32_t offset){
  for(int i = 0; i < 4; ++i){ //iterate over columns
    uint8_t a[4];
    uint8_t b[4]; 
    uint8_t h;
  
    for(int j = 0; j < 4; ++j){
      a[j] = block[offset + 4*i + j];
      h = (uint8_t)((int8_t)a[j] >> 7);
      b[j] = a[j] << 1;
      b[j] ^= 0x1b & h;
    } 

    block[offset + 4*i + 0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
    block[offset + 4*i + 1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
    block[offset + 4*i + 2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
    block[offset + 4*i + 3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]; 

  }
}

//shift rows left by 0,1,2,3 bytes respectively (1 block per thread)
__device__ void shift_rows(uint8_t *sblock, uint32_t offset){
  uint8_t tmp;

  uint8_t *block = sblock + offset; 

  //row 0 remains unshifted

  //shift row 1 left by 1
  tmp = block[1];
  block[1] = block[5];
  block[5] = block[9];
  block[9] = block[13];
  block[13] = tmp;

  //shift row 2 letf by 2
  tmp = block[2];
  block[2] = block[10];
  block[10] = tmp;

  tmp = block[6];
  block[6] = block[14];
  block[14] = tmp;

  //shift row 3 left by 3
  tmp = block[3];
  block[3] = block[15];
  block[15] = block[11];
  block[11] = block[7];
  block[7] = tmp;

}

//aes 128 encryption with expanded key supplied
//implemented using 1 t-tables (with rotation) and sbox
//1 block per thread
__device__ void encrypt_one_table(uint8_t *block, const uint8_t *rkey, uint32_t offset){

  uint8_t *b = (block + offset);
  uint32_t *bword = (uint32_t *)(block + offset);

  add_round_key(block, rkey, offset);

  for(int i = 1; i < 10; ++i){

    uint32_t *ckey = (uint32_t *)(rkey + 16*i);

    uint32_t c1 = te0[b[0]]   ^ (te0[b[5]]<<8 | te0[b[5]]>>24)    ^ (te0[b[10]]<<16 | te0[b[10]]>>16) ^ (te0[b[15]]<<24 | te0[b[15]]>>8) ^ ckey[0];
    uint32_t c2 = te0[b[4]]   ^ (te0[b[9]]<<8 | te0[b[9]]>>24)    ^ (te0[b[14]]<<16 | te0[b[14]]>>16) ^ (te0[b[3]]<<24 | te0[b[3]]>>8)   ^ ckey[1];
    uint32_t c3 = te0[b[8]]   ^ (te0[b[13]]<<8 | te0[b[13]]>>24)  ^ (te0[b[2]]<<16 | te0[b[2]]>>16)   ^ (te0[b[7]]<<24 | te0[b[7]]>>8)   ^ ckey[2];
    uint32_t c4 = te0[b[12]]  ^ (te0[b[1]]<<8 | te0[b[1]]>>24)    ^ (te0[b[6]]<<16 | te0[b[6]]>>16)   ^ (te0[b[11]]<<24 | te0[b[11]]>>8) ^ ckey[3];

    bword[0] = c1;
    bword[1] = c2;
    bword[2] = c3;
    bword[3] = c4;

  }

  sub_bytes(block, offset);
  shift_rows(block, offset);
  add_round_key(block, rkey + 160, offset);

}

//basic encryption kernel.  Unused for ctr mode encryption
__device__ void encrypt_k(uint8_t *data, const uint8_t *rkey, uint32_t numblock){
  int bindex = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = bindex * 16;
  if(bindex >= numblock) return;
    encrypt_one_table(data, rkey, offset);
}

//generate round keys from initial key
__host__ void expand_key(uint8_t *key, uint8_t *rkey){

  uint32_t i,j,k;
  uint8_t tempa[4];
  uint32_t nround = 10;

  //first round key is just the key
  for(i = 0; i < 4; ++i){
    rkey[4*i + 0] = key[4*i + 0];
    rkey[4*i + 1] = key[4*i + 1];
    rkey[4*i + 2] = key[4*i + 2];
    rkey[4*i + 3] = key[4*i + 3];
  }

  for(i = 4; i < 4*(nround + 1); ++i){
    for(j = 0; j < 4; ++j){
      tempa[j] = rkey[(i-1)*4 + j];
    }

    if(i % 4 == 0){
      //rotate 4 bytes in word
      k = tempa[0];
      tempa[0] = tempa[1];
      tempa[1] = tempa[2];
      tempa[2] = tempa[3];
      tempa[3] = k;


      tempa[0] = sbox_host[tempa[0]];
      tempa[1] = sbox_host[tempa[1]];
      tempa[2] = sbox_host[tempa[2]];
      tempa[3] = sbox_host[tempa[3]];
  
      tempa[0] = tempa[0] ^ rcon[i/4];

    }

    rkey[4*i + 0] = rkey[4*(i-4) + 0] ^ tempa[0];
    rkey[4*i + 1] = rkey[4*(i-4) + 1] ^ tempa[1];
    rkey[4*i + 2] = rkey[4*(i-4) + 2] ^ tempa[2];
    rkey[4*i + 3] = rkey[4*(i-4) + 3] ^ tempa[3];

  }

}
