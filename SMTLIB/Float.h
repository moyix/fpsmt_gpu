//===----------------------------------------------------------------------===//
//
//                                     JFS
//
// Copyright 2017-2018 Daniel Liew
//
// This file is distributed under the MIT license.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
#ifndef JFS_RUNTIME_SMTLIB_FLOAT_H
#define JFS_RUNTIME_SMTLIB_FLOAT_H
#include "BitVector.h"
#include "BufferRef.h"
#include "NativeFloat.h"
#include <stdint.h>
#include <type_traits>

// Arbitary precision floating point with
// EB exponent bits and SB significand bits (includes implicit bit)
// that mimics the semantics of SMT-LIBv2
// TODO: Implement genric version
template <uint64_t EB, uint64_t SB> class Float {};

typedef Float<8, 24> Float32;
typedef Float<11, 53> Float64;

// FIXME: Refactor this so we don't duplicate code
// Specialize for native types
template <> class Float<8, 24> {
private:
  jfs_nr_float32 data;

public:
  __device__ Float(jfs_nr_float32 value) : data(value) {}
  __device__ Float() : data(0.0f) {}
  __device__ Float(const Float<8, 24>& other) { data = other.data; }
  __device__ Float(BitVector<1> sign, BitVector<8> exponent, BitVector<23> significand) {
    data = jfs_nr_make_float32_from_triple(sign.data, exponent.data,
                                           significand.data);
  }
  __device__ Float(const BitVector<32> bits)
      : data(jfs_nr_bitcast_bv_to_float32(bits.data)) {}

  // Conversion
  template <uint64_t NEW_EB, uint64_t NEW_SB>
  __device__ Float<NEW_EB, NEW_SB> convertToFloat(JFS_NR_RM rm) const;

  // TODO: Need to support non native BitVector
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ static Float32 convertFromUnsignedBV(JFS_NR_RM rm,
                                       const BitVector<BVWIDTH> bvValue) {
    return jfs_nr_convert_from_unsigned_bv_to_float32(rm, bvValue.data,
                                                      BVWIDTH);
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ static Float32 convertFromSignedBV(JFS_NR_RM rm,
                                     const BitVector<BVWIDTH> bvValue) {
    return jfs_nr_convert_from_signed_bv_to_float32(rm, bvValue.data, BVWIDTH);
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ BitVector<BVWIDTH> convertToUnsignedBV(JFS_NR_RM rm) const {
    return BitVector<BVWIDTH>(
        jfs_nr_float32_convert_to_unsigned_bv(rm, data, BVWIDTH));
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ BitVector<BVWIDTH> convertToSignedBV(JFS_NR_RM rm) const {
    return BitVector<BVWIDTH>(
        jfs_nr_float32_convert_to_signed_bv(rm, data, BVWIDTH));
  }

  // Special constants
  __device__ static Float32 getPositiveInfinity() {
    return jfs_nr_float32_get_infinity(true);
  }
  __device__ static Float32 getNegativeInfinity() {
    return jfs_nr_float32_get_infinity(false);
  }
  __device__ static Float32 getPositiveZero() { return jfs_nr_float32_get_zero(true); }
  __device__ static Float32 getNegativeZero() { return jfs_nr_float32_get_zero(false); }
  __device__ static Float32 getNaN() { return jfs_nr_float32_get_nan(true); }

  // SMT-LIBv2 bit comparison
  __device__ bool operator==(const Float32& other) const {
    return jfs_nr_float32_smtlib_equals(data, other.data);
  }

  __device__ bool ieeeEquals(const Float32& other) const {
    return jfs_nr_float32_ieee_equals(data, other.data);
  }

  __device__ bool fplt(const Float32& other) const {
    return jfs_nr_float32_lt(data, other.data);
  }
  __device__ bool fpleq(const Float32& other) const {
    return jfs_nr_float32_leq(data, other.data);
  }
  __device__ bool fpgt(const Float32& other) const {
    return jfs_nr_float32_gt(data, other.data);
  }
  __device__ bool fpgeq(const Float32& other) const {
    return jfs_nr_float32_geq(data, other.data);
  }

  // Arithmetic
  __device__ Float32 abs() const { return jfs_nr_float32_abs(data); }
  __device__ Float32 neg() const { return jfs_nr_float32_neg(data); }
  __device__ Float32 add(JFS_NR_RM rm, const Float32& other) const {
    return jfs_nr_float32_add(rm, data, other.data);
  };
  __device__ Float32 sub(JFS_NR_RM rm, const Float32& other) const {
    return jfs_nr_float32_sub(rm, data, other.data);
  };
  __device__ Float32 mul(JFS_NR_RM rm, const Float32& other) const {
    return jfs_nr_float32_mul(rm, data, other.data);
  };
  __device__ Float32 div(JFS_NR_RM rm, const Float32& other) const {
    return jfs_nr_float32_div(rm, data, other.data);
  };
  __device__ Float32 fma(JFS_NR_RM rm, const Float32& b, const Float32& c) const {
    return jfs_nr_float32_fma(rm, data, b.data, c.data);
  };
  __device__ Float32 sqrt(JFS_NR_RM rm) const { return jfs_nr_float32_sqrt(rm, data); }
  __device__ Float32 rem(const Float32& other) const {
    return jfs_nr_float32_rem(data, other.data);
  };
  __device__ Float32 roundToIntegral(JFS_NR_RM rm) const {
    return jfs_nr_float32_round_to_integral(rm, data);
  }
  __device__ Float32 min(const Float32& other) const {
    return jfs_nr_float32_min(data, other.data);
  }
  __device__ Float32 max(const Float32& other) const {
    return jfs_nr_float32_max(data, other.data);
  }

  // Prediactes
  __device__ bool isNormal() const { return jfs_nr_float32_is_normal(data); }
  __device__ bool isSubnormal() const { return jfs_nr_float32_is_subnormal(data); }
  __device__ bool isZero() const { return jfs_nr_float32_is_zero(data); }
  __device__ bool isInfinite() const { return jfs_nr_float32_is_infinite(data); }
  __device__ bool isPositive() const { return jfs_nr_float32_is_positive(data); }
  __device__ bool isNegative() const { return jfs_nr_float32_is_negative(data); }
  __device__ bool isNaN() const { return jfs_nr_float32_is_nan(data); }

  // For testing
  __device__ uint32_t getRawBits() const { return jfs_nr_float32_get_raw_bits(data); }
  __device__ jfs_nr_float32 getRawData() const { return data; }
};

template <> class Float<11, 53> {
private:
  jfs_nr_float64 data;

public:
  __device__ Float(jfs_nr_float64 value) : data(value) {}
  __device__ Float() : data(0.0) {}
  __device__ Float(const Float<11, 53>& other) { data = other.data; }
  __device__ Float(BitVector<1> sign, BitVector<11> exponent, BitVector<52> significand) {
    data = jfs_nr_make_float64_from_triple(sign.data, exponent.data,
                                           significand.data);
  }
  __device__ Float(const BitVector<64> bits)
      : data(jfs_nr_bitcast_bv_to_float64(bits.data)) {}

  // Conversion
  template <uint64_t NEW_EB, uint64_t NEW_SB>
  __device__ Float<NEW_EB, NEW_SB> convertToFloat(JFS_NR_RM rm) const;

  // TODO: Need to support non native BitVector
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ static Float64 convertFromUnsignedBV(JFS_NR_RM rm,
                                       const BitVector<BVWIDTH> bvValue) {
    return jfs_nr_convert_from_unsigned_bv_to_float64(rm, bvValue.data,
                                                      BVWIDTH);
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ static Float64 convertFromSignedBV(JFS_NR_RM rm,
                                     const BitVector<BVWIDTH> bvValue) {
    return jfs_nr_convert_from_signed_bv_to_float64(rm, bvValue.data, BVWIDTH);
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ BitVector<BVWIDTH> convertToUnsignedBV(JFS_NR_RM rm) const {
    return BitVector<BVWIDTH>(
        jfs_nr_float64_convert_to_unsigned_bv(rm, data, BVWIDTH));
  }
  template <uint64_t BVWIDTH,
            typename = typename std::enable_if<
                (BVWIDTH <= JFS_NR_BITVECTOR_TY_BITWIDTH)>::type>
  __device__ BitVector<BVWIDTH> convertToSignedBV(JFS_NR_RM rm) const {
    return BitVector<BVWIDTH>(
        jfs_nr_float64_convert_to_signed_bv(rm, data, BVWIDTH));
  }

  // Special constants
  __device__ static Float64 getPositiveInfinity() {
    return jfs_nr_float64_get_infinity(true);
  }
  __device__ static Float64 getNegativeInfinity() {
    return jfs_nr_float64_get_infinity(false);
  }
  __device__ static Float64 getPositiveZero() { return jfs_nr_float64_get_zero(true); }
  __device__ static Float64 getNegativeZero() { return jfs_nr_float64_get_zero(false); }
  __device__ static Float64 getNaN() { return jfs_nr_float64_get_nan(true); }

  // SMT-LIBv2 bit comparison
  __device__ bool operator==(const Float64& other) const {
    return jfs_nr_float64_smtlib_equals(data, other.data);
  }

  __device__ bool ieeeEquals(const Float64& other) const {
    return jfs_nr_float64_ieee_equals(data, other.data);
  }

  __device__ bool fplt(const Float64& other) const {
    return jfs_nr_float64_lt(data, other.data);
  }
  __device__ bool fpleq(const Float64& other) const {
    return jfs_nr_float64_leq(data, other.data);
  }
  __device__ bool fpgt(const Float64& other) const {
    return jfs_nr_float64_gt(data, other.data);
  }
  __device__ bool fpgeq(const Float64& other) const {
    return jfs_nr_float64_geq(data, other.data);
  }

  // Arithmetic
  __device__ Float64 abs() const { return jfs_nr_float64_abs(data); }
  __device__ Float64 neg() const { return jfs_nr_float64_neg(data); }
  __device__ Float64 add(JFS_NR_RM rm, const Float64& other) const {
    return jfs_nr_float64_add(rm, data, other.data);
  };
  __device__ Float64 sub(JFS_NR_RM rm, const Float64& other) const {
    return jfs_nr_float64_sub(rm, data, other.data);
  };
  __device__ Float64 mul(JFS_NR_RM rm, const Float64& other) const {
    return jfs_nr_float64_mul(rm, data, other.data);
  };
  __device__ Float64 div(JFS_NR_RM rm, const Float64& other) const {
    return jfs_nr_float64_div(rm, data, other.data);
  };
  __device__ Float64 fma(JFS_NR_RM rm, const Float64& b, const Float64& c) const {
    return jfs_nr_float64_fma(rm, data, b.data, c.data);
  };
  __device__ Float64 sqrt(JFS_NR_RM rm) const { return jfs_nr_float64_sqrt(rm, data); }
  __device__ Float64 rem(const Float64& other) const {
    return jfs_nr_float64_rem(data, other.data);
  };
  __device__ Float64 roundToIntegral(JFS_NR_RM rm) const {
    return jfs_nr_float64_round_to_integral(rm, data);
  }
  __device__ Float64 min(const Float64& other) const {
    return jfs_nr_float64_min(data, other.data);
  }
  __device__ Float64 max(const Float64& other) const {
    return jfs_nr_float64_max(data, other.data);
  }

  // Predicates
  __device__ bool isNormal() const { return jfs_nr_float64_is_normal(data); }
  __device__ bool isSubnormal() const { return jfs_nr_float64_is_subnormal(data); }
  __device__ bool isZero() const { return jfs_nr_float64_is_zero(data); }
  __device__ bool isInfinite() const { return jfs_nr_float64_is_infinite(data); }
  __device__ bool isPositive() const { return jfs_nr_float64_is_positive(data); }
  __device__ bool isNegative() const { return jfs_nr_float64_is_negative(data); }
  __device__ bool isNaN() const { return jfs_nr_float64_is_nan(data); }

  // For testing
  __device__ uint64_t getRawBits() const { return jfs_nr_float64_get_raw_bits(data); }
  __device__ jfs_nr_float64 getRawData() const { return data; }
};

template <uint64_t EB, uint64_t SB>
__device__ Float<EB, SB> makeFloatFrom(BufferRef<const uint8_t> buffer, uint64_t lowBit,
                            uint64_t highBit);

// Specialize for Float32
template <>
__device__ Float32 makeFloatFrom(BufferRef<const uint8_t> buffer, uint64_t lowBit,
                      uint64_t highBit);

// Specialize for Float64
template <>
__device__ Float64 makeFloatFrom(BufferRef<const uint8_t> buffer, uint64_t lowBit,
                      uint64_t highBit);
#endif
