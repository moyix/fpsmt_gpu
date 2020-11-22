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
// This is the implemenation of the runtime for SMTLIB BitVectors that
// uses native machine operations. It is written with a C compatible interface
// so that in the future we can easily use LLVM's JIT.

#include "SMTLIB/NativeBitVector.h"
#include "SMTLIB/jassert.h"

// Helper constants/functions
namespace {

const jfs_nr_width_ty jfs_nr_bitvector_ty_bit_width =
    sizeof(jfs_nr_bitvector_ty) * 8;

__device__ jfs_nr_bitvector_ty jfs_nr_get_bitvector_mask(const jfs_nr_width_ty bitWidth) {
  static_assert(jfs_nr_bitvector_ty_bit_width <= 64, "Wrong width");
  jassert(bitWidth <= jfs_nr_bitvector_ty_bit_width);
  return (bitWidth >= jfs_nr_bitvector_ty_bit_width)
             ? UINT64_MAX
             : ((UINT64_C(1) << bitWidth) - 1);
}


__device__ jfs_nr_bitvector_ty
jfs_nr_get_most_signficiant_bit_mask(const jfs_nr_width_ty bitWidth) {
  jassert(bitWidth <= jfs_nr_bitvector_ty_bit_width);
  return (UINT64_C(1) << (bitWidth - 1));
}

}

#ifdef __cplusplus
extern "C" {
#endif

// Public functions

__device__ jfs_nr_bitvector_ty jfs_nr_get_bitvector_mod(const jfs_nr_bitvector_ty value,
                                             const jfs_nr_width_ty bitWidth) {
  static_assert(jfs_nr_bitvector_ty_bit_width <= 64, "Wrong width");
  if (bitWidth >= jfs_nr_bitvector_ty_bit_width) {
    return value;
  } else {
    return value % (UINT64_C(1) << bitWidth);
  }
}

__device__ bool jfs_nr_is_valid(const jfs_nr_bitvector_ty value,
                     const jfs_nr_width_ty width) {
  return jfs_nr_get_bitvector_mod(value, width) == value;
}

__device__ jfs_nr_bitvector_ty jfs_nr_concat(const jfs_nr_bitvector_ty lhs,
                                  const jfs_nr_width_ty lhsBitWidth,
                                  const jfs_nr_bitvector_ty rhs,
                                  const jfs_nr_width_ty rhsBitWidth) {
  jassert(jfs_nr_is_valid(lhs, lhsBitWidth));
  jassert(jfs_nr_is_valid(rhs, rhsBitWidth));
  jassert(((lhsBitWidth + rhsBitWidth) <= jfs_nr_bitvector_ty_bit_width) &&
          "concat too wide");
  jfs_nr_bitvector_ty newValue = rhs;
  newValue |= (lhs << rhsBitWidth);
  return newValue;
}

// Extract bits [highBit, lowBit]
__device__ jfs_nr_bitvector_ty jfs_nr_extract(const jfs_nr_bitvector_ty value,
                                   const jfs_nr_width_ty bitWidth,
                                   const jfs_nr_width_ty highBit,
                                   const jfs_nr_width_ty lowBit) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert(highBit >= lowBit && "Invalid highBit and/or lowBit value");
  jassert(highBit < bitWidth && "Invalid highBit bit");
  jassert(lowBit < bitWidth && "Invalid lowBit bit");
  const jfs_nr_width_ty newWidth = (highBit - lowBit) + 1;
  if (newWidth == bitWidth)
    return value;
  jfs_nr_bitvector_ty temp = value;
  // Remove higher bits that we don't want
  jfs_nr_bitvector_ty mask = jfs_nr_get_bitvector_mask(highBit + 1);
  temp &= mask;

  // Remove lower bits that we don't want.
  temp >>= lowBit;
  jassert(jfs_nr_is_valid(temp, newWidth));
  return temp;
}

// Zero extend to bitvector (bitWidth + extraBits) wide
__device__ jfs_nr_bitvector_ty jfs_nr_zero_extend(const jfs_nr_bitvector_ty value,
                                       const jfs_nr_width_ty bitWidth,
                                       const jfs_nr_width_ty extraBits) {
  // No really work to do provided internal invariant that unused biits
  // are zero is maintained.
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert((bitWidth + extraBits) <= jfs_nr_bitvector_ty_bit_width);
  return value;
}

__device__ jfs_nr_bitvector_ty jfs_nr_sign_extend(const jfs_nr_bitvector_ty value,
                                       const jfs_nr_width_ty bitWidth,
                                       const jfs_nr_width_ty extraBits) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert((bitWidth + extraBits) <= jfs_nr_bitvector_ty_bit_width);
  if (value & jfs_nr_get_most_signficiant_bit_mask(bitWidth)) {
    // msb is not zero. Must do sign extend with ones.
    const jfs_nr_bitvector_ty currentWidthMask =
        jfs_nr_get_bitvector_mask(bitWidth);
    const jfs_nr_bitvector_ty newWidthMask =
        jfs_nr_get_bitvector_mask(bitWidth + extraBits);
    return (value | (~currentWidthMask)) & newWidthMask;
  } else {
    // Just do zero extend
    return jfs_nr_zero_extend(value, bitWidth, extraBits);
  }
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvneg(const jfs_nr_bitvector_ty value,
                                 const jfs_nr_width_ty bitWidth) {
  // [[(bvneg s)]] := nat2bv[m](2^m - bv2nat([[s]]))
  jassert(jfs_nr_is_valid(value, bitWidth));
  if (value == 0) {
    return 0;
  }

  // In two's complement, flipping bits and adding one negates
  // the number.
  return ((~value) & jfs_nr_get_bitvector_mask(bitWidth)) + 1;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvadd(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  // [[(bvadd s t)]] := nat2bv[m](bv2nat([[s]]) + bv2nat([[t]]))
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return jfs_nr_get_bitvector_mod(lhs + rhs, bitWidth);
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvsub(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  // (bvsub s t) abbreviates (bvadd s (bvneg t))
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  // TODO: Verify the implementation is semantically equivalent
  // to SMT-LIBv2
  return jfs_nr_get_bitvector_mod(lhs - rhs, bitWidth);
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvmul(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  // [[(bvmul s t)]] := nat2bv[m](bv2nat([[s]]) * bv2nat([[t]]))
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return jfs_nr_get_bitvector_mod(lhs * rhs, bitWidth);
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvudiv(const jfs_nr_bitvector_ty dividend,
                                  const jfs_nr_bitvector_ty divisor,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(dividend, bitWidth));
  jassert(jfs_nr_is_valid(divisor, bitWidth));
  //   [[(bvudiv s t)]] := if bv2nat([[t]]) = 0
  //                       then λx:[0, m). 1
  //                       else nat2bv[m](bv2nat([[s]]) div bv2nat([[t]]))
  if (divisor == 0) {
    return jfs_nr_get_bitvector_mask(bitWidth);
  }
  jfs_nr_bitvector_ty result = dividend / divisor;
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvurem(const jfs_nr_bitvector_ty dividend,
                                  const jfs_nr_bitvector_ty divisor,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(dividend, bitWidth));
  jassert(jfs_nr_is_valid(divisor, bitWidth));
  //  [[(bvurem s t)]] := if bv2nat([[t]]) = 0
  //                      then [[s]]
  //                      else nat2bv[m](bv2nat([[s]]) rem bv2nat([[t]]))
  jfs_nr_bitvector_ty result = 0;
  if (divisor == 0) {
    result = dividend;
  } else {
    result = dividend % divisor;
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvsdiv(const jfs_nr_bitvector_ty dividend,
                                  const jfs_nr_bitvector_ty divisor,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(dividend, bitWidth));
  jassert(jfs_nr_is_valid(divisor, bitWidth));
  // (bvsdiv s t) abbreviates
  //  (let ((?msb_s ((_ extract |m-1| |m-1|) s))
  //        (?msb_t ((_ extract |m-1| |m-1|) t)))
  //    (ite (and (= ?msb_s #b0) (= ?msb_t #b0))
  //         (bvudiv s t)
  //    (ite (and (= ?msb_s #b1) (= ?msb_t #b0))
  //         (bvneg (bvudiv (bvneg s) t))
  //    (ite (and (= ?msb_s #b0) (= ?msb_t #b1))
  //         (bvneg (bvudiv s (bvneg t)))
  //         (bvudiv (bvneg s) (bvneg t))))))
  const jfs_nr_bitvector_ty msbMask =
      jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  bool msb_s = dividend & msbMask;
  bool msb_t = divisor & msbMask;
  jfs_nr_bitvector_ty result = 0;
  // TODO: Can we write this more efficiently?
  if (!msb_s && !msb_t) {
    // Both operands are postive in two's complement
    result = jfs_nr_bvudiv(dividend, divisor, bitWidth);
  } else if (msb_s && !msb_t) {
    // lhs is negative and rhs is positive in two's complement
    result = jfs_nr_bvneg(
        jfs_nr_bvudiv(jfs_nr_bvneg(dividend, bitWidth), divisor, bitWidth),
        bitWidth);
  } else if (!msb_s && msb_t) {
    // lhs is positive and rhs is negative in two's complement
    result = jfs_nr_bvneg(
        jfs_nr_bvudiv(dividend, jfs_nr_bvneg(divisor, bitWidth), bitWidth),
        bitWidth);
  } else {
    // Both operands are negative in two's complement
    result = jfs_nr_bvudiv(jfs_nr_bvneg(dividend, bitWidth),
                           jfs_nr_bvneg(divisor, bitWidth), bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvsrem(const jfs_nr_bitvector_ty dividend,
                                  const jfs_nr_bitvector_ty divisor,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(dividend, bitWidth));
  jassert(jfs_nr_is_valid(divisor, bitWidth));
  // 2's complement signed remainder (sign follows dividend)
  // (bvsrem s t) abbreviates
  //  (let ((?msb_s ((_ extract |m-1| |m-1|) s))
  //        (?msb_t ((_ extract |m-1| |m-1|) t)))
  //    (ite (and (= ?msb_s #b0) (= ?msb_t #b0))
  //         (bvurem s t)
  //    (ite (and (= ?msb_s #b1) (= ?msb_t #b0))
  //         (bvneg (bvurem (bvneg s) t))
  //    (ite (and (= ?msb_s #b0) (= ?msb_t #b1))
  //         (bvurem s (bvneg t)))
  //         (bvneg (bvurem (bvneg s) (bvneg t))))))
  const jfs_nr_bitvector_ty msbMask =
      jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  bool msb_s = dividend & msbMask;
  bool msb_t = divisor & msbMask;
  jfs_nr_bitvector_ty result = 0;
  // TODO: Can we write this more efficiently?
  if (!msb_s && !msb_t) {
    // Both operands are postive in two's complement
    result = jfs_nr_bvurem(dividend, divisor, bitWidth);
  } else if (msb_s && !msb_t) {
    // lhs is negative and rhs is positive in two's complement
    result = jfs_nr_bvneg(
        jfs_nr_bvurem(jfs_nr_bvneg(dividend, bitWidth), divisor, bitWidth),
        bitWidth);
  } else if (!msb_s && msb_t) {
    // lhs is positive and rhs is negative in two's complement
    result = jfs_nr_bvurem(dividend, jfs_nr_bvneg(divisor, bitWidth), bitWidth);
  } else {
    // Both operands are negative in two's complement
    result =
        jfs_nr_bvneg(jfs_nr_bvurem(jfs_nr_bvneg(dividend, bitWidth),
                                   jfs_nr_bvneg(divisor, bitWidth), bitWidth),
                     bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvsmod(const jfs_nr_bitvector_ty dividend,
                                  const jfs_nr_bitvector_ty divisor,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(dividend, bitWidth));
  jassert(jfs_nr_is_valid(divisor, bitWidth));
  // 2's complement signed remainder (sign follows divisor)
  // (bvsmod s t) abbreviates
  //  (let ((?msb_s ((_ extract |m-1| |m-1|) s))
  //        (?msb_t ((_ extract |m-1| |m-1|) t)))
  //    (let ((abs_s (ite (= ?msb_s #b0) s (bvneg s)))
  //          (abs_t (ite (= ?msb_t #b0) t (bvneg t))))
  //      (let ((u (bvurem abs_s abs_t)))
  //        (ite (= u (_ bv0 m))
  //             u
  //        (ite (and (= ?msb_s #b0) (= ?msb_t #b0))
  //             u
  //        (ite (and (= ?msb_s #b1) (= ?msb_t #b0))
  //             (bvadd (bvneg u) t)
  //        (ite (and (= ?msb_s #b0) (= ?msb_t #b1))
  //             (bvadd u t)
  //             (bvneg u))))))))
  // TODO: Can we write this more efficiently?
  const jfs_nr_bitvector_ty msbMask =
      jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  bool msb_s = dividend & msbMask;
  bool msb_t = divisor & msbMask;
  jfs_nr_bitvector_ty abs_s =
      msb_s ? jfs_nr_bvneg(dividend, bitWidth) : dividend;
  jfs_nr_bitvector_ty abs_t = msb_t ? jfs_nr_bvneg(divisor, bitWidth) : divisor;
  jfs_nr_bitvector_ty u = jfs_nr_bvurem(abs_s, abs_t, bitWidth);
  jfs_nr_bitvector_ty result = 0;
  if (u == 0) {
    result = u;
  } else if (!msb_s && !msb_t) {
    // Both operands are postive in two's complement
    result = u;
  } else if (msb_s && !msb_t) {
    // lhs is negative and rhs is positive in two's complement
    result = jfs_nr_bvadd(jfs_nr_bvneg(u, bitWidth), divisor, bitWidth);
  } else if (!msb_s && msb_t) {
    // lhs is positive and rhs is negative in two's complement
    result = jfs_nr_bvadd(u, divisor, bitWidth);
  } else {
    // Both operands are negative in two's complement
    result = jfs_nr_bvneg(u, bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvshl(const jfs_nr_bitvector_ty value,
                                 const jfs_nr_bitvector_ty shift,
                                 const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert(jfs_nr_is_valid(shift, bitWidth));
  //  [[(bvshl s t)]] := nat2bv[m](bv2nat([[s]]) * 2^(bv2nat([[t]])))
  jfs_nr_bitvector_ty result = 0;
  if (shift >= bitWidth) {
    // Overshift to zero
    result = 0;
  } else {
    result = jfs_nr_get_bitvector_mod(value << shift, bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvlshr(const jfs_nr_bitvector_ty value,
                                  const jfs_nr_bitvector_ty shift,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert(jfs_nr_is_valid(shift, bitWidth));
  // [[(bvlshr s t)]] := nat2bv[m](bv2nat([[s]]) div 2^(bv2nat([[t]])))
  jfs_nr_bitvector_ty result = 0;
  if (shift >= bitWidth) {
    // Overshift to zero
    result = 0;
  } else {
    result = jfs_nr_get_bitvector_mod(value >> shift, bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvashr(const jfs_nr_bitvector_ty value,
                                  const jfs_nr_bitvector_ty shift,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jassert(jfs_nr_is_valid(shift, bitWidth));
  // (bvashr s t) abbreviates
  //  (ite (= ((_ extract |m-1| |m-1|) s) #b0)
  //       (bvlshr s t)
  //       (bvnot (bvlshr (bvnot s) t)))
  // TODO: Can we do this more efficiently?
  bool lhsNeg = value & jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  jfs_nr_bitvector_ty result = 0;
  if (!lhsNeg) {
    result = jfs_nr_bvlshr(value, shift, bitWidth);
  } else {
    result = jfs_nr_bvnot(
        jfs_nr_bvlshr(jfs_nr_bvnot(value, bitWidth), shift, bitWidth),
        bitWidth);
  }
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_rotate_left(const jfs_nr_bitvector_ty value,
                                       const jfs_nr_width_ty shift,
                                       const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  // ((_ rotate_left 0) t) stands for t
  // ((_ rotate_left i) t) abbreviates t if m = 1, and
  //   ((_ rotate_left |i-1|)
  //     (concat ((_ extract |m-2| 0) t) ((_ extract |m-1| |m-1|) t))
  //   otherwise
  jfs_nr_bitvector_ty effective_shift = shift % bitWidth;
  // Shift bits to the left
  jassert(effective_shift < jfs_nr_bitvector_ty_bit_width);
  jfs_nr_bitvector_ty result = value << effective_shift;
  // bitwise or with the bits that got shifted out and
  // should now appear in the lsb.
  jassert((bitWidth - effective_shift) < jfs_nr_bitvector_ty_bit_width);
  result |= value >> (bitWidth - effective_shift);
  result &= jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_rotate_right(const jfs_nr_bitvector_ty value,
                                        const jfs_nr_width_ty shift,
                                        const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  // ((_ rotate_right 0) t) stands for t
  // ((_ rotate_right i) t) abbreviates t if m = 1, and
  //   ((_ rotate_right |i-1|)
  //     (concat ((_ extract 0 0) t) ((_ extract |m-1| 1) t)))
  //   otherwise
  jfs_nr_bitvector_ty effective_shift = shift % bitWidth;
  // Shift bits to the right
  jassert(effective_shift < jfs_nr_bitvector_ty_bit_width);
  jfs_nr_bitvector_ty result = value >> effective_shift;
  // bitwise or with the bits that got shifted out and
  // should now appear in the msb.
  jassert((bitWidth - effective_shift) < jfs_nr_bitvector_ty_bit_width);
  result |= value << (bitWidth - effective_shift);
  result &= jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvand(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result = lhs & rhs;
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvor(const jfs_nr_bitvector_ty lhs,
                                const jfs_nr_bitvector_ty rhs,
                                const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result = lhs | rhs;
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvnand(const jfs_nr_bitvector_ty lhs,
                                  const jfs_nr_bitvector_ty rhs,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result =
      (~(lhs & rhs)) & jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvnor(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result =
      (~(lhs | rhs)) & jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvxor(const jfs_nr_bitvector_ty lhs,
                                 const jfs_nr_bitvector_ty rhs,
                                 const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result =
      (lhs ^ rhs) & jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvxnor(const jfs_nr_bitvector_ty lhs,
                                  const jfs_nr_bitvector_ty rhs,
                                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  jfs_nr_bitvector_ty result =
      (~(lhs ^ rhs)) & jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ jfs_nr_bitvector_ty jfs_nr_bvnot(const jfs_nr_bitvector_ty value,
                                 const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(value, bitWidth));
  jfs_nr_bitvector_ty result = (~value) & jfs_nr_get_bitvector_mask(bitWidth);
  jassert(jfs_nr_is_valid(result, bitWidth));
  return result;
}

__device__ bool jfs_nr_bvult(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return lhs < rhs;
}

__device__ bool jfs_nr_bvule(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return lhs <= rhs;
}

__device__ bool jfs_nr_bvugt(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return lhs > rhs;
}

__device__ bool jfs_nr_bvuge(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  return lhs >= rhs;
}

__device__ bool jfs_nr_bvslt(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  // (bvslt s t) abbreviates:
  //  (or (and (= ((_ extract |m-1| |m-1|) s) #b1)
  //           (= ((_ extract |m-1| |m-1|) t) #b0))
  //      (and (= ((_ extract |m-1| |m-1|) s) ((_ extract |m-1| |m-1|) t))
  //           (bvult s t)))
  bool lhsNeg = lhs & jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  bool rhsNeg = rhs & jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  if (lhsNeg && !rhsNeg) {
    return true;
  }
  return (lhsNeg == rhsNeg) && jfs_nr_bvult(lhs, rhs, bitWidth);
}

__device__ bool jfs_nr_bvsle(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  jassert(jfs_nr_is_valid(lhs, bitWidth));
  jassert(jfs_nr_is_valid(rhs, bitWidth));
  // (bvsle s t) abbreviates:
  //  (or (and (= ((_ extract |m-1| |m-1|) s) #b1)
  //           (= ((_ extract |m-1| |m-1|) t) #b0))
  //      (and (= ((_ extract |m-1| |m-1|) s) ((_ extract |m-1| |m-1|) t))
  //           (bvule s t)))
  bool lhsNeg = lhs & jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  bool rhsNeg = rhs & jfs_nr_get_most_signficiant_bit_mask(bitWidth);
  if (lhsNeg && !rhsNeg) {
    return true;
  }
  return (lhsNeg == rhsNeg) && jfs_nr_bvule(lhs, rhs, bitWidth);
}

__device__ bool jfs_nr_bvsgt(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  // (bvsgt s t) abbreviates (bvslt t s)
  return jfs_nr_bvslt(rhs, lhs, bitWidth);
}

__device__ bool jfs_nr_bvsge(const jfs_nr_bitvector_ty lhs, const jfs_nr_bitvector_ty rhs,
                  const jfs_nr_width_ty bitWidth) {
  // (bvsge s t) abbreviates (bvsle t s)
  return jfs_nr_bvsle(rhs, lhs, bitWidth);
}

// FIXME: We should either handle byte aligned access specially, or provide
// another function for this.
// Convenience function for creating a BitVector
// from any arbitrary bit offset in a buffer. Offset
// is [lowbit, highbit].
__device__ jfs_nr_bitvector_ty jfs_nr_make_bitvector(const uint8_t* bufferData,
                                          const uint64_t bufferSize,
                                          const uint64_t lowBit,
                                          const uint64_t highBit) {
  jassert(highBit >= lowBit && "invalid lowBit and highBit");
  jassert(highBit < (bufferSize * 8));
  const uint64_t bitWidth = ((highBit - lowBit) + 1);
  const size_t lowBitByte = lowBit / 8;
  const size_t shiftOffset = lowBit % 8;
  // NOTE: doing `highBit / 8` to compute `highBitByte` is wrong. For [1,8]
  // that gives a highBit of 1 which is wrong for the loop below (should be 0).
  // const size_t highBitByte = (lowBitByte + ((BITWIDTH + 7) / 8)) - 1;
  const size_t highBitByte = (lowBitByte + (((highBit - lowBit) + 8) / 8)) - 1;
  jassert(lowBitByte < bufferSize);
  jassert(highBitByte < bufferSize);
  jfs_nr_bitvector_ty data = 0;
  uint8_t* dataView = reinterpret_cast<uint8_t*>(&data);
  jfs_nr_bitvector_ty dataMask = jfs_nr_get_bitvector_mask(bitWidth);
  // Copy byte-by-byte shifting if necessary
  for (size_t index = lowBitByte; index <= highBitByte; ++index) {
    const size_t viewIndex = index - lowBitByte;
    jassert(index < bufferSize);
    jassert(viewIndex < sizeof(data));
    uint8_t bufferByte = bufferData[index];
    dataView[viewIndex] |= (bufferByte >> shiftOffset);
    if (shiftOffset == 0) {
      // If there is no shift offset then we didn't shift any bits
      // out.
      continue;
    }
    // Doing the shift means we have zero bits in MSB rather than the actually
    // bits we want.
    uint8_t nextIterByteValue = 0;
    if ((index + 1) < bufferSize) {
      // Avoid out of bounds access
      nextIterByteValue = bufferData[index + 1];
    }
    dataView[viewIndex] |= (nextIterByteValue << (8 - shiftOffset));
  }
  // Now mask off the data
  data &= dataMask;
  return data;
}

__device__ void jfs_nr_write_bitvector(const jfs_nr_bitvector_ty bv,
                            const jfs_nr_width_ty bitWidth,
                            uint8_t* bufferData,
                            const uint64_t bufferSize,
                            const uint64_t bitOffset) {
  // Copy the bit vector so we can shift bits off as we go.
  // TODO: Or just mark it not const in the params...?
  // FIXME: Rework this algorithm to avoid the copy and shifting here if it
  // appears to be a bottleneck. (It seems okay since bit vectors are a small,
  // fixed size.)
  jfs_nr_bitvector_ty tempBv = bv;
  jassert(jfs_nr_is_valid(tempBv, bitWidth));
  const uint64_t endBit = bitOffset + bitWidth - 1;
  jassert(bufferData != nullptr);
  jassert(bitOffset < (bufferSize * 8));
  jassert(endBit < (bufferSize * 8));
  const size_t startByte = bitOffset / 8;
  const size_t endByte = endBit / 8;
  jassert(startByte < bufferSize);
  jassert(endByte < bufferSize);
  // Track the current bit offset over time as well as the bits left to write.
  uint64_t currentBitOffset = bitOffset;
  jfs_nr_width_ty bitsRemaining = bitWidth;
  // Track a view of the first byte of the bit vector.
  const uint8_t* bvByteView = reinterpret_cast<const uint8_t*>(&tempBv);
  // Write in the buffer byte by byte while preserving surrounding bits.
  for (size_t currentByte = startByte; currentByte <= endByte; ++currentByte) {
    // Example: `currentBitOffset` is 6. We can take bits [1, 0] from the BV and
    // write them to bits [7, 6] in buffer.
    const size_t offsetInByte = currentBitOffset % 8;
    size_t bitsToWrite = 8 - offsetInByte;
    // For the end byte, we have to cap `bitsToWrite` at the amount remaining.
    if (bitsToWrite > bitsRemaining) {
      bitsToWrite = bitsRemaining;
    }
    jassert(currentBitOffset < (bufferSize * 8));
    jassert(currentBitOffset + bitsToWrite - 1 <= endBit);
    const uint8_t bvMask = jfs_nr_get_bitvector_mask(bitsToWrite);
    const uint8_t bufferMask = ~(bvMask << offsetInByte);
    bufferData[currentByte] = (bufferData[currentByte] & bufferMask) |
                              ((*bvByteView & bvMask) << offsetInByte);
    // Shift the bits written off the bit vector.
    tempBv >>= bitsToWrite;
    currentBitOffset += bitsToWrite;
    bitsRemaining -= bitsToWrite;
  }
}

#ifdef __cplusplus
}
#endif
