// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
// Copyright 2024 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/bme.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

// Borrowed structure from 4x4 version
// In that version:
//  1. A loads 4 scalar values from along a column
//  2. B loads 4 LMUL=4 vectors
//  3. Sequence of scalar-vector products in outer product style (incrementing A along row direction, increment B for next row), until k=0

// With OPU you don't need this confusing
// Just issue OPACC with two vectors sequentially until k=0

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1vx1v__rvv(
    size_t mr,  // Number of rows
    size_t nc,  // Number of cols
    size_t kc,  // Inner dimension
    const int8_t* restrict a, // A Matrix
    size_t a_stride,
    const void* restrict w,   // B Matrix
    int8_t* restrict c,       // C Matrix
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict 1])
    //const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  // assert(mr <= 4); // No restriction; no hardcode scalar loads
  assert(nc != 0);
  assert(kc != 0);

  // Strict guideline to ensure full utilization of OPU
  // assert(mr != __riscv_vsetvlmax_e8m1()/8);
  // assert(nc != __riscv_vsetvlmax_e8m1()/8);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  // Need to check amount left for vector loads

  const size_t nr = __riscv_vsetvlmax_e8m1(); // MAX amount loaded from B row
  size_t vl = nr;

  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_zero_point = params->fp32_scalar.output_zero_point;
  const float output_min_less_zero_point_f = (float) output_min_less_zero_point;
  const float output_max_less_zero_point_f = (float) output_max_less_zero_point;
  const float output_zero_point_f = (float) output_zero_point;
  do {

    // No need set vl < MAXVL because OPU uses full vector
    // if XNN_UNLIKELY(nc < nr) {
    //   vl = __riscv_vsetvl_e32m4(nc);
    // }

    nc = nc - vl;

    // Load offset into accumulators (D in C = AB + D)
    __asm__ volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl));  // Element is 32-bit (maybe should be 8-bit)
    // TODO: sign extend int8 to int32 only if bias is NOT int32
    __asm__ volatile("vle32.v v0, (%0)" : : "r"((const int32_t*)w));

    // vint32m4_t vacc1 = vacc0;
    // vint32m4_t vacc2 = vacc0;
    // vint32m4_t vacc3 = vacc0;
    OPU_BCAST(m0, v0);


    // TODO: Zero the OPU
    // vint8m1_t zero_v = __riscv_vmv_v_x_i8m1(0, vl); // vl is MAX as of here
    // OPU_BCAST(m0, zero_v);

    w = (const int32_t*) w + nr;

    size_t k = kc;
    do {
      // Load vectors
      __asm__ volatile("vle8.v v0, (%0)" : : "r"((const int8_t*)a)); // Load A column
      __asm__ volatile("vle8.v v1, (%0)" : : "r"((const int8_t*)w)); // Load B row

      OPU_VOPACC(m0, v0, v1); // Execute outer product

      // Increment A and B ptrs
      w = (const int8_t*) w + nr;
      a = (const int8_t*) a + nr;

      k -= sizeof(int8_t);
    } while (k != 0);

    for (int mrf_r = 0; mrf_r < __riscv_vsetvlmax_e8m1(); mrf_r++) {
      // Set 32-bit elements, LMUL=4
      vl = __riscv_vsetvlmax_e32m4(); // MAX amount loaded from B row

      // Move vector out (whole MRF row)
      OPU_MVOUT(v29, mrf_r, m0);

      // Convert to floating point
      __asm__ volatile("vfcvt.f.x.v v29, v29 \n\t" : : );


      // Scale by channel scale factor
      __asm__ volatile("vle32.v v28, (%0)" : : "r"((const float*)w)); // Load channel scale factor
      __asm__ volatile("vfmul.vv v29, v28, v29 \n\t"
        :
        :
        : );   // Scale


      w = (const float*) w + nr;

      // Clamp with min + max
      __asm__ volatile(
        "vfmax.vf v29, v29, %[min_cmp] \n\t"
        "vfmin.vf v29, v29, %[max_cmp] \n\t"
        :
        : [min_cmp] "f" (output_min_less_zero_point_f),
          [max_cmp] "f" (output_max_less_zero_point_f)
        :
      );


      vl = __riscv_vsetvl_e16m2(vl); // Convert to element 16
      __asm__ volatile(
        "vsetvli %[vl], %[vl], e16, m2, ta, ma \n\t"
        "vfncvt.x.f.w v28, v28 \n\t"
        : [vl] "+r" (vl)
        :
        : "cc"
      ); // Convert fp32 to int16


      __asm__ volatile("vadd.vx v28, v28, %[zero_pt] \n\t"
        :
        : [zero_pt] "r" ((int16_t) output_zero_point)
        : "cc"
      );


      // Convert int16 -> int8
      __asm__ volatile(
        "vsetvli %[vl], %[vl], e8, m1, ta, ma \n\t"
        "vncvt.x.x.w v28, v28 \n\t"
        : [vl] "+r" (vl)
        :
        : );


      __asm__ volatile("vse8.v v28, (%[C_ptr]) \n\t"
        :
        : [C_ptr] "r" (c0)
        : "memory"
      );
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
    }

    // Undo pointer incrementing in outer product loop
    a0 = (const int8_t*) ((uintptr_t) a0 - kc);
    a1 = (const int8_t*) ((uintptr_t) a1 - kc);
    a2 = (const int8_t*) ((uintptr_t) a2 - kc);
    a3 = (const int8_t*) ((uintptr_t) a3 - kc);

  } while (nc != 0);
}
