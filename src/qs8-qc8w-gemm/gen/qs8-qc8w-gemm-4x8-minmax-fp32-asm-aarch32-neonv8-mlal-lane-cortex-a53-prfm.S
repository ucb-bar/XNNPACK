// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "src/xnnpack/assembly.h"

.syntax unified

// void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            (r2) -> sp + 56 -> r5
//     const int8_t* restrict a,              r3
//     size_t a_stride,           sp + 96 -> (r7)
//     const void* restrict w,     sp + 100 -> r9
//     int8_t* restrict c,         sp + 104 -> r11
//     size_t cm_stride,          sp + 108 -> (r6)
//     size_t cn_stride,          sp + 112 -> r7
//     xnn_qs8_qc8w_conv_minmax_params params)  sp + 116 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0-d1 q0
// A1  r12  d2-d3 q1
// A2  r10  d4-d5 q2
// A3   r0  d6-d7 q3
// B    r9  d8-d9 q4 q5
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15

// r2,r14 A53 gpr temporary loads
// unused d15

// params structure is 4 bytes
//  struct {
//    int16_t output_zero_point;  d13[2]
//    int8_t output_min;          d13[6]
//    int8_t output_max;          d13[7]
//  } xnn_qs8_minmax_params.neonv8;

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm
        # Push 96 bytes
        PUSH        {r2, r4, r5, r6, r7, r8, r9, r10, r11, lr}  // 40
        SUB         sp, sp, 8                           // +8
        VPUSH       {d8-d13}                            // +48 = 96

        LDR         r7, [sp, 96]            // a_stride
        LDR         r11, [sp, 104]          // c
        LDR         r6, [sp, 108]           // cm_stride
        LDR         r9, [sp, 100]           // w
        LDR         r5, [sp, 116]           // params

        # Clamp A and C pointers
        CMP         r0, 2                   // if mr >= 2
        ADD         r12, r3, r7             //   a1 = a0 + a_stride
        ADD         r4, r11, r6             //   c1 = c0 + cm_stride
        MOVLO       r12, r3                 // a1
        MOVLO       r4, r11                 // c1
                                        // if mr > 2
        ADD         r10, r12, r7            //   a2 = a1 + a_stride
        ADD         r8, r4, r6              //   c2 = c1 + cm_stride
        MOVLS       r10, r12                // a2
        MOVLS       r8, r4                  // c2

        CMP         r0, 4                   // if mr >=4
        ADD         r0, r10, r7             //   a3 = a2 + a_stride
        ADD         r6, r8, r6              //   c3 = c2 + cm_stride
        MOVLO       r0, r10                 // a3
        MOVLO       r6, r8                  // c3

        # Load params values
        VLD1.32     {d13[]}, [r5]           // QC8 neonv8 params
        LDR         r7, [sp, 112]               // cn_stride

        PLD         [r9,  64]               // Prefetch B
        PLD         [r9, 128]
        PLD         [r9, 192]
        PLD         [r9, 256]
        PLD         [r9, 320]
        PLD         [r9, 384]

        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias
        SUBS        r5, r2, 8               // k = kc - 8

        VMOV        q10, q8
        PLD         [r3,  64]               // Prefetch A
        VMOV        q11, q9
        PLD         [r12, 64]
        VMOV        q12, q8
        PLD         [r10, 64]
        VMOV        q13, q9
        PLD         [r0,  64]
        VMOV        q14, q8
        VMOV        q15, q9
        BLO         4f                      // less than 8 channels?

        // Prologue - load 4A's and B0
        VLD1.8      {d0},  [r3]!            // A0
        VLD1.8      {d2}, [r12]!            // A1
        VLD1.8      {d4}, [r10]!            // A2
        VLD1.8      {d6},  [r0]!            // A3
        VLD1.8      {d8},  [r9]!            // B0

        SUBS        r5, r5, 8               // k = k - 8
        BLO         2f                      // less than 8 channels?

        // Main loop - 8 bytes
        // 64 bytes for weights.
        // 5 VMOVL = 4 A and 1 B = 5 cycles
        // 7 blocks with VLD B, VMOVL, 8 VMLA = 10 cycles
        // 1 blocks with VLD B, VMLA = 9 cycles
        // total = 84 cycles
        .p2align    3
1:
        // Extend - 5 cycles
        VMOVL.S8 q0, d0
        PLD         [r3, 128]
        VMOVL.S8    q4, d8
        PLD         [r9, 448]
        VMOVL.S8 q1, d2
        PLD         [r12, 128]
        VMOVL.S8 q2, d4
        PLD         [r0, 128]
        VMOVL.S8 q3, d6
        PLD         [r10, 128]

        // BLOCK 0 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]

        // BLOCK 1 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q8, d10, d0[1]
        VMLAL.S16   q9, d11, d0[1]
        VMLAL.S16   q10, d10, d2[1]
        VMLAL.S16   q11, d11, d2[1]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d4[1]
        VMLAL.S16   q13, d11, d4[1]
        VMLAL.S16   q14, d10, d6[1]
        VMLAL.S16   q15, d11, d6[1]

        // BLOCK 2 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]

        // BLOCK 3 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q8, d10, d0[3]
        VMLAL.S16   q9, d11, d0[3]
        VMLAL.S16   q10, d10, d2[3]
        VMLAL.S16   q11, d11, d2[3]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d4[3]
        LDR         r2, [r3]                // A0 low
        VMLAL.S16   q13, d11, d4[3]
        LDR         r14, [r3, 4]            // A0 high
        VMLAL.S16   q14, d10, d6[3]
        ADD         r3, r3, 8
        VMLAL.S16   q15, d11, d6[3]

        // BLOCK 4 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B5
        VMOV        d0, r2, r14             // A0 VMOV
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d5[0]
        LDR         r2, [r12]               // A1 low
        VMLAL.S16   q13, d9, d5[0]
        LDR         r14, [r12, 4]           // A1 high
        VMLAL.S16   q14, d8, d7[0]
        ADD         r12, r12, 8
        VMLAL.S16   q15, d9, d7[0]

        // BLOCK 5 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B6
        VMOV        d2, r2, r14             // A1 VMOV
        VMLAL.S16   q8, d10, d1[1]
        VMLAL.S16   q9, d11, d1[1]
        VMLAL.S16   q10, d10, d3[1]
        VMLAL.S16   q11, d11, d3[1]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d5[1]
        LDR         r2, [r10]               // A2 low
        VMLAL.S16   q13, d11, d5[1]
        LDR         r14, [r10, 4]           // A2 high
        VMLAL.S16   q14, d10, d7[1]
        ADD         r10, r10, 8
        VMLAL.S16   q15, d11, d7[1]

        // BLOCK 6 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B7
        VMOV        d4, r2, r14             // A2 VMOV
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d5[2]
        LDR         r2, [r0]                // A3 low
        VMLAL.S16   q13, d9, d5[2]
        LDR         r14, [r0, 4]            // A3 high
        VMLAL.S16   q14, d8, d7[2]
        ADD         r0, r0, 8
        VMLAL.S16   q15, d9, d7[2]

        // BLOCK 7 - 9 cycles
        VLD1.8      {d8},  [r9]!            // B0
        VMOV        d6, r2, r14             // A3 VMOV
        VMLAL.S16   q8, d10, d1[3]
        VMLAL.S16   q9, d11, d1[3]
        VMLAL.S16   q10, d10, d3[3]
        VMLAL.S16   q11, d11, d3[3]
        VMLAL.S16   q12, d10, d5[3]
        VMLAL.S16   q13, d11, d5[3]
        SUBS        r5, r5, 8
        VMLAL.S16   q14, d10, d7[3]
        VMLAL.S16   q15, d11, d7[3]
        BHS         1b

        // Epilogue

        .p2align    3
2:
        VMOVL.S8 q0, d0
        VMOVL.S8    q4, d8
        VMOVL.S8 q1, d2
        VMOVL.S8 q2, d4
        VMOVL.S8 q3, d6

        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]

        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q8, d10, d0[1]
        VMLAL.S16   q9, d11, d0[1]
        VMLAL.S16   q10, d10, d2[1]
        VMLAL.S16   q11, d11, d2[1]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d4[1]
        VMLAL.S16   q13, d11, d4[1]
        VMLAL.S16   q14, d10, d6[1]
        VMLAL.S16   q15, d11, d6[1]

        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]

        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q8, d10, d0[3]
        VMLAL.S16   q9, d11, d0[3]
        VMLAL.S16   q10, d10, d2[3]
        VMLAL.S16   q11, d11, d2[3]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d4[3]
        VMLAL.S16   q13, d11, d4[3]
        VMLAL.S16   q14, d10, d6[3]
        VMLAL.S16   q15, d11, d6[3]

        VLD1.8      {d10},  [r9]!           // B5
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d5[0]
        VMLAL.S16   q13, d9, d5[0]
        VMLAL.S16   q14, d8, d7[0]
        VMLAL.S16   q15, d9, d7[0]

        VLD1.8      {d8},  [r9]!            // B6
        VMLAL.S16   q8, d10, d1[1]
        VMLAL.S16   q9, d11, d1[1]
        VMLAL.S16   q10, d10, d3[1]
        VMLAL.S16   q11, d11, d3[1]
        VMOVL.S8    q4, d8
        VMLAL.S16   q12, d10, d5[1]
        VMLAL.S16   q13, d11, d5[1]
        VMLAL.S16   q14, d10, d7[1]
        VMLAL.S16   q15, d11, d7[1]

        VLD1.8      {d10},  [r9]!           // B7
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VMOVL.S8    q5, d10
        VMLAL.S16   q12, d8, d5[2]
        VMLAL.S16   q13, d9, d5[2]
        VMLAL.S16   q14, d8, d7[2]
        VMLAL.S16   q15, d9, d7[2]

        VMLAL.S16   q8, d10, d1[3]
        VMLAL.S16   q9, d11, d1[3]
        VMLAL.S16   q10, d10, d3[3]
        VMLAL.S16   q11, d11, d3[3]
        VMLAL.S16   q12, d10, d5[3]
        VMLAL.S16   q13, d11, d5[3]
        ADDS        r5, r5, 8
        VMLAL.S16   q14, d10, d7[3]
        VMLAL.S16   q15, d11, d7[3]

        # Is there a remainder?- 1-7 bytes of A
        BNE         4f

3:
        # QC8 FP32 quantization
        VLD1.8      {q0-q1},  [r9]!

        VCVT.F32.S32    q8,  q8
        VCVT.F32.S32    q9,  q9
        VCVT.F32.S32    q10, q10
        VCVT.F32.S32    q11, q11
        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13
        VCVT.F32.S32    q14, q14
        VCVT.F32.S32    q15, q15

        VMUL.F32    q8,  q8, q0             // multiplier
        VMUL.F32    q9,  q9, q1
        VMUL.F32    q10, q10, q0
        VMUL.F32    q11, q11, q1
        VMUL.F32    q12, q12, q0
        VMUL.F32    q13, q13, q1
        VMUL.F32    q14, q14, q0
        VMUL.F32    q15, q15, q1

        VCVTN.S32.F32   q8,  q8
        VCVTN.S32.F32   q9,  q9
        VCVTN.S32.F32   q10, q10
        VCVTN.S32.F32   q11, q11
        VCVTN.S32.F32   q12, q12
        VCVTN.S32.F32   q13, q13
        VCVTN.S32.F32   q14, q14
        VCVTN.S32.F32   q15, q15

        VDUP.16     q0, d13[2]              // output_zero_point

        VQMOVN.S32  d16, q8
        VQMOVN.S32  d17, q9
        VQMOVN.S32  d18, q10
        VQMOVN.S32  d19, q11
        VQMOVN.S32  d20, q12
        VQMOVN.S32  d21, q13
        VQMOVN.S32  d22, q14
        VQMOVN.S32  d23, q15

        VQADD.S16   q8,  q8, q0
        VQADD.S16   q9,  q9, q0
        VQADD.S16   q10, q10, q0
        VQADD.S16   q11, q11, q0

        VDUP.8      q12, d13[6]             // output_min

        VQMOVN.S16 d0,  q8
        VQMOVN.S16 d1,  q9
        VQMOVN.S16 d2, q10
        VQMOVN.S16 d3, q11

        VDUP.8      q13, d13[7]             // output_max

        VMAX.S8 q0, q0, q12
        VMAX.S8 q1, q1, q12

        LDR         r2, [sp, 56]            // kc
        SUBS        r1, r1, 8

        VMIN.S8 q0, q0, q13
        VMIN.S8 q1, q1, q13

        # Store full 4 x 8
        BLO         5f
        VST1.8      {d0}, [r11], r7
        SUB         r3, r3, r2
        VST1.8      {d1}, [r4], r7
        SUB         r12, r12, r2
        VST1.8      {d2}, [r8], r7
        SUB         r10, r10, r2
        VST1.8      {d3}, [r6], r7
        SUB         r0, r0, r2
        BHI         0b

        VPOP        {d8-d13}
        ADD         sp, sp, 12              // skip pad of 8 + r2
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

        # Remainder- 1 to 7 bytes of A
        .p2align    3
4:
        AND         r5, r5, 7               // kc remainder 1 to 7

        VLD1.8      {d0},  [r3], r5
        VLD1.8      {d8},  [r9]!
        VLD1.8      {d2}, [r12], r5
        VLD1.8      {d4}, [r10], r5
        VLD1.8      {d6},  [r0], r5

        VMOVL.S8 q0, d0
        VMOVL.S8    q4, d8
        VMOVL.S8 q1, d2
        VMOVL.S8 q2, d4
        VMOVL.S8 q3, d6
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]
        CMP         r5, 2
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d0[1]
        VMLAL.S16   q9, d9, d0[1]
        VMLAL.S16   q10, d8, d2[1]
        VMLAL.S16   q11, d9, d2[1]
        VMLAL.S16   q12, d8, d4[1]
        VMLAL.S16   q13, d9, d4[1]
        VMLAL.S16   q14, d8, d6[1]
        VMLAL.S16   q15, d9, d6[1]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]
        CMP         r5, 4
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d0[3]
        VMLAL.S16   q9, d9, d0[3]
        VMLAL.S16   q10, d8, d2[3]
        VMLAL.S16   q11, d9, d2[3]
        VMLAL.S16   q12, d8, d4[3]
        VMLAL.S16   q13, d9, d4[3]
        VMLAL.S16   q14, d8, d6[3]
        VMLAL.S16   q15, d9, d6[3]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VMLAL.S16   q12, d8, d5[0]
        VMLAL.S16   q13, d9, d5[0]
        VMLAL.S16   q14, d8, d7[0]
        VMLAL.S16   q15, d9, d7[0]
        CMP         r5, 6
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d1[1]
        VMLAL.S16   q9, d9, d1[1]
        VMLAL.S16   q10, d8, d3[1]
        VMLAL.S16   q11, d9, d3[1]
        VMLAL.S16   q12, d8, d5[1]
        VMLAL.S16   q13, d9, d5[1]
        VMLAL.S16   q14, d8, d7[1]
        VMLAL.S16   q15, d9, d7[1]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VMOVL.S8    q4, d8
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VMLAL.S16   q12, d8, d5[2]
        VMLAL.S16   q13, d9, d5[2]
        VMLAL.S16   q14, d8, d7[2]
        VMLAL.S16   q15, d9, d7[2]
        B           3b

        # Store odd width
        .p2align    3
5:
        TST         r1, 4
        BEQ         6f
        VST1.32     {d0[0]}, [r11]!
        VST1.32     {d1[0]}, [r4]!
        VST1.32     {d2[0]}, [r8]!
        VST1.32     {d3[0]}, [r6]!
        VEXT.8      q0, q0, q0, 4
        VEXT.8      q1, q1, q1, 4
6:
        TST         r1, 2
        BEQ         7f
        VST1.16     {d0[0]}, [r11]!
        VST1.16     {d1[0]}, [r4]!
        VST1.16     {d2[0]}, [r8]!
        VST1.16     {d3[0]}, [r6]!
        VEXT.8      q0, q0, q0, 2
        VEXT.8      q1, q1, q1, 2

7:
        TST         r1, 1
        BEQ         8f
        VST1.8      {d0[0]}, [r11]
        VST1.8      {d1[0]}, [r4]
        VST1.8      {d2[0]}, [r8]
        VST1.8      {d3[0]}, [r6]

8:
        VPOP        {d8-d13}
        ADD         sp, sp, 12              // skip pad of 8 + r2
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}


END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

