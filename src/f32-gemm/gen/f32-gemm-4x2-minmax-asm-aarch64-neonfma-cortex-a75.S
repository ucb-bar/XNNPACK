// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x2-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

# void xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const float* w,           x5
#     float* c,                 x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_minmax_params* params)  [sp + 8] -> x8

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0  v4
// A1 x11 v1  v5
// A2 x12 v2  v6
// A3  x4 v3  v7
// B   x5 v16 v17 v18 v19 v20 v21 v22 v23
// C0  x6 v24 v25
// C1  x9 v26 v27
// C2 x10 v28 v29
// C3  x7 v30 v31
// clamp  v4 v5

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max values
        LD2R        {v4.2s, v5.2s}, [x8]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x11, x3, x4             // a1 = a0 + a_stride
        ADD         x9, x6, x7              // c1 = c0 + cm_stride
        CSEL        x11, x3, x11, LO        //   a1 = a0
        CSEL        x9, x6, x9, LO          //   c1 = c0

        ADD         x12, x11, x4            // a2 = a1 + a_stride
        ADD         x10, x9, x7             // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x12, x11, x12, LS       //   a2 = a1
        CSEL        x10, x9, x10, LS        //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x4, x12, x4             // a3 = a2 + a_stride
        ADD         x7, x10, x7             // c3 = c2 + cm_stride
        CSEL        x4, x12, x4, LO         //   a3 = a2
        CSEL        x7, x10, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LDR         d24, [x5], 8
        MOV         v26.8b, v24.8b
        MOV         v30.8b, v24.8b
        MOV         v28.8b, v24.8b
        MOVI        v25.2s, 0
        MOVI        v27.2s, 0
        MOVI        v29.2s, 0
        MOVI        v31.2s, 0

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS        x0, x2, 32              // k = kc - 32
        B.LO        4f

        # Prologue
        # Read first block of 4 A and B.
        LDR         q0,  [x3], 16
        LDP         d20, d21, [x5], 16
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3,  [x4], 16
        LDP         d22, d23, [x5], 16

        # Is there at least 32.  yes do main loop
        SUBS        x0, x0, 32
        B.LO        2f

        # Main loop - 8 floats of A (32 bytes)
1:
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.2s, v20.2s, v0.s[0]
        LDR         q4, [x3], 16
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        LDR         d16, [x5, 0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        LDR         q5, [x11], 16
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        LDR         q6, [x12], 16
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        LDR         q7, [x4], 16
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        LDR         d17, [x5, 8]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        LDR         d18, [x5, 16]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        LDR         d19, [x5, 24]
        FMLA        v31.2s, v23.2s, v3.s[3]

        # Second block of 4.  FMA for second 4, loads for 1st block of 4.
        FMLA        v24.2s, v16.2s, v4.s[0]
        LDR         q0, [x3], 16
        FMLA        v26.2s, v16.2s, v5.s[0]
        FMLA        v28.2s, v16.2s, v6.s[0]
        LDR         d20, [x5, 32]
        FMLA        v30.2s, v16.2s, v7.s[0]
        FMLA        v25.2s, v17.2s, v4.s[1]
        LDR         q1, [x11], 16
        FMLA        v27.2s, v17.2s, v5.s[1]
        FMLA        v29.2s, v17.2s, v6.s[1]
        LDR         q2, [x12], 16
        FMLA        v31.2s, v17.2s, v7.s[1]
        FMLA        v24.2s, v18.2s, v4.s[2]
        LDR         q3, [x4], 16
        FMLA        v26.2s, v18.2s, v5.s[2]
        FMLA        v28.2s, v18.2s, v6.s[2]
        LDR         d21, [x5, 40]
        FMLA        v30.2s, v18.2s, v7.s[2]
        SUBS        x0, x0, 32
        FMLA        v25.2s, v19.2s, v4.s[3]
        LDR         d22, [x5, 48]
        FMLA        v27.2s, v19.2s, v5.s[3]
        LDR         d23, [x5, 56]
        FMLA        v29.2s, v19.2s, v6.s[3]
        ADD         x5, x5, 64
        FMLA        v31.2s, v19.2s, v7.s[3]
        B.HS        1b

2:
        # Epilogue
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.2s, v20.2s, v0.s[0]
        LDR         q4, [x3], 16
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        LDR         d16, [x5, 0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        LDR         q5, [x11], 16
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        LDR         q6, [x12], 16
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        LDR         q7, [x4], 16
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        LDR         d17, [x5, 8]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        LDR         d18, [x5, 16]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        LDR         d19, [x5, 24]
        FMLA        v31.2s, v23.2s, v3.s[3]

        # Second block of 4.  FMA for second 4, no loads
        FMLA        v24.2s, v16.2s, v4.s[0]
        FMLA        v26.2s, v16.2s, v5.s[0]
        FMLA        v28.2s, v16.2s, v6.s[0]
        FMLA        v30.2s, v16.2s, v7.s[0]
        FMLA        v25.2s, v17.2s, v4.s[1]
        FMLA        v27.2s, v17.2s, v5.s[1]
        FMLA        v29.2s, v17.2s, v6.s[1]
        FMLA        v31.2s, v17.2s, v7.s[1]
        FMLA        v24.2s, v18.2s, v4.s[2]
        FMLA        v26.2s, v18.2s, v5.s[2]
        FMLA        v28.2s, v18.2s, v6.s[2]
        ADDS        x0, x0, 32
        FMLA        v30.2s, v18.2s, v7.s[2]
        FMLA        v25.2s, v19.2s, v4.s[3]
        ADD         x5, x5, 32
        FMLA        v27.2s, v19.2s, v5.s[3]
        FMLA        v29.2s, v19.2s, v6.s[3]
        LD2R        {v4.2s, v5.2s}, [x8]        // Load min/max values
        FMLA        v31.2s, v19.2s, v7.s[3]

        # Is there a remainder? up to 8 floats (32 bytes)
        B.NE        4f

3:
        FADD        v24.2s, v24.2s, v25.2s
        FADD        v26.2s, v26.2s, v27.2s
        FADD        v28.2s, v28.2s, v29.2s
        FADD        v30.2s, v30.2s, v31.2s

        # Clamp
        FMAX        v24.2s, v24.2s, v4.2s
        FMAX        v26.2s, v26.2s, v4.2s
        FMAX        v28.2s, v28.2s, v4.2s
        FMAX        v30.2s, v30.2s, v4.2s
        SUBS        x1, x1, 2
        FMIN        v24.2s, v24.2s, v5.2s
        FMIN        v26.2s, v26.2s, v5.2s
        FMIN        v28.2s, v28.2s, v5.2s
        FMIN        v30.2s, v30.2s, v5.2s

        # Store full 4 x 2
        B.LO        7f

        STR         d24, [x6]
        SUB         x3,  x3, x2             // a0 -= kc
        ADD         x6,  x6, x14
        STR         d26, [x9]
        SUB         x11, x11, x2            // a1 -= kc
        ADD         x9,  x9, x14
        STR         d28, [x10]
        SUB         x12, x12, x2            // a2 -= kc
        ADD         x10, x10, x14
        STR         d30, [x7]
        SUB         x4,  x4, x2             // a3 -= kc
        ADD         x7,  x7, x14

        B.HI        0b
        RET

4:
        # Remainder- 4 floats of A (16 bytes)
        TBZ         x0, 4, 5f

        LDR         q0,  [x3], 16
        LDP         d20, d21, [x5], 16
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3,  [x4], 16
        LDP         d22, d23, [x5], 16
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        FMLA        v31.2s, v23.2s, v3.s[3]

5:
        # Remainder- 2 floats of A (8 bytes)
        TBZ         x0, 3, 6f

        LDR         d0,  [x3], 8
        LDP         d20, d21, [x5], 16
        LDR         d1, [x11], 8
        LDR         d2, [x12], 8
        LDR         d3,  [x4], 8
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        FMLA        v31.2s, v21.2s, v3.s[1]

6:
        # Remainder- 1 float of A (4 bytes)
        TBZ         x0, 2, 3b

        LDR         s0,  [x3], 4
        LDR         d20, [x5], 8
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        B           3b

        # Store odd width
7:
        STR         s24,  [x6]
        STR         s26,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]
        RET


END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
