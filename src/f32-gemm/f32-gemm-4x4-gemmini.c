// Copyright Xu Jinsheng
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>

#include "gemmini_params_fp32.h"
#include "gemmini.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

void xnn_f32_gemm_minmax_ukernel_4x4__gemmini(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(cn_stride == 4*sizeof(float));

    tiled_matmul_auto(mr, nc, kc, (const float*)a, (const float*)w, NULL,
                        (float*)c, a_stride, kc*sizeof(float), 1, cm_stride, 1.0, 1.0, 1.0,
                        NO_ACTIVATION, 1.0f, 1.0f, 0, 0, 0, 0, 0, 0, WS);
}