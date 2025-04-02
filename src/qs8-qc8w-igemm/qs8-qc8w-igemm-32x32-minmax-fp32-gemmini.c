// Copyright Xu Jinsheng
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>

#include "gemmini.h"
#include "gemmini_params.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_32x32__gemmini(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) {
    assert(mr != 0);
    assert(mr <= 32);
    assert(nc != 0);
    assert(kc != 0);
    assert(cn_stride == 1);

    assert(false);
}