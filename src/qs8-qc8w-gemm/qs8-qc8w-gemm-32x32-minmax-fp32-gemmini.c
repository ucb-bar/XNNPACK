// Copyright Xu Jinsheng
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>

#include "gemmini_params_int8.h"
#include "gemmini.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_32x32__gemmini(
    size_t mr, size_t nc, size_t kc, const int8_t* restrict a, size_t a_stride,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params
        params[restrict XNN_MIN_ELEMENTS(1)]) {
    assert(mr != 0);
    assert(mr <= 32);
    assert(nc != 0);
    assert(kc != 0);
    assert(cn_stride == 1);

    //   tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K, const elem_t
    //   *A, const elem_t *B, const void *D, void *C, size_t stride_A, size_t
    //   stride_B, size_t stride_D, size_t stride_C, scale_t A_scale_factor,
    //   scale_t B_scale_factor, scale_acc_t D_scale_factor, int act, acc_scale_t
    //   scale, acc_scale_t bert_scale, bool repeating_bias, bool transpose_A,
    //   bool transpose_B, bool full_C, bool low_D, uint8_t weightA, enum
    //   tiled_matmul_type_t tiled_matmul_type)
    tiled_matmul_auto(mr, nc, kc, (const int8_t*)a, (const int8_t*)w, NULL,
                        (int8_t*)c, a_stride, 1, 1, cm_stride, 1.0, 1.0, 1.0,
                        NO_ACTIVATION, 1.0f, 1.0f, 0, 0, 0, 0, 0, 0, WS);
}