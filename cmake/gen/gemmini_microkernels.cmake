# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for gemmini
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_GEMMINI_MICROKERNEL_SRCS
  src/qs8-qc8w-gemm/qs8-qc8w-gemm-32x32-minmax-fp32-gemmini.c
  src/qs8-qc8w-igemm/qs8-qc8w-igemm-32x32-minmax-fp32-gemmini.c)

SET(NON_PROD_GEMMINI_MICROKERNEL_SRCS)

SET(ALL_GEMMINI_MICROKERNEL_SRCS ${PROD_GEMMINI_MICROKERNEL_SRCS} + ${NON_PROD_GEMMINI_MICROKERNEL_SRCS})
