"""
Microkernel filenames lists for gemmini.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_GEMMINI_MICROKERNEL_SRCS = [
    "src/f32-gemm/f32-gemm-4x4-gemmini.c",
    "src/qs8-qc8w-gemm/qs8-qc8w-gemm-32x32-minmax-fp32-gemmini.c",
    "src/qs8-qc8w-igemm/qs8-qc8w-igemm-32x32-minmax-fp32-gemmini.c",
]

NON_PROD_GEMMINI_MICROKERNEL_SRCS = [
    "src/f32-igemm/f32-igemm-4x4-gemmini.c",
]

ALL_GEMMINI_MICROKERNEL_SRCS = PROD_GEMMINI_MICROKERNEL_SRCS + NON_PROD_GEMMINI_MICROKERNEL_SRCS
