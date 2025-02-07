# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for amd64
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AMD64_ASM_MICROKERNEL_SRCS
  src/bf16-f32-gemm/gen/bf16-f32-gemm-1x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-7x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/f32-gemm/gen/f32-gemm-1x32c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-5x32c2-minmax-asm-amd64-avx512f-broadcast.S)

SET(NON_PROD_AMD64_ASM_MICROKERNEL_SRCS
  src/bf16-f32-gemm/gen/bf16-f32-gemm-1x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-1x64c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-2x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-2x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-2x64c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-3x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-3x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-3x64c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-4x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-4x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-4x64c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-5x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-5x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-5x64c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-6x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-6x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-7x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-8x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-8x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-9x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-9x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-10x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-10x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-11x16c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/bf16-f32-gemm/gen/bf16-f32-gemm-11x32c2-minmax-asm-amd64-avx512bf16-broadcast.S
  src/f32-gemm/gen/f32-gemm-1x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-1x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-1x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-1x64-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-2x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-2x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-2x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-2x32c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-2x64-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-3x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-3x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-3x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-3x32c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-3x64-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-4x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-4x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-4x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-4x32c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-4x64-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-5x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-5x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-5x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-5x64-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-6x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-6x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-6x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-7x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-7x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-7x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-8x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-8x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-8x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-9x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-9x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-9x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-10x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-10x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-10x32-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-11x16-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-11x16c2-minmax-asm-amd64-avx512f-broadcast.S
  src/f32-gemm/gen/f32-gemm-11x32-minmax-asm-amd64-avx512f-broadcast.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x64-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x64-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x64-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x64-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x64-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x32-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-11x16-minmax-asm-amd64-avx512vnni.S
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-11x32-minmax-asm-amd64-avx512vnni.S)

SET(ALL_AMD64_ASM_MICROKERNEL_SRCS ${PROD_AMD64_ASM_MICROKERNEL_SRCS} + ${NON_PROD_AMD64_ASM_MICROKERNEL_SRCS})
