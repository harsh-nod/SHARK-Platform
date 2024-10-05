# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel.wave.constraints import MMAType
import os
import json
from torch.testing import assert_close
from dataclasses import dataclass


@dataclass
class GEMMTunableParameters:
    M: int
    N: int
    K: int
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    RATIO_M: int
    RATIO_N: int
    INTRINSIC_M: int
    INTRINSIC_N: int
    INTRINSIC_K: int


def generate_gemm_mlir(config: GEMMTunableParameters) -> str:
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / config.RATIO_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / config.RATIO_N)]

    mma_type = None
    if (
        config.INTRINSIC_M == 16
        and config.INTRINSIC_N == 16
        and config.INTRINSIC_K == 16
    ):
        mma_type = MMAType.F32_16x16x16_F16

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(config.RATIO_M, config.RATIO_N, 1),
            mma_type=mma_type,
        )
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, SHARED_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, SHARED_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        # Fixed parameters
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        # Tunable parameters
        BLOCK_M: config.BLOCK_M,
        BLOCK_N: config.BLOCK_N,
        BLOCK_K: config.BLOCK_K,
        M: config.M,
        N: config.N,
        K: config.K,
        # Unused scheduling parameters
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }
    tk_config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        run_config=tk_config,
        schedule=False,
        add_host_codegen_call=True,
    ):
        a = torch.randn(config.M, config.K, dtype=torch.float16)
        b = torch.randn(config.N, config.K, dtype=torch.float16)
        c = torch.zeros(config.M, config.N, dtype=torch.float32)
        mb = gemm(a, b, c)
        return mb.module_op.get_asm()
