# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Generate possible candidates when tuning a TK kernel.
This file generates possible candidates for a given operation and problem size.
These candidates are then evaluated to see which kernel performs the best.
"""

import argparse
import logging
import math
import pickle
import re
import z3
from dataclasses import asdict, dataclass
from enum import Enum
from os import mkdir, path, makedirs
from typing import Callable, Optional
from textwrap import indent
from abc import ABC, abstractmethod
from .tk_gemm import generate_gemm_mlir, GEMMTunableParameters

tune_logger = logging.getLogger("tune")


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    batch_matmul = 3


class ElementType(Enum):
    i8 = 1
    i32 = 2
    f8 = 3
    f16 = 4
    f32 = 5

    @property
    def bitwidth(self) -> int:
        match self:
            case ElementType.i8 | ElementType.f8:
                return 8
            case ElementType.f16:
                return 16
            case ElementType.i32 | ElementType.f32:
                return 32
            case _:
                assert False, "unhandled case"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_str(cls, s: str):
        if s == "f16":
            return cls.f16
        if s == "f32":
            return cls.f32
        if s == "i8":
            return cls.i8
        if s == "i32":
            return cls.i32
        if s == "f8":
            return cls.f8
        raise ValueError(f"Invalid element type: {s}")


@dataclass
class MatmulSize:
    M: int
    N: int
    K: int
    B: int = 1


@dataclass
class ConvDimInfo:
    n: int
    oh: int
    ow: int
    oc: int
    fh: int
    fw: int
    ic: int

    def to_matmul_size(self) -> MatmulSize:
        # M = oh * ow,
        # N = oc,
        # K = fh * fw * ic,
        # B = n,
        # Here we only compute M, N, K as B is not used in the matmul size
        return MatmulSize(self.oh * self.ow, self.oc, self.fh * self.fw * self.ic)


@dataclass
class ProblemSize:
    sizes: MatmulSize | ConvDimInfo
    dispatch_kind: DispatchKind
    input_dtype: ElementType = ElementType.f16
    output_dtype: ElementType = ElementType.f32

    @property
    def MNK(self) -> tuple[int, int, int]:
        if isinstance(self.sizes, ConvDimInfo):
            return self.sizes.to_matmul_size()
        return (self.sizes.M, self.sizes.N, self.sizes.K)


@dataclass
class MfmaIntrinsic:
    output_type: ElementType
    m: int
    n: int
    k: int
    input_type: ElementType

    def __str__(self) -> str:
        input = str(self.input_type).upper()
        output = str(self.output_type).upper()
        return f"MFMA_{output}_{self.m}x{self.n}x{self.k}_{input}"

    @staticmethod
    def mfma_f32_16x16x16_f16():
        return MfmaIntrinsic(ElementType.f32, 16, 16, 16, ElementType.f16)

    @staticmethod
    def mfma_f32_32x32x8_f16():
        return MfmaIntrinsic(ElementType.f32, 32, 32, 8, ElementType.f16)

    @staticmethod
    def mfma_i32_16x16x32_i8():
        return MfmaIntrinsic(ElementType.i32, 16, 16, 32, ElementType.i8)

    @staticmethod
    def mfma_i32_32x32x16_i8():
        return MfmaIntrinsic(ElementType.i32, 32, 32, 16, ElementType.i8)

    @staticmethod
    def all():
        # TODO: Add support for more intrinsics
        return [
            MfmaIntrinsic.mfma_f32_16x16x16_f16(),
            # MfmaIntrinsic.mfma_f32_32x32x8_f16(),
            # MfmaIntrinsic.mfma_i32_16x16x32_i8(),
            # MfmaIntrinsic.mfma_i32_32x32x16_i8(),
        ]


@dataclass
class Configuration:
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: MfmaIntrinsic
    tile_sizes: list[int]
    subgroup_m_count: int
    subgroup_n_count: int
    waves_per_eu: int


def get_compatible_mfma_intrinsics(problem_size: ProblemSize) -> list[MfmaIntrinsic]:
    def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
        if problem_size.output_dtype != intrinsic.output_type:
            return False
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
            if problem_size.input_dtype != intrinsic.input_type:
                return False
        return True

    return list(filter(is_compatible, MfmaIntrinsic.all()))


def get_mfma_intrinsic_constraints(
    problem_size: ProblemSize,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
) -> z3.BoolRef:
    compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size)
    assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
    return z3.Or(
        *(
            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
            for mfma in compatible_intrinsics
        )
    )


def get_dispatch_constraints(
    problem_size: ProblemSize,
    tile_m: z3.ArithRef,
    tile_n: z3.ArithRef,
    tile_k: z3.ArithRef,
) -> list[z3.BoolRef]:
    if problem_size.dispatch_kind != DispatchKind.conv:
        return []

    dim_info = problem_size.sizes
    conv_constraints = []
    # WARNING: This sometimes makes the constraints UNSAT for some reason.
    conv_constraints += [tile_m <= dim_info.ow]
    conv_constraints += [tile_n <= dim_info.oc]
    conv_constraints += [tile_k <= dim_info.ic]
    return conv_constraints


def calculate_shared_memory_usage_in_bytes(
    problem_size: ProblemSize,
    m: int | z3.ArithRef,
    n: int | z3.ArithRef,
    k: int | z3.ArithRef,
) -> int | z3.ArithRef:
    lhs_memory = m * k * (problem_size.input_dtype.bitwidth // 8)
    rhs_memory = k * n * (problem_size.input_dtype.bitwidth // 8)
    return lhs_memory + rhs_memory


def generate_constraints(
    problem_size: ProblemSize,
    tile_sizes,
    num_subgroups,
    subgroup_size,
    intrinsic_size,
    workgroup_size,
    subgroup_m_count,
    subgroup_n_count,
    waves_per_eu,
):
    M, N, K = problem_size.MNK
    m, n, k = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = z3.Int("wg_threads")
    constraints = [wg_threads == wg_x * wg_y * wg_z]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            problem_size, intrinsic_mn, intrinsic_mn, intrinsic_k
        )
    ]
    subgroup_k_count = 1
    constraints += [
        m >= intrinsic_mn,
        m <= 512,
        m <= M,
        M % m == 0,  # M should be divisible by m
        # TODO: Figure out why this was excluded from the original (unaligned tuning?)
    ]
    constraints += [n >= intrinsic_mn, n <= 512, n <= N, N % n == 0]
    constraints += [k >= intrinsic_k, k <= 512, k <= K, K % k == 0]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")
    for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 32]

    constraints += [m == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn]
    constraints += [n == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn]
    constraints += [k == subgroup_k_count * subgroup_k_tile_count * intrinsic_k]
    # These two lines are the only change for Tk kernels.
    constraints += [wg_x == subgroup_size * subgroup_m_count]
    constraints += [wg_y == subgroup_n_count]
    # ---------------------------------------------------
    constraints += [wg_z == subgroup_k_count]
    constraints += [z3.Or(wg_x <= n, wg_x <= m)]
    constraints += [k % intrinsic_mn == 0]
    constraints += [(k * n) % wg_threads == 0]
    constraints += [(k * m) % wg_threads == 0]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]

    constraints += [waves_per_eu == 2]
    # constraints += [z3.Or(waves_per_eu == 2, waves_per_eu == 3, waves_per_eu == 4)]

    shared_memory = calculate_shared_memory_usage_in_bytes(problem_size, m, n, k)
    constraints += [shared_memory <= 65536]

    constraints += get_dispatch_constraints(problem_size, m, n, k)

    return constraints


def generate_solutions(problem_size: ProblemSize, num_subgrups: int):
    M, N, K = problem_size.MNK
    tune_logger.info(f"{M},{N},{K}")
    m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")
    all_vars = [
        m,
        n,
        k,
        subgroup_size,
        intrinsic_mn,
        intrinsic_k,
        wg_x,
        wg_y,
        wg_z,
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    ]

    solver = z3.Solver()
    constraints = generate_constraints(
        problem_size,
        [m, n, k],
        num_subgrups,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )
    solver.add(z3.simplify(z3.And(constraints)))
    tune_logger.debug(f"Initial constraints: {solver}")
    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()

        config = Configuration(
            lookup(subgroup_size),
            [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
            MfmaIntrinsic(
                problem_size.input_dtype,
                lookup(intrinsic_mn),
                lookup(intrinsic_mn),
                lookup(intrinsic_k),
                problem_size.output_dtype,
            ),
            [lookup(m), lookup(n), lookup(k)],
            lookup(sg_m_cnt),
            lookup(sg_n_cnt),
            lookup(waves_per_eu),
        )
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        yield config


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


def tune(
    problem_size: ProblemSize,
    output: str = "",  # Path to the output directory, auto creates one if not given
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # Number of subgroups per workgroup to use. (-1 == unconstrained)
):

    if not output:
        output = get_default_output_dir()

    # Create the directory if it does not exist
    makedirs(str(output), exist_ok=True)

    tune_logger.debug(f"Output directory {output}")

    tune_logger.debug(str(problem_size))
    configs = []
    for i, config in enumerate(generate_solutions(problem_size, num_subgroups)):
        if i >= limit:
            break
        tune_logger.info(f"Solution #{i+1}: {config}")
        configs.append(config)
        asm = None
        if problem_size.dispatch_kind == DispatchKind.mmt:
            # TODO: Add waves-per-eu as translation info to the module generated
            gemm_config = GEMMTunableParameters(
                problem_size.MNK[0],
                problem_size.MNK[1],
                problem_size.MNK[2],
                config.tile_sizes[0],
                config.tile_sizes[1],
                config.tile_sizes[2],
                config.subgroup_m_count,
                config.subgroup_n_count,
                config.intrinsic.m,
                config.intrinsic.n,
                config.intrinsic.k,
            )
            asm = generate_gemm_mlir(gemm_config)
        with open(path.join(output, f"{i}.mlir"), "w") as f:
            f.write(asm)

    with open(path.join(output, "configs.pkl"), "wb") as file:
        pickle.dump(configs, file)

    tune_logger.info(f"Generated {len(configs)} candidates")
    tune_logger.info(f"Configurations .pkl is stored in {output}/configs.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--problem", help="Problem type", type=str, choices=["mmt", "conv"]
    )
    # GEMM-specific flags
    parser.add_argument("-M", help="M dimension of GEMM", type=int)
    parser.add_argument("-N", help="N dimension of GEMM", type=int)
    parser.add_argument("-K", help="K dimension of GEMM", type=int)
    # Conv-specific flags
    parser.add_argument("-OH", help="Output height of conv", type=int)
    parser.add_argument("-OW", help="Output width of conv", type=int)
    parser.add_argument("-OC", help="Output channels of conv", type=int)
    parser.add_argument("-FH", help="Filter height of conv", type=int)
    parser.add_argument("-FW", help="Filter width of conv", type=int)
    parser.add_argument("-IC", help="Input channels of conv", type=int)
    parser.add_argument("-N", help="Batch size of conv", type=int)
    # Dtype flags
    parser.add_argument("--input-dtype", help="Input dtype", type=str, default="f16")
    parser.add_argument("--output-dtype", help="Output dtype", type=str, default="f32")
    # Other flags
    parser.add_argument(
        "-o", "--output", help="Output dir", type=str, default=get_default_output_dir()
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of candidates generated",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )

    args = parser.parse_args()
    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create printing formatter for logging info
    formatter = logging.Formatter("%(message)s")

    # Create a handler to print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    # # Optionally, add a file handler to log to a file
    file_handler = logging.FileHandler("tune.log")
    file_handler.setFormatter(formatter)
    tune_logger.addHandler(file_handler)

    problem_size = None
    input_dtype = ElementType.from_str(args.input_dtype)
    output_dtype = ElementType.from_str(args.output_dtype)
    if args.problem == "mmt":
        problem_size = ProblemSize(
            MatmulSize(args.M, args.N, args.K),
            DispatchKind.mmt,
            input_dtype,
            output_dtype,
        )
    elif args.problem == "conv":
        problem_size = ProblemSize(
            ConvDimInfo(args.N, args.OH, args.OW, args.OC, args.FH, args.FW, args.IC),
            DispatchKind.conv,
            input_dtype,
            output_dtype,
        )

    tune(
        problem_size,
        args.output,
        args.limit,
        args.num_subgroups,
    )


if __name__ == "__main__":
    args = main()
