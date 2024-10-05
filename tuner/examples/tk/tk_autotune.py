# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Sample Usage:

python -m examples.tk -p mmt -M 1024 -N 256 -K 128 --input-dtype f16 --output-dtype f32 --devices=hip://0,hip://1 --num-candidates=64


Recommended Trial Run:

python -m examples.tk -p mmt -M 1024 -N 256 -K 128 --input-dtype f16 --output-dtype f32 --num-candidates=1

Dry Run Test (no gpu requried):

python -m examples.tk -p mmt -M 1024 -N 256 -K 128 --input-dtype f16 --output-dtype f32 --num-candidates=1 --num-model-candidates=10 --dry-run

"""

from tuner import libtuner
from pathlib import Path


class TkGemmClient(libtuner.TuningClient):
    def get_dispatch_compile_timeout_s(self) -> int:
        return 4

    def get_dispatch_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        mlir_path = candidate_tracker.dispatch_mlir_path
        assert mlir_path is not None
        out_file = mlir_path.stem + ".vmfb"
        out_dir = mlir_path.parent.as_posix() + "/compiled"
        out_path = Path(out_dir + "/" + out_file)
        command = [
            "iree-compile",
            f"--iree-hal-target-backends=rocm",
            "--iree-hip-target=gfx942",
            "--iree-hal-benchmark-dispatch-repeat-count=1000",
            "-o",
            out_path.resolve().as_posix(),
            mlir_path.as_posix(),
        ]
        return command

    def get_dispatch_benchmark_timeout_s(self) -> int:
        return 15

    def get_dispatch_benchmark_command(
        self,
        candidate_tracker: libtuner.CandidateTracker,
    ) -> list[str]:
        compiled_vmfb_path = candidate_tracker.compiled_dispatch_path
        assert compiled_vmfb_path is not None
        args = candidate_tracker.args

        command = [
            "iree-benchmark-module",
            f"--device={libtuner.DEVICE_ID_PLACEHOLDER}",
            f"--module={compiled_vmfb_path.resolve()}",
            "--function=isolated_benchmark",
            f"--input={args.M}x{args.K}x{args.input_dtype}",
            f"--input={args.N}x{args.K}x{args.input_dtype}",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--batch_size=1000",
            "--benchmark_repetitions=3",
            f"--benchmark_out=dispatch_{candidate_tracker.candidate_id}_bm.json",
            "--benchmark_out_format=json",
        ]

        return command

    def get_model_compile_timeout_s(self) -> int:
        return 300

    def get_model_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return [""]

    def get_model_benchmark_timeout_s(self) -> int:
        return 180

    def get_model_benchmark_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return [""]


def main():
    args = libtuner.parse_arguments(add_tk_args=True)
    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[libtuner.CandidateTracker] = []
    gemm_client = TkGemmClient()
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    print("Generating candidates...")
    candidates = libtuner.generate_candidates_tk(args, path_config, candidate_trackers)
    print(f"Stored candidates in {path_config.candidates_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_candidates = libtuner.compile_dispatches(
        args, path_config, candidates, candidate_trackers, gemm_client
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled candidates...")
    top_candidates = libtuner.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers, gemm_client
    )
    print(f"Stored results in {path_config.output_unilog}\n")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
        return

    libtuner.summerize_top_candidates(path_config, candidate_trackers)
    print(f"Stored top candidates info in {path_config.result_summary_log}\n")

    libtuner.save_pickle(path_config.candidate_trackers_pkl, candidate_trackers)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed execution logs in:")
    print(path_config.run_log)

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
        if args.verbose:
            print(candidate)
