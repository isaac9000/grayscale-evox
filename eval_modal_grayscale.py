"""
Deployable Modal H100 evaluator for the grayscale kernel task.
Evaluation methodology matches SkyDiscover GPU Mode benchmarks.

Deploy once:
    uv run modal deploy eval_modal_grayscale.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# Matches SkyDiscover gpu_mode/grayscale/reference.py
TEST_CASES = [
    {"size": 256,  "seed": 42},
    {"size": 512,  "seed": 123},
    {"size": 1024, "seed": 456},
    {"size": 2048, "seed": 789},
]

BENCHMARK_CASES = [
    {"size": 1024, "seed": 1001},
    {"size": 2048, "seed": 1002},
    {"size": 4096, "seed": 1003},
    {"size": 8192, "seed": 1004},
]

SCORE_SCALE = 3000.0
BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001       # stop when stderr/mean < 0.1%
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9      # 10 seconds per benchmark case
BENCH_WALL_TIMEOUT_NS = 120e9 # 120 seconds wall time
BENCH_WARMUP_BUDGET_NS = 10e7 # 100ms warmup budget

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("grayscale-kernel-eval")


@app.function(gpu="H100", image=image, timeout=300)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import json as _json
    import math
    import time
    import traceback
    import importlib.util
    import tempfile
    import os as _os
    import gc

    import torch

    def ref_kernel(data):
        rgb, output = data
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device, dtype=rgb.dtype)
        output[...] = torch.sum(rgb * weights, dim=-1)
        return output

    def generate_input(size, seed):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        rgb = torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()
        output = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()
        return (rgb, output)

    def _stats(durations_ns):
        n = len(durations_ns)
        mean = sum(durations_ns) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in durations_ns) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": mean, "std": std, "err": err}

    def _bench_single(kernel_fn, case, max_time_ns):
        data = generate_input(case["size"], case["seed"])
        durations_ns = []
        bm_start = time.perf_counter_ns()

        for i in range(BENCH_MAX_REPEATS):
            torch.cuda.synchronize()

            if BENCH_USE_CUDA_EVENTS:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                output = kernel_fn(data)
                e.record()
                torch.cuda.synchronize()
                duration_ns = s.elapsed_time(e) * 1e6  # ms -> ns
            else:
                t0 = time.perf_counter_ns()
                output = kernel_fn(data)
                torch.cuda.synchronize()
                duration_ns = time.perf_counter_ns() - t0

            del output
            durations_ns.append(duration_ns)

            if i > 1:
                st = _stats(durations_ns)
                if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                    break
                if st["mean"] * st["runs"] > max_time_ns:
                    break
                if (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                    break

        return _stats(durations_ns)

    def _warmup(kernel_fn):
        _bench_single(kernel_fn, BENCHMARK_CASES[0], max_time_ns=BENCH_WARMUP_BUDGET_NS)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    tmp_dir = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")
    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h100",
        })

    custom_kernel = getattr(module, "custom_kernel", None)
    if custom_kernel is None:
        return _json.dumps({
            "success": False,
            "error": "submission.py does not define custom_kernel(data)",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h100",
        })

    # Correctness tests
    test_details = []
    tests_passed = 0
    for case in TEST_CASES:
        size, seed = case["size"], case["seed"]
        try:
            data_ref = generate_input(size, seed=seed)
            data_test = generate_input(size, seed=seed)
            expected = ref_kernel(data_ref)
            torch.cuda.synchronize()
            actual = custom_kernel(data_test)
            torch.cuda.synchronize()
            if actual.shape != expected.shape:
                test_details.append({
                    "size": size, "seed": seed, "passed": False,
                    "error": f"shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}",
                })
            elif not torch.allclose(actual, expected, rtol=1e-4, atol=1e-4):
                max_diff = (actual - expected).abs().max().item()
                test_details.append({
                    "size": size, "seed": seed, "passed": False,
                    "error": f"values differ, max abs diff: {max_diff:.6f}",
                })
            else:
                test_details.append({"size": size, "seed": seed, "passed": True})
                tests_passed += 1
        except Exception:
            test_details.append({
                "size": size, "seed": seed, "passed": False,
                "error": traceback.format_exc()[:600],
            })

    if tests_passed < len(TEST_CASES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h100",
        })

    if mode == "test":
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h100",
        })

    # Clean up before benchmarking
    del data_ref, data_test, expected, actual
    gc.collect()
    torch.cuda.empty_cache()

    # Warmup
    _warmup(custom_kernel)

    # Adaptive benchmarking
    benchmark_details = []
    bench_means_ns = []

    for case in BENCHMARK_CASES:
        try:
            st = _bench_single(custom_kernel, case, max_time_ns=BENCH_MAX_TIME_NS)
            mean_us = st["mean"] / 1e3
            std_us = st["std"] / 1e3
            err_us = st["err"] / 1e3
            benchmark_details.append({
                "size": case["size"],
                "seed": case["seed"],
                "mean_us": round(mean_us, 3),
                "std_us": round(std_us, 3),
                "err_us": round(err_us, 3),
                "runs": st["runs"],
            })
            bench_means_ns.append(st["mean"])
        except Exception:
            return _json.dumps({
                "success": False,
                "tests_passed": tests_passed,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
                "error": f"Benchmark failed at size={case['size']}:\n{traceback.format_exc()}",
                "gpu_name": gpu_name,
                "torch_version": torch_ver,
                "platform": "modal-h100",
            })

    means_s = [ns / 1e9 for ns in bench_means_ns]
    import math as _math
    geomean_s = _math.pow(_math.prod(means_s), 1.0 / len(means_s))
    geomean_us = geomean_s * 1e6
    score = SCORE_SCALE / geomean_us

    return _json.dumps({
        "success": True,
        "tests_passed": tests_passed,
        "tests_total": len(TEST_CASES),
        "test_details": test_details,
        "benchmark": {
            "geomean_us": round(geomean_us, 3),
            "score": round(score, 3),
        },
        "benchmark_details": benchmark_details,
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        "platform": "modal-h100",
    })
