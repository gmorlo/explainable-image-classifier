"""Smoke-test PyTorch and CUDA availability."""

from __future__ import annotations


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed")
        print("Install CUDA build, e.g.:")
        print(
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
        return 1

    print(f"torch: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda_available: {cuda_ok}")

    if not cuda_ok:
        print("FAIL: CUDA not available (CPU-only build or missing drivers)")
        return 1

    device_count = torch.cuda.device_count()
    print(f"device_count: {device_count}")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"device[{i}]: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GiB)")

    x = torch.randn(2, 3, device="cuda")
    y = x @ x.T
    print(f"smoke_matmul_ok: {y.shape == (2, 2)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
