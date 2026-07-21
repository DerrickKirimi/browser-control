"""Print the CUDA facts that control model dtype and memory decisions."""

from __future__ import annotations

import json

import torch

from browser_control.hardware import inspect_cuda_device


def main() -> None:
    device = inspect_cuda_device(torch)
    report = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device.to_dict(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

