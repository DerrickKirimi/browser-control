"""CUDA capability checks for Colab model execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CudaDevice:
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int

    @property
    def supports_bfloat16(self) -> bool:
        return self.compute_capability[0] >= 8

    @property
    def training_dtype(self) -> str:
        return "bfloat16" if self.supports_bfloat16 else "float16"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compute_capability"] = ".".join(
            str(part) for part in self.compute_capability
        )
        payload["training_dtype"] = self.training_dtype
        payload["supports_bfloat16"] = self.supports_bfloat16
        return payload


def inspect_cuda_device(torch_module: Any, index: int = 0) -> CudaDevice:
    """Validate CUDA availability and describe one device."""

    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Colab, select Runtime > Change runtime type > "
            "GPU, reconnect, and rerun the diagnostic."
        )
    if index < 0 or index >= torch_module.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {index}")
    properties = torch_module.cuda.get_device_properties(index)
    capability = tuple(torch_module.cuda.get_device_capability(index))
    return CudaDevice(
        index=index,
        name=str(properties.name),
        compute_capability=(int(capability[0]), int(capability[1])),
        total_memory_bytes=int(properties.total_memory),
    )

