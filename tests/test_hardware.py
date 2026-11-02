from __future__ import annotations

from dataclasses import dataclass

import pytest

from browser_control.hardware import inspect_cuda_device


@dataclass
class DeviceProperties:
    name: str = "Tesla T4"
    total_memory: int = 16 * 1024**3


class FakeCuda:
    def __init__(self, *, available: bool, capability: tuple[int, int] = (7, 5)):
        self._available = available
        self._capability = capability

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> DeviceProperties:
        assert index == 0
        return DeviceProperties()

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return self._capability


@dataclass
class FakeTorch:
    cuda: FakeCuda


def test_t4_selects_float16() -> None:
    device = inspect_cuda_device(FakeTorch(cuda=FakeCuda(available=True)))

    assert device.name == "Tesla T4"
    assert device.compute_capability == (7, 5)
    assert not device.supports_bfloat16
    assert device.training_dtype == "float16"
    assert device.to_dict()["compute_capability"] == "7.5"


def test_ampere_selects_bfloat16() -> None:
    device = inspect_cuda_device(
        FakeTorch(cuda=FakeCuda(available=True, capability=(8, 0)))
    )

    assert device.supports_bfloat16
    assert device.training_dtype == "bfloat16"


def test_cuda_unavailable_has_colab_instructions() -> None:
    with pytest.raises(RuntimeError, match="Change runtime type"):
        inspect_cuda_device(FakeTorch(cuda=FakeCuda(available=False)))


def test_device_index_is_validated() -> None:
    with pytest.raises(ValueError, match="out of range"):
        inspect_cuda_device(FakeTorch(cuda=FakeCuda(available=True)), index=1)
