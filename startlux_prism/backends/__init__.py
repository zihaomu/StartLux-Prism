"""Backend identity and capability gates shared by the engine and kernels."""

import functools
import os
import torch


def is_hip():
    return torch.version.hip is not None


@functools.lru_cache(None)
def architecture(device=None):
    p = torch.cuda.get_device_properties(device)
    if is_hip():
        return p.gcnArchName.split(":")[0]
    return f"sm_{p.major}{p.minor}" + ("a" if p.major == 9 else "")


def require_single_gpu(tp=1, gpus=""):
    if not is_hip():
        return
    if tp != 1 or (gpus and gpus != "0"):
        raise ValueError("Prism ROCm supports one visible GPU only: tp=1, logical GPU 0")
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"Prism ROCm requires exactly one visible GPU; found {count}. "
            "Select one physical card with HIP_VISIBLE_DEVICES before startup."
        )


def validate_runtime(tp=1, gpus="", device=None):
    """CDNA4 uses portable Triton/RCCL; retain the Radeon profile's restrictions."""
    if not is_hip():
        return
    arch = architecture(device)
    if arch == "gfx950":
        return
    if arch != "gfx1201":
        raise ValueError(f"Unsupported ROCm architecture {arch}; expected gfx950 or gfx1201")
    require_single_gpu(tp, gpus)


def supports_kv_fp8(device=None):
    if is_hip():
        return architecture(device) == "gfx950"
    return torch.cuda.get_device_capability(device) >= (8, 9)


def device_env(gpus):
    """Select HIP ordinals on ROCm and CUDA ordinals on NVIDIA in child processes."""
    env = dict(os.environ)
    key, other = (("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if is_hip()
                  else ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"))
    env.pop(other, None)
    env[key] = str(gpus)
    return env


@functools.lru_cache(None)
def gpu_tag(device=None):
    if is_hip():
        return architecture(device)
    name = torch.cuda.get_device_name(device).lower().replace("nvidia", "").replace("geforce", "")
    return "".join(name.split())
