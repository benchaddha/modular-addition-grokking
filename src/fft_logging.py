import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _resolve_path(obj: Any, path: str) -> Any:
    current = obj
    for token in path.split("."):
        if token.isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def power_spectrum(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().float().reshape(-1)
    values = values - values.mean()
    spectrum = torch.fft.rfft(values)
    power = spectrum.abs().pow(2)
    return power


def top_frequency_bins(
    power: torch.Tensor,
    top_k: int = 5,
    skip_dc: bool = True,
) -> List[Dict[str, float]]:
    if power.numel() == 0:
        return []

    start_idx = 1 if skip_dc and power.numel() > 1 else 0
    candidate = power[start_idx:]
    if candidate.numel() == 0:
        candidate = power
        start_idx = 0

    k = min(top_k, candidate.numel())
    top_power, top_idx = torch.topk(candidate, k=k)
    total = power.sum().item() + 1e-12

    rows: List[Dict[str, float]] = []
    for idx, value in zip(top_idx.tolist(), top_power.tolist()):
        rows.append(
            {
                "freq_bin": float(idx + start_idx),
                "power": float(value),
                "relative_power": float(value / total),
            }
        )
    return rows


def append_fft_log(
    model: Any,
    tensor_path: str,
    epoch: int,
    output_path: Path,
    top_k: int = 5,
) -> Dict[str, float]:
    tensor = _resolve_path(model, tensor_path)
    power = power_spectrum(tensor)
    top_bins = top_frequency_bins(power, top_k=top_k)

    summary = {
        "epoch": float(epoch),
        "tensor_path": tensor_path,
        "num_bins": float(power.numel()),
    }
    if top_bins:
        summary["peak_freq_bin"] = top_bins[0]["freq_bin"]
        summary["peak_relative_power"] = top_bins[0]["relative_power"]
    else:
        summary["peak_freq_bin"] = 0.0
        summary["peak_relative_power"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "top_bins": top_bins}
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return summary
