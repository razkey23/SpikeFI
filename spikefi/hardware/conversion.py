# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6
"""Convert a SLAYER SNN model to its bit-serial CIM equivalent.

    cfg = hw.CimConfig(bits=8, ir_drop="lut.pt", conductance_variation=0.05)
    cim_model = hw.to_cim(model, cfg)   # deep copy; original unchanged
    output = cim_model(input_spikes)
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from slayerSNN import slayer as slayer_lib
from spikefi.hardware.cim_array import CimArray
from spikefi.hardware.cim_layers import CimConv, CimDense
from spikefi.hardware.nonidealities import (
    CimNoise,
    ConductanceVariation,
    IrDropNoise,
    ReadDisturb,
)


# Hardware configuration

@dataclass
class CimConfig:
    """Hardware simulation settings consumed by :func:`to_cim`.

    ``CimConfig()`` gives an 8-bit, untiled, ideal (noiseless) simulation.

    *tiling_cfg* maps dot-separated layer names (from ``model.named_modules()``)
    to ``{"R_max": int, "C_max": int}`` dicts.  Fallback keys
    ``"conv_default"`` / ``"dense_default"`` are checked when no exact match
    exists; omitting them maps the whole weight matrix to a single tile.

    *ir_drop* accepts ``None`` (off), a path string / :class:`~pathlib.Path`
    (auto-loaded and packed), or an :class:`~spikefi.hardware.nonidealities.IrDropNoise`
    instance (auto-packed if not already).  Requires ``simulate_bitserial=True``.
    """

    bits: int = 8
    tiling_cfg: dict[str, dict] = field(default_factory=dict)
    simulate_bitserial: bool = True

    # Non-ideality flags
    ir_drop: IrDropNoise | str | Path | None = None
    conductance_variation: float | None = None
    read_disturb: float | None = None
    # call_every schedules for each noise type.
    # ConductanceVariation models D2D variation — physically fixed at fabrication,
    # so the default is 0 (applied once at build time by to_cim).
    # ReadDisturb models cycle-to-cycle wear-out — default 1 (every forward pass).
    conductance_call_every: int = 0
    read_disturb_call_every: int = 1

    def has_nonidealities(self) -> bool:
        """Return ``True`` if any non-ideality is enabled."""
        return (
            self.ir_drop is not None
            or self.conductance_variation is not None
            or self.read_disturb is not None
        )


# Private helpers

def _quantize_symmetric(
    w: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor quantization → ``(w_dequant, q_int16, scale)``."""
    if bits <= 0:
        raise ValueError("bits must be a positive integer.")
    qmax = (1 << (bits - 1)) - 1
    with torch.no_grad():
        max_abs = w.abs().max()
        if max_abs == 0:
            return (
                torch.zeros_like(w),
                torch.zeros_like(w, dtype=torch.int16),
                torch.tensor(1.0, device=w.device, dtype=torch.float32),
            )
        scale = max_abs / qmax
        q = torch.round(w / scale).clamp(-qmax, qmax)
        q_int = q.to(torch.int16)
        w_q = q_int.float() * scale
    return w_q, q_int, scale


def _quantize_model(model: nn.Module, bits: int) -> dict[str, dict]:
    """Quantize all ``*.weight`` parameters in-place.

    Returns ``{param_name: {"q_int": Tensor, "scale": Tensor}}`` for each
    weight, consumed by the :class:`~spikefi.hardware.cim_array.CimArray` factory.
    """
    quant_info: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        w_q, q_int, scale = _quantize_symmetric(param.data, bits)
        param.data.copy_(w_q)
        quant_info[name] = {"q_int": q_int, "scale": scale}
    return quant_info


def _resolve_ir_drop(
    ir_drop_cfg: IrDropNoise | str | Path | None,
    bits: int,
) -> IrDropNoise | None:
    """Load and auto-pack an :class:`IrDropNoise` from the config field.

    Paths are loaded via :meth:`IrDropNoise.from_file`; un-packed instances
    are packed with ``C_max = n_cols // bits``.
    """
    if ir_drop_cfg is None:
        return None
    if isinstance(ir_drop_cfg, (str, Path)):
        ir_noise = IrDropNoise.from_file(ir_drop_cfg)
    else:
        ir_noise = ir_drop_cfg
    if ir_noise.a_packed is None:
        C_max_inferred = ir_noise.n_cols // bits
        ir_noise.pack_params(C_max=C_max_inferred, bits=bits)
    return ir_noise


def _build_noise_models(cfg: CimConfig) -> list[CimNoise]:
    """Instantiate weight-level noise models from the config flags."""
    models: list[CimNoise] = []
    if cfg.conductance_variation is not None:
        models.append(
            ConductanceVariation(
                sigma=cfg.conductance_variation,
                call_every=cfg.conductance_call_every,
            )
        )
    if cfg.read_disturb is not None:
        models.append(
            ReadDisturb(
                p_flip=cfg.read_disturb,
                call_every=cfg.read_disturb_call_every,
            )
        )
    return models


def _array_from_conv(
    weight_q_int: torch.Tensor,
    scale: torch.Tensor,
    layer_name: str,
    tiling_cfg: dict[str, dict],
    bits: int,
) -> CimArray:
    """Build a :class:`CimArray` from a Conv3d-style weight ``[C_out, C_in, kH, kW, 1]``."""
    w_4d = weight_q_int[..., 0]           # drop trailing 1 → [C_out, C_in, kH, kW]
    C_out, C_in, kH, kW = w_4d.shape
    K = C_in * kH * kW
    W_phys = w_4d.view(C_out, K).t().contiguous()  # [K, C_out]

    cfg_tile = tiling_cfg.get(
        layer_name,
        tiling_cfg.get("conv_default", {"R_max": K, "C_max": C_out}),
    )
    return CimArray.from_weights(
        W_phys_int=W_phys,
        scale=scale,
        R_max=cfg_tile["R_max"],
        C_max=cfg_tile["C_max"],
        bits=bits,
        layer_name=layer_name,
        bias=None,
    )


def _array_from_dense(
    weight_q_int: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: str,
    tiling_cfg: dict[str, dict],
    bits: int,
) -> CimArray:
    """Build a :class:`CimArray` from a Dense Conv3d-style weight ``[C_out, C_in, kH, kW, 1]``."""
    w_4d = weight_q_int[..., 0]           # drop trailing 1 → [C_out, C_in, kH, kW]
    C_out, C_in, kH, kW = w_4d.shape
    K = C_in * kH * kW
    W_phys = w_4d.view(C_out, K).t().contiguous()  # [K, C_out]

    cfg_tile = tiling_cfg.get(
        layer_name,
        tiling_cfg.get("dense_default", {"R_max": K, "C_max": C_out}),
    )
    return CimArray.from_weights(
        W_phys_int=W_phys,
        scale=scale,
        R_max=cfg_tile["R_max"],
        C_max=cfg_tile["C_max"],
        bits=bits,
        layer_name=layer_name,
        bias=bias,
    )


def _apply_static_noise(layer: "CimLayer", compute_dtype: torch.dtype = torch.float32) -> None:  # noqa: F821
    """Apply ``call_every=0`` (static) noise models immediately after array construction.

    ``prepare_runtime()`` is called first to populate the float bit-plane caches
    that noise models write into.  Static models are then removed from the layer's
    list so ``forward()`` never re-applies them.
    """
    layer.array.prepare_runtime(compute_dtype=compute_dtype)
    static = [nm for nm in layer.noise_models if nm.call_every == 0]
    for nm in static:
        nm.apply(layer.array)
    # Record applied names for __str__/__repr__ (models are removed from noise_models below).
    layer._applied_static_names = [f"{type(nm).__name__}(static)" for nm in static]
    # keep only dynamic (call_every > 0) models for forward-pass scheduling
    layer.noise_models = [nm for nm in layer.noise_models if nm.call_every > 0]


# Public API

def to_cim(
    model: nn.Module,
    cfg: CimConfig | None = None,
) -> nn.Module:
    """Return a deep copy of *model* with SLAYER layers replaced by CIM layers.

    Pipeline: deep-copy → quantise weights → replace SLAYER layers with
    :class:`~spikefi.hardware.cim_layers.CimConv` / ``CimDense`` → apply
    static noise (``call_every=0``) once at build time → attach dynamic
    noise and IR-drop for forward-pass dispatch.

    IR-drop is shared across all layers (no per-layer mutable state).
    Weight-level noise models are deep-copied per layer so each has an
    independent ``_noise_step`` counter.
    """
    if cfg is None:
        cfg = CimConfig()

    ir_noise = _resolve_ir_drop(cfg.ir_drop, cfg.bits)
    noise_models = _build_noise_models(cfg)

    model_cim = copy.deepcopy(model)
    quant_info = _quantize_model(model_cim, cfg.bits)

    def _recurse(module: nn.Module, prefix: str = "") -> None:
        # Walk the module tree; replace recognised SLAYER layer types in-place
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            if isinstance(child, slayer_lib._convLayer):
                w_key = full + ".weight"
                if w_key not in quant_info:
                    _recurse(child, full)
                    continue
                q = quant_info[w_key]
                array = _array_from_conv(
                    q["q_int"], q["scale"], full, cfg.tiling_cfg, cfg.bits,
                )
                kH, kW, _ = child.kernel_size
                sH, sW, _ = child.stride
                pH, pW, _ = child.padding
                setattr(module, name, CimConv(
                    array=array,
                    kernel_size=(kH, kW),
                    stride=(sH, sW),
                    padding=(pH, pW),
                    simulate_bitserial=cfg.simulate_bitserial,
                    noise_models=copy.deepcopy(noise_models),
                    ir_drop=ir_noise,
                ))
                _apply_static_noise(getattr(module, name))

            elif isinstance(child, slayer_lib._denseLayer):
                w_key = full + ".weight"
                if w_key not in quant_info:
                    _recurse(child, full)
                    continue
                q = quant_info[w_key]
                array = _array_from_dense(
                    q["q_int"], q["scale"], child.bias, full, cfg.tiling_cfg, cfg.bits,
                )
                kH, kW, _ = child.kernel_size
                sH, sW, _ = child.stride
                pH, pW, _ = child.padding
                setattr(module, name, CimDense(
                    array=array,
                    kernel_size=(kH, kW),
                    stride=(sH, sW),
                    padding=(pH, pW),
                    simulate_bitserial=cfg.simulate_bitserial,
                    noise_models=copy.deepcopy(noise_models),
                    ir_drop=ir_noise,
                ))
                _apply_static_noise(getattr(module, name))

            else:
                _recurse(child, full)

    _recurse(model_cim)
    return model_cim
