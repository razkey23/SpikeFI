# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6
"""Tiled bit-serial weight storage for CIM layers."""

from __future__ import annotations

import logging
import math

import torch

_log = logging.getLogger(__name__)


class CimArray:
    """Bit-serial, sign-split weight matrix mapped onto crossbar tiles.

    Weights are quantized → sign-split → zero-padded → tiled into
    ``[n_rt, n_ct, R_max, C_max]`` blocks → bit-decomposed into ``bits``
    uint8 planes.  Noise models are owned by the calling
    :class:`~spikefi.hardware.cim_layers.CimLayer`, not stored here.
    """

    def __init__(
        self,
        K: int,
        C_out: int,
        R_max: int,
        C_max: int,
        num_row_tiles: int,
        num_col_tiles: int,
        bits: int,
        W_pos_bits: torch.Tensor,
        W_neg_bits: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        pos_density_per_tile: torch.Tensor | None = None,
        neg_density_per_tile: torch.Tensor | None = None,
    ) -> None:
        self.K, self.C_out = K, C_out
        self.R_max, self.C_max = R_max, C_max
        self.num_row_tiles, self.num_col_tiles = num_row_tiles, num_col_tiles
        self.bits = bits
        self.W_pos_bits, self.W_neg_bits = W_pos_bits, W_neg_bits
        self.scale, self.bias = scale, bias
        self.pos_density_per_tile = pos_density_per_tile
        self.neg_density_per_tile = neg_density_per_tile
        # Float copies of bit-planes, populated once by prepare_runtime()
        self.W_pos_bits_f: torch.Tensor | None = None
        self.W_neg_bits_f: torch.Tensor | None = None

    @classmethod
    def from_weights(
        cls,
        W_phys_int: torch.Tensor,
        scale: torch.Tensor,
        R_max: int,
        C_max: int,
        bits: int = 8,
        layer_name: str = "",
        bias: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> "CimArray":
        """Build from a signed int16 weight matrix ``[K, C_out]``.

        ``W_float ≈ W_phys_int * scale``.  Device inferred from *W_phys_int*
        when *device* is ``None``.
        """
        if device is None:
            device = W_phys_int.device

        K, C_out = W_phys_int.shape
        W = W_phys_int.to(device=device, dtype=torch.int16)

        # Sign-split: positive and negative magnitudes stored separately
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(-W, min=0)

        n_rt = math.ceil(K / R_max)
        n_ct = math.ceil(C_out / C_max)

        # Zero-pad both halves to exact tile multiples
        W_pos_pad = torch.zeros(n_rt * R_max, n_ct * C_max, device=device, dtype=torch.int16)
        W_neg_pad = torch.zeros(n_rt * R_max, n_ct * C_max, device=device, dtype=torch.int16)
        W_pos_pad[:K, :C_out] = W_pos
        W_neg_pad[:K, :C_out] = W_neg

        def _tile_and_decompose(W_pad: torch.Tensor) -> torch.Tensor:
            # Tile: [K_pad, C_pad] → [n_rt, n_ct, R_max, C_max]
            tiles = W_pad.view(n_rt, R_max, n_ct, C_max).permute(0, 2, 1, 3).contiguous()
            # Bit-decompose: right-shift by 0..bits-1 and mask LSB → uint8
            shifts = torch.arange(bits, device=device, dtype=torch.int16).view(1, 1, 1, 1, bits)
            return ((tiles.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)

        W_pos_bits = _tile_and_decompose(W_pos_pad)
        W_neg_bits = _tile_and_decompose(W_neg_pad)

        # Per-tile density: used by IrDropNoise to look up the correct Beta params.
        # Density is averaged across all bit-planes — a single aggregate per tile
        # is used for every bit in the VMM loop (an approximation; per-bit density
        # would be more accurate but significantly more expensive).
        vol = R_max * C_max * bits
        pos_density = (W_pos_bits > 0).sum(dim=(2, 3, 4)).float() / vol  # [n_rt, n_ct]
        neg_density = (W_neg_bits > 0).sum(dim=(2, 3, 4)).float() / vol  # [n_rt, n_ct]

        scale_t = torch.as_tensor(scale, device=device, dtype=torch.float32)
        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        _log.debug(
            "CimArray.from_weights: %s  tiles=%d  (K=%d, C_out=%d)  bits=%d",
            layer_name or "<unnamed>",
            n_rt * n_ct,
            K,
            C_out,
            bits,
        )

        return cls(
            K=K, C_out=C_out, R_max=R_max, C_max=C_max,
            num_row_tiles=n_rt, num_col_tiles=n_ct, bits=bits,
            W_pos_bits=W_pos_bits, W_neg_bits=W_neg_bits,
            scale=scale_t, bias=bias,
            pos_density_per_tile=pos_density, neg_density_per_tile=neg_density,
        )

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> "CimArray":
        """Return a copy moved to *device* (float tensors optionally recast to *dtype*)."""
        if dtype is None:
            dtype = self.scale.dtype

        def _mv(t: torch.Tensor | None, dt: torch.dtype = dtype) -> torch.Tensor | None:
            return None if t is None else t.to(device=device, dtype=dt)

        out = CimArray(
            K=self.K, C_out=self.C_out, R_max=self.R_max, C_max=self.C_max,
            num_row_tiles=self.num_row_tiles, num_col_tiles=self.num_col_tiles,
            bits=self.bits,
            W_pos_bits=self.W_pos_bits.to(device=device),
            W_neg_bits=self.W_neg_bits.to(device=device),
            scale=_mv(self.scale), bias=_mv(self.bias),
            pos_density_per_tile=_mv(self.pos_density_per_tile),
            neg_density_per_tile=_mv(self.neg_density_per_tile),
        )
        # Preserve noise-perturbed float planes across device moves.
        # Without this, static noise baked in before .to() is silently discarded
        # and prepare_runtime() would regenerate clean planes from the uint8 originals.
        if self.W_pos_bits_f is not None:
            out.W_pos_bits_f = self.W_pos_bits_f.to(device=device, dtype=dtype)
        if self.W_neg_bits_f is not None:
            out.W_neg_bits_f = self.W_neg_bits_f.to(device=device, dtype=dtype)
        return out

    def prepare_runtime(self, *, compute_dtype: torch.dtype) -> None:
        """Cast uint8 bit-planes to *compute_dtype* once before the first forward pass.

        Avoids repeated dtype casts inside the hot VMM loop.  Call again after
        to() if the device changes.

        warning:
            This re-casts from the clean uint8 originals (``W_pos_bits`` /
            ``W_neg_bits``), overwriting any noise already written into
            ``W_pos_bits_f`` / ``W_neg_bits_f``.  Static noise models
            (``call_every=0``) are applied *after* this call by
            :func:`~spikefi.hardware.conversion._apply_static_noise`.
        """
        dev = self.W_pos_bits.device
        self.W_pos_bits_f = self.W_pos_bits.to(device=dev, dtype=compute_dtype)
        self.W_neg_bits_f = self.W_neg_bits.to(device=dev, dtype=compute_dtype)

    def __repr__(self) -> str:
        return (
            f"CimArray("
            f"K={self.K!r}, C_out={self.C_out!r}, "
            f"R_max={self.R_max!r}, C_max={self.C_max!r}, "
            f"tiles={self.num_row_tiles * self.num_col_tiles!r}, "
            f"bits={self.bits!r}, "
            f"bias={'yes' if self.bias is not None else 'no'})"
        )

    def __str__(self) -> str:
        return (
            f"CimArray[K={self.K}, C_out={self.C_out}, "
            f"{self.num_row_tiles}×{self.num_col_tiles} tiles @ "
            f"{self.R_max}×{self.C_max}, {self.bits}b"
            f"{', +bias' if self.bias is not None else ''}]"
        )
