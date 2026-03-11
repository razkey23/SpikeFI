# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6
"""CIM-aware nn.Module drop-in replacements for SLAYER conv and dense layers.

:class:`CimConv` replaces ``slayerSNN._convLayer``;
:class:`CimDense` replaces ``slayerSNN._denseLayer``.
Both are thin subclasses of :class:`CimLayer` which owns all shared logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spikefi.hardware.cim_array import CimArray
from spikefi.hardware.engine import _snn_im2col_vmm, signed_vmm
from spikefi.hardware.nonidealities import CimNoise, IrDropNoise


class CimLayer(nn.Module):
    """Shared base for bit-serial CIM layers.

    Manages the :class:`~spikefi.hardware.cim_array.CimArray`, the list of
    weight-level noise models, optional IR-drop noise, and the ``_noise_step``
    counter that drives ``call_every`` scheduling.

    *kernel_size*, *stride*, *padding* follow the same conventions as
    ``torch.nn.Conv2d``.  Set ``simulate_bitserial=False`` for a faster
    vectorised path that is valid only for ideal (noiseless) weights.
    """

    def __init__(
        self,
        array: CimArray,
        kernel_size: tuple[int, int] | int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        simulate_bitserial: bool = True,
        noise_models: list[CimNoise] | None = None,
        ir_drop: IrDropNoise | None = None,
    ) -> None:
        super().__init__()
        self.array = array
        # Normalise all geometry to 2-tuples so downstream code never needs to branch.
        self.kernel_size: tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride: tuple[int, int] = (
            (stride, stride) if isinstance(stride, int) else stride
        )
        self.padding: tuple[int, int] = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        self.simulate_bitserial = simulate_bitserial
        self.noise_models: list[CimNoise] = list(noise_models) if noise_models else []
        self.ir_drop: IrDropNoise | None = ir_drop
        self._noise_step: int = 0
        # Names of static (call_every=0) noise models applied at build time by to_cim().
        # Kept for repr/str only; the models themselves are not re-applied during forward.
        self._applied_static_names: list[str] = []

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized ``[C_out, K, 1, 1]`` float weight for SpikeFI hook compatibility.

        Reconstructs the weight from tiled bit-planes so SpikeFI's synapse
        fault-injection hooks can read and write it in a consistent format.
        """
        device = self.array.W_pos_bits.device
        bits = self.array.bits
        # Bit-weighted sum: value = Σ_b (bit_b × 2^b).
        # Plain .sum(-1) is wrong — it treats all bit-planes equally, returning 1
        # for any set bit instead of the correct reconstructed integer magnitude.
        bit_weights = (
            2 ** torch.arange(bits, device=device, dtype=torch.int32)
        ).view(1, 1, 1, 1, bits)
        W_pos_int = (self.array.W_pos_bits.to(torch.int32) * bit_weights).sum(-1)
        W_neg_int = (self.array.W_neg_bits.to(torch.int32) * bit_weights).sum(-1)
        W_int = (W_pos_int - W_neg_int).to(torch.int16)
        n_rt, n_ct, R_max, C_max = (
            self.array.num_row_tiles, self.array.num_col_tiles,
            self.array.R_max, self.array.C_max,
        )
        W_phys = (
            W_int.permute(0, 2, 1, 3)
            .contiguous()
            .view(n_rt * R_max, n_ct * C_max)
            [:self.array.K, :self.array.C_out]
        ).float() * self.array.scale
        return W_phys.t().unsqueeze(-1).unsqueeze(-1)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        # Only dynamic noise models (call_every > 0) remain here after to_cim().
        # Static ones (call_every == 0) were already applied at build time.
        # Both CimConv and CimDense route through _snn_im2col_vmm because SLAYER
        # implements dense layers as 1×1 convolutions internally.
        self._noise_step += 1
        for nm in self.noise_models:
            if self._noise_step % nm.call_every == 0:
                nm.apply(self.array)
        return _snn_im2col_vmm(
            v_in, self.array, self.kernel_size,
            self.stride, self.padding,
            self.simulate_bitserial,
            ir_drop=self.ir_drop,
        )

    def _active_nonidealities(self) -> list[str]:
        # Static models were removed from noise_models at build time; their names
        # are preserved in _applied_static_names for display purposes only.
        parts = list(self._applied_static_names)
        parts += [type(nm).__name__ for nm in self.noise_models]
        if self.ir_drop is not None:
            parts.append("IrDrop")
        return parts


# ---------------------------------------------------------------------------
# Concrete layer types
# ---------------------------------------------------------------------------

class CimConv(CimLayer):
    """Bit-serial CIM convolutional layer — drop-in for ``slayerSNN._convLayer``.

    No bias is applied (SLAYER conv layers carry no bias term).
    """

    def __repr__(self) -> str:
        ni = self._active_nonidealities()
        ni_str = f", nonidealities=[{', '.join(ni)}]" if ni else ""
        return (
            f"CimConv(K={self.array.K!r}, C_out={self.array.C_out!r}, "
            f"kernel_size={self.kernel_size!r}, stride={self.stride!r}, "
            f"padding={self.padding!r}, "
            f"simulate_bitserial={self.simulate_bitserial!r}{ni_str})"
        )

    def __str__(self) -> str:
        ni = self._active_nonidealities()
        suffix = (", " + ", ".join(ni)) if ni else ""
        return (
            f"CimConv[{self.array.K}->{self.array.C_out}, "
            f"k={self.kernel_size}, {self.array.bits}b{suffix}]"
        )


class CimDense(CimLayer):
    """Bit-serial CIM dense layer — drop-in for ``slayerSNN._denseLayer``.

    Applies ``array.bias`` when present.
    """

    def __repr__(self) -> str:
        ni = self._active_nonidealities()
        ni_str = f", nonidealities=[{', '.join(ni)}]" if ni else ""
        return (
            f"CimDense(K={self.array.K!r}, C_out={self.array.C_out!r}, "
            f"kernel_size={self.kernel_size!r}, stride={self.stride!r}, "
            f"padding={self.padding!r}, "
            f"simulate_bitserial={self.simulate_bitserial!r}{ni_str})"
        )

    def __str__(self) -> str:
        ni = self._active_nonidealities()
        suffix = (", " + ", ".join(ni)) if ni else ""
        return (
            f"CimDense[{self.array.K}->{self.array.C_out}, "
            f"k={self.kernel_size}, {self.array.bits}b{suffix}]"
        )
