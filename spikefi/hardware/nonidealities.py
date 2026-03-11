# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6
"""CIM non-ideality models.

Tier 1 — Weight-level (:class:`CimNoise` subclasses): perturb
``W_pos/neg_bits_f`` in the :class:`~spikefi.hardware.cim_array.CimArray`
before each VMM call.

Tier 2 — VMM-level (:class:`IrDropNoise`): sampled inside the bit-serial
loop by :mod:`spikefi.hardware.engine` at each bit-plane.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch


# Tier 1 — Weight-level perturbation

class CimNoise(ABC):
    """Abstract base for weight-level CIM non-ideality models.

    *call_every* controls scheduling:

    * ``0`` — **static**: applied once at array build time by :func:`~spikefi.hardware.conversion.to_cim`.
      Use for device-to-device variation (fixed at fabrication).
    * ``N ≥ 1`` — **dynamic**: the layer re-applies this model every *N* forward
      passes.  Use for cycle-to-cycle variation or wear-out accumulation.
    """

    def __init__(self, call_every: int = 0) -> None:
        if call_every < 0:
            raise ValueError("call_every must be >= 0.")
        self.call_every = call_every

    @abstractmethod
    def apply(self, array: "CimArray") -> None:  # noqa: F821
        """Perturb the weight representation stored in *array* in-place.

        Write into ``array.W_pos_bits_f`` / ``array.W_neg_bits_f`` (float
        bit-planes used by the engine).  Do **not** modify the uint8 originals
        ``array.W_pos_bits`` / ``array.W_neg_bits``.
        """
        ...

    def __str__(self) -> str:  return f"{type(self).__name__}(call_every={self.call_every})"
    def __repr__(self) -> str: return f"{type(self).__name__}(call_every={self.call_every!r})"


class ConductanceVariation(CimNoise):
    """Device-to-device / cycle-to-cycle conductance spread.

    Models relative Gaussian noise on each cell: ``G ← G_nom · N(1, sigma)``.
    Re-sampled each call so ``call_every=1`` gives cycle-to-cycle variation;
    ``call_every=0`` gives a static device-to-device draw.

    .. warning::
        ``apply()`` is not yet implemented and raises :exc:`NotImplementedError`.
    """

    def __init__(self, sigma: float = 0.05, call_every: int = 1) -> None:
        super().__init__(call_every)
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        self.sigma = sigma

    def apply(self, array: "CimArray") -> None:  # noqa: F821
        raise NotImplementedError(
            "ConductanceVariation.apply() is not yet implemented.\n"
            "Planned: sample N(1, sigma) multiplicative noise on W_pos/neg_bits_f."
        )

    def __str__(self) -> str:  return f"ConductanceVariation(sigma={self.sigma}, call_every={self.call_every})"
    def __repr__(self) -> str: return f"ConductanceVariation(sigma={self.sigma!r}, call_every={self.call_every!r})"


class ReadDisturb(CimNoise):
    """Probabilistic bit-flip from repeated read cycling.

    Each read flips a stored bit with probability *p_flip* per cell.
    Use ``call_every=1`` to accumulate wear-out across forward passes.

    .. warning::
        ``apply()`` is not yet implemented and raises :exc:`NotImplementedError`.
    """

    def __init__(self, p_flip: float = 1e-5, call_every: int = 1) -> None:
        super().__init__(call_every)
        if not (0.0 <= p_flip <= 1.0):
            raise ValueError("p_flip must be in [0, 1].")
        self.p_flip = p_flip

    def apply(self, array: "CimArray") -> None:  # noqa: F821
        raise NotImplementedError(
            "ReadDisturb.apply() is not yet implemented.\n"
            "Planned: Bernoulli(p_flip) XOR mask on W_pos/neg_bits_f."
        )

    def __str__(self) -> str:  return f"ReadDisturb(p_flip={self.p_flip}, call_every={self.call_every})"
    def __repr__(self) -> str: return f"ReadDisturb(p_flip={self.p_flip!r}, call_every={self.call_every!r})"


# Tier 2 — VMM-level perturbation

class IrDropNoise:
    """IR-drop noise model for bit-serial CIM crossbars.

    Wraps a pre-fitted LUT of Beta-distribution parameters indexed by
    ``(input_density, weight_density)`` pairs.  During inference the compute
    engine samples multiplicative noise from this LUT and applies it to the
    partial sums ``Y_pos_b`` / ``Y_neg_b`` at each bit-plane.

    This class is **not** a :class:`CimNoise` subclass; pass instances via
    :class:`~spikefi.hardware.conversion.CimConfig` or directly to
    :class:`~spikefi.hardware.cim_layers.CimConv` /
    :class:`~spikefi.hardware.cim_layers.CimDense`.
    """

    def __init__(
        self,
        input_bins: torch.Tensor,
        weight_bins: torch.Tensor,
        columns: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        source_flag: torch.Tensor,
        *,
        clamp: tuple[float, float] | None = None,
        build_pair_lut: bool = True,
        pair_lut_max: int = 100,
        pair_lut_step: float = 1.0,
    ) -> None:
        self.input_bins = input_bins
        self.weight_bins = weight_bins
        self.columns = columns
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        self.source_flag = source_flag
        self.clamp = clamp

        # Packed tensors indexed [P, C_max, bits] for fast bit-slice access
        self.a_packed: torch.Tensor | None = None
        self.b_packed: torch.Tensor | None = None
        self.loc_packed: torch.Tensor | None = None
        self.scale_packed: torch.Tensor | None = None
        self.packed_C_max: int | None = None
        self.packed_bits: int | None = None

        # 2-D grid LUT: (input_density_idx, weight_density_idx) → pair_idx
        self.pair_idx_lut: torch.Tensor | None = None
        self.pair_lut_max: int | None = None
        self.pair_lut_step: float | None = None

        if build_pair_lut:
            self.build_pair_index_lut(max_pct=pair_lut_max, step=pair_lut_step)

    # Factories

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        *,
        clamp: tuple[float, float] | None = None,
        build_pair_lut: bool = True,
        pair_lut_max: int = 100,
        pair_lut_step: float = 1.0,
    ) -> "IrDropNoise":
        """Load an :class:`IrDropNoise` from a ``.pt`` LUT file.

        Required tensor keys: ``input_bins``, ``weight_bins``, ``columns``,
        ``a``, ``b``, ``loc``, ``scale``, ``source_flag``.
        """
        path = Path(path)
        raw = torch.load(path, map_location=device)

        def _t(key: str, dt: torch.dtype = dtype) -> torch.Tensor:
            return raw[key].to(device=device, dtype=dt)

        return cls(
            input_bins=_t("input_bins"),
            weight_bins=_t("weight_bins"),
            columns=_t("columns", torch.int64),
            a=_t("a"),
            b=_t("b"),
            loc=_t("loc"),
            scale=_t("scale"),
            source_flag=_t("source_flag", torch.int16),
            clamp=clamp,
            build_pair_lut=build_pair_lut,
            pair_lut_max=pair_lut_max,
            pair_lut_step=pair_lut_step,
        )

    # Beta sampling and LUT indexing

    @staticmethod
    @torch.no_grad()
    def _sample_beta(params: dict[str, torch.Tensor]) -> torch.Tensor:
        # torch._standard_gamma is a private PyTorch API, but is used here deliberately:
        # torch.distributions.Beta.rsample() routes through _Dirichlet.apply (an autograd
        # Function) even inside no_grad, incurring significant Python overhead for the
        # large [N, n_rt, n_ct, C_max] tensors produced during CIM inference.
        # _standard_gamma dispatches directly to the CUDA/CPU kernel and is ~5-10x faster.
        # It respects the device of its argument, so callers must ensure params are on the
        # correct device before calling this method.
        x = torch._standard_gamma(params["a"])
        y = torch._standard_gamma(params["b"])
        raw = x / (x + y + 1e-12)
        return raw * params["scale"] + params["loc"]

    @property
    def n_pairs(self) -> int:
        return int(self.input_bins.numel())

    @property
    def n_cols(self) -> int:
        return int(self.columns.numel())

    def _to_tensor(self, x: torch.Tensor | float | int) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=self.input_bins.device, dtype=self.input_bins.dtype)
        return torch.tensor(x, device=self.input_bins.device, dtype=self.input_bins.dtype)

    @torch.no_grad()
    def build_pair_index_lut(
        self, *, max_pct: int = 100, step: float = 1.0
    ) -> None:
        """Pre-compute a 2-D grid (input_density × weight_density) → pair_idx.

        Stored as ``self.pair_idx_lut[i, j]`` where ``i`` and ``j`` are
        density percentages quantised to *step*-sized bins.
        """
        if step <= 0:
            raise ValueError("step must be > 0.")
        dev = self.input_bins.device
        dt = self.input_bins.dtype
        P = self.n_pairs
        n = int(round(max_pct / step)) + 1
        grid = torch.linspace(0.0, float(max_pct), steps=n, device=dev, dtype=dt)
        I, J = torch.meshgrid(grid, grid, indexing="ij")
        N = I.numel()
        in_flat = I.reshape(N, 1)
        w_flat = J.reshape(N, 1)
        dist2 = (
            (self.input_bins.view(1, P) - in_flat).square()
            + (self.weight_bins.view(1, P) - w_flat).square()
        )
        self.pair_idx_lut = dist2.argmin(dim=1).view(n, n).to(torch.long)
        self.pair_lut_max = max_pct
        self.pair_lut_step = float(step)

    def nearest_pair_indices(
        self,
        input_density: torch.Tensor | float | int,
        weight_density: torch.Tensor | float | int,
    ) -> torch.Tensor:
        """Return closest LUT pair index for each (input_density, weight_density) entry.

        Uses the pre-built 2-D grid LUT when available; falls back to exact
        O(N·P) search otherwise.
        """
        in_d = self._to_tensor(input_density)
        w_d = self._to_tensor(weight_density)
        if self.pair_idx_lut is not None:
            step = self.pair_lut_step
            n = self.pair_idx_lut.shape[0]
            i = torch.round(in_d / step).clamp(0, n - 1).to(torch.long)
            j = torch.round(w_d / step).clamp(0, n - 1).to(torch.long)
            return self.pair_idx_lut[i, j]
        # Fallback: exact O(N·P) search
        P = self.n_pairs
        in_flat = in_d.reshape(-1)
        w_flat = w_d.reshape(-1)
        dist2 = (
            (self.input_bins.view(1, P) - in_flat.view(-1, 1)).square()
            + (self.weight_bins.view(1, P) - w_flat.view(-1, 1)).square()
        )
        return dist2.argmin(dim=1).view(in_d.shape)

    @torch.no_grad()
    def pack_params(self, *, C_max: int, bits: int, drop_flat: bool = False) -> None:
        """Reshape flat ``[P, C_real]`` params into ``[P, C_max, bits]`` for fast bit-slice access.

        ``C_real`` must equal ``C_max * bits``.  Set *drop_flat* to free the
        original flat tensors after packing.
        """
        P = self.n_pairs
        C_real = self.n_cols
        expected = C_max * bits
        if C_real != expected:
            raise ValueError(
                f"Cannot pack: LUT has C_real={C_real}, "
                f"expected C_max*bits={C_max}*{bits}={expected}."
            )
        self.a_packed = self.a.view(P, C_max, bits)
        self.b_packed = self.b.view(P, C_max, bits)
        self.loc_packed = self.loc.view(P, C_max, bits)
        self.scale_packed = self.scale.view(P, C_max, bits)
        self.packed_C_max = int(C_max)
        self.packed_bits = int(bits)
        if drop_flat:
            self.a = self.b = self.loc = self.scale = None  # type: ignore[assignment]

    def params_for_pair_idx_bit(
        self,
        pair_idx: torch.Tensor,
        b: int,
    ) -> dict[str, torch.Tensor]:
        """Return Beta params for bit-plane *b*, shape ``(*pair_idx.shape, C_max)``."""
        if self.a_packed is None:
            raise RuntimeError("Call pack_params(C_max=..., bits=...) first.")
        if pair_idx.dtype != torch.long:
            pair_idx = pair_idx.long()
        dev = self.a_packed.device
        pair_idx = pair_idx.to(dev)
        bits = self.packed_bits
        # LUT params are packed MSB-first in the last axis: index 0 = bit (bits-1).
        # Invert b so that caller's b=0 (LSB) maps to the correct packed slice.
        k = bits - 1 - int(b)
        idx_flat = pair_idx.reshape(-1)
        a_sel = self.a_packed[:, :, k][idx_flat]
        b_sel = self.b_packed[:, :, k][idx_flat]
        loc_sel = self.loc_packed[:, :, k][idx_flat]
        sc_sel = self.scale_packed[:, :, k][idx_flat]
        out_shape = (*pair_idx.shape, int(self.packed_C_max))
        return {
            "a": a_sel.view(out_shape),
            "b": b_sel.view(out_shape),
            "loc": loc_sel.view(out_shape),
            "scale": sc_sel.view(out_shape),
        }

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> "IrDropNoise":
        """Return a copy of this noise model moved to *device*."""
        if dtype is None:
            dtype = self.a.dtype

        def _mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device, dtype=dtype)

        out = IrDropNoise(
            input_bins=self.input_bins.to(device=device, dtype=dtype),
            weight_bins=self.weight_bins.to(device=device, dtype=dtype),
            columns=self.columns.to(device=device),
            a=self.a.to(device=device, dtype=dtype),
            b=self.b.to(device=device, dtype=dtype),
            loc=self.loc.to(device=device, dtype=dtype),
            scale=self.scale.to(device=device, dtype=dtype),
            source_flag=self.source_flag.to(device=device),
            clamp=self.clamp,
            build_pair_lut=False,
        )
        out.a_packed = _mv(self.a_packed)
        out.b_packed = _mv(self.b_packed)
        out.loc_packed = _mv(self.loc_packed)
        out.scale_packed = _mv(self.scale_packed)
        out.packed_C_max = self.packed_C_max
        out.packed_bits = self.packed_bits
        if self.pair_idx_lut is not None:
            out.pair_idx_lut = self.pair_idx_lut.to(device=device)
            out.pair_lut_max = self.pair_lut_max
            out.pair_lut_step = self.pair_lut_step
        return out

    def __str__(self) -> str:
        packed = f", packed(C_max={self.packed_C_max})" if self.a_packed is not None else ""
        return f"IrDropNoise(pairs={self.n_pairs}, cols={self.n_cols}{packed})"

    def __repr__(self) -> str:
        return (
            f"IrDropNoise("
            f"n_pairs={self.n_pairs!r}, n_cols={self.n_cols!r}, "
            f"packed={self.a_packed is not None!r}, "
            f"clamp={self.clamp!r})"
        )
