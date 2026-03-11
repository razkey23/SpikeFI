# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6
"""Bit-serial signed VMM compute kernels.

Weight-level non-idealities (:class:`~spikefi.hardware.nonidealities.CimNoise`
subclasses) are applied **before** this engine is called, from the layer's
``forward()`` method.

VMM-level IR-drop noise (:class:`~spikefi.hardware.nonidealities.IrDropNoise`) is
applied **inside** the bit-serial accumulation loop via the optional
``ir_drop`` argument to :func:`signed_vmm` / :func:`signed_vmm_chunk`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from spikefi.hardware.cim_array import CimArray

if TYPE_CHECKING:
    from spikefi.hardware.nonidealities import IrDropNoise


def signed_vmm_chunk(
    X_chunk: torch.Tensor,
    array: CimArray,
    simulate_bitserial: bool = True,
    ir_drop: "IrDropNoise | None" = None,
) -> torch.Tensor:
    """Bit-serial signed VMM for one chunk of input vectors ``[N_chunk, K]``.

    When *simulate_bitserial* is ``True`` accumulates bit-by-bit (hardware-
    faithful); ``False`` uses a vectorised shortcut valid only for ideal weights.
    IR-drop noise requires ``simulate_bitserial=True``.
    Returns ``[N_chunk, C_out]``, same dtype as *X_chunk*.
    """
    if X_chunk.ndim != 2:
        raise ValueError(f"X_chunk must be 2-D [N_chunk, K], got {X_chunk.shape}")

    if ir_drop is not None and not simulate_bitserial:
        raise ValueError(
            "ir_drop requires simulate_bitserial=True; "
            "the vectorised path does not support per-bit noise injection."
        )

    N_chunk, K = X_chunk.shape
    device = X_chunk.device
    out_dtype = X_chunk.dtype

    if K != array.K:
        raise ValueError(f"X_chunk K={K}, but array.K={array.K}")

    n_rt = array.num_row_tiles
    n_ct = array.num_col_tiles
    R_max = array.R_max
    C_max = array.C_max
    bits = array.bits
    C_out = array.C_out

    # Populate float bit-plane cache on first call only.
    # If already set (e.g. by _apply_static_noise), do NOT overwrite —
    # that would discard any noise already baked into the float planes.
    if array.W_pos_bits_f is None or array.W_neg_bits_f is None:
        array.prepare_runtime(compute_dtype=X_chunk.dtype)

    W_pos_f = array.W_pos_bits_f
    W_neg_f = array.W_neg_bits_f
    assert W_pos_f is not None and W_neg_f is not None

    compute_dtype = W_pos_f.dtype
    scale = array.scale.to(device=device, dtype=out_dtype)

    # Pad inputs to tile-aligned width
    K_pad = n_rt * R_max
    if K_pad == K:
        X_padded = X_chunk
    else:
        X_padded = torch.zeros(N_chunk, K_pad, device=device, dtype=out_dtype)
        X_padded[:, :K] = X_chunk

    X_tiles = X_padded.view(N_chunk, n_rt, R_max)
    if X_tiles.dtype != compute_dtype:
        X_tiles = X_tiles.to(dtype=compute_dtype)

    if simulate_bitserial:
        # Pre-compute IR-drop pair indices once outside the per-bit loop
        pair_idx_pos = pair_idx_neg = None
        if ir_drop is not None:
            if ir_drop.a_packed is None:
                raise RuntimeError(
                    "IrDropNoise.pack_params(C_max=..., bits=...) must be called "
                    "before passing ir_drop to the engine."
                )
            X_nonzero = (X_tiles.abs() > 0)                           # [N, n_rt, R_max]
            input_density_pct = (
                X_nonzero.sum(dim=2).float() / float(R_max)
            ) * 100.0
            input_density = input_density_pct.unsqueeze(-1).expand(  # [N, n_rt, n_ct]
                -1, -1, n_ct
            ).contiguous()

            pos_w_density = (                                          # [N, n_rt, n_ct]
                array.pos_density_per_tile.float().to(device) * 100.0
            ).unsqueeze(0).expand(N_chunk, -1, -1).contiguous()
            neg_w_density = (
                array.neg_density_per_tile.float().to(device) * 100.0
            ).unsqueeze(0).expand(N_chunk, -1, -1).contiguous()

            pair_idx_pos = ir_drop.nearest_pair_indices(  # [N, n_rt, n_ct]
                input_density, pos_w_density
            )
            pair_idx_neg = ir_drop.nearest_pair_indices(
                input_density, neg_w_density
            )

        # Accumulate weighted partial sums: Y += (Y_pos_b - Y_neg_b) * 2^b
        Y_acc = torch.zeros(
            N_chunk, n_rt, n_ct, C_max, device=device, dtype=torch.float32
        )
        for b in range(bits):
            W_pos_b = W_pos_f[..., b]  # [n_rt, n_ct, R_max, C_max]
            W_neg_b = W_neg_f[..., b]
            Y_pos_b = torch.einsum("nrk,rckm->nrcm", X_tiles, W_pos_b)  # nrk,rckm: N,n_rt,K × n_rt,n_ct,K,C_max
            Y_neg_b = torch.einsum("nrk,rckm->nrcm", X_tiles, W_neg_b)

            # Apply multiplicative IR-drop noise to partial sums
            if ir_drop is not None:
                p_pos = ir_drop.params_for_pair_idx_bit(pair_idx_pos, b)
                p_neg = ir_drop.params_for_pair_idx_bit(pair_idx_neg, b)
                # Move Beta params to the compute device before sampling.
                # The LUT may live on CPU (never explicitly moved); sampling
                # on CPU for large [N, n_rt, n_ct, C_max] tensors is prohibitively slow.
                p_pos = {k: v.to(device=device) for k, v in p_pos.items()}
                p_neg = {k: v.to(device=device) for k, v in p_neg.items()}
                noise_pos = ir_drop._sample_beta(p_pos)  # [N, n_rt, n_ct, C_max]
                noise_neg = ir_drop._sample_beta(p_neg)
                if ir_drop.clamp is not None:
                    noise_pos = noise_pos.clamp(*ir_drop.clamp)
                    noise_neg = noise_neg.clamp(*ir_drop.clamp)
                Y_pos_b = Y_pos_b.float() * noise_pos
                Y_neg_b = Y_neg_b.float() * noise_neg
                del p_pos, p_neg, noise_pos, noise_neg

            Y_acc += (Y_pos_b.float() - Y_neg_b.float()) * float(1 << b)
            del Y_pos_b, Y_neg_b
    else:
        W_pos = array.W_pos_bits.to(device=device, dtype=out_dtype)
        W_neg = array.W_neg_bits.to(device=device, dtype=out_dtype)
        W_signed = W_pos - W_neg
        bit_weights = (2 ** torch.arange(bits, device=device, dtype=out_dtype))
        Y_acc = torch.einsum(
            "nrk,rckmb,b->nrcm", X_tiles.to(out_dtype), W_signed, bit_weights
        )

    # Sum over row tiles then crop to true C_out
    Y_tiles_sum = Y_acc.sum(dim=1)                         # [N_chunk, n_ct, C_max]
    del Y_acc
    Y_int = Y_tiles_sum.reshape(N_chunk, n_ct * C_max)[:, :C_out]
    return Y_int.to(out_dtype) * scale


def signed_vmm(
    X_all: torch.Tensor,
    array: CimArray,
    max_vec_chunk: int = 65_536,
    simulate_bitserial: bool = True,
    ir_drop: "IrDropNoise | None" = None,
) -> torch.Tensor:
    """Chunk-wise bit-serial signed VMM over the full input batch ``[N_vec, K]``.

    Splits into chunks of at most *max_vec_chunk* rows to bound peak memory.
    Returns ``[N_vec, C_out]``, same dtype as *X_all*.
    """
    if X_all.ndim != 2:
        raise ValueError(f"X_all must be 2-D [N_vec, K], got {X_all.shape}")

    N_vec, K = X_all.shape
    if K != array.K:
        raise ValueError(f"X_all K={K}, but array.K={array.K}")

    device = X_all.device
    dtype = X_all.dtype
    C_out = array.C_out
    Y_all = torch.empty(N_vec, C_out, device=device, dtype=dtype)

    for start in range(0, N_vec, max_vec_chunk):
        end = min(start + max_vec_chunk, N_vec)
        Y_chunk = signed_vmm_chunk(
            X_all[start:end], array,
            simulate_bitserial=simulate_bitserial,
            ir_drop=ir_drop,
        )
        Y_all[start:end] = Y_chunk
        del Y_chunk

    return Y_all


def _snn_im2col_vmm(
    v_in: torch.Tensor,
    array: CimArray,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    simulate_bitserial: bool,
    ir_drop: "IrDropNoise | None" = None,
) -> torch.Tensor:
    """Im2col + signed VMM for SLAYER's 5-D spike tensor ``[B, C_in, H, W, T]``.

    Collapses the time axis into the batch dimension before calling
    :func:`signed_vmm`, then restores the original layout on output.
    Bias (stored in ``array.bias``) is added when present.
    Returns ``[B, C_out, H_out, W_out, T]``.
    """
    B, C_in, H, W, T = v_in.shape
    kH, kW = kernel_size
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    pH, pW = (padding, padding) if isinstance(padding, int) else padding

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    L = H_out * W_out
    K_unfold = C_in * kH * kW

    if K_unfold != array.K:
        raise ValueError(
            f"im2col K={K_unfold} != array.K={array.K}. "
            "Check layer dimensions vs tiling configuration."
        )

    # Collapse time into batch: [B, C_in, H, W, T] → [B*T, C_in, H, W]
    v_flat = v_in.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C_in, H, W)

    # im2col → [B*T, K, L] then flatten spatial positions → [B*T*L, K]
    patches = F.unfold(v_flat, kernel_size=(kH, kW), padding=(pH, pW), stride=(sH, sW))
    X_all = patches.transpose(1, 2).contiguous().view(B * T * L, K_unfold)

    Y_all = signed_vmm(X_all, array, simulate_bitserial=simulate_bitserial, ir_drop=ir_drop)

    if array.bias is not None:
        Y_all = Y_all + array.bias.view(1, -1)

    # Restore SLAYER layout: [B*T, L, C_out] → [B, C_out, H_out, W_out, T]
    return (
        Y_all.view(B * T, L, array.C_out)
        .transpose(1, 2)
        .contiguous()
        .view(B, T, array.C_out, H_out, W_out)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
    )
