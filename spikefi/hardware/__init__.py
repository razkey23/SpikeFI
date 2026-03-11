# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024 Theofilos Spyrou, Sorbonne Université, CNRS, LIP6

"""spikefi.hardware — CIM hardware backend for SpikeFI.

Provides a bit-serial CIM simulation layer sitting between an ideal
SLAYER SNN and the SpikeFI fault-injection campaign.

Quick start::

    import spikefi.hardware as hw

    cfg = hw.CimConfig(
        bits=8,
        tiling_cfg={"fc1": {"R_max": 256, "C_max": 64}},
        ir_drop="lut.pt",           # auto-loaded and auto-packed
        conductance_variation=0.05, # 5 % D2D sigma
    )
    cim_model = hw.to_cim(model, cfg)
    output = cim_model(input_spikes)
"""

__all__ = [
    # Configuration
    "CimConfig",
    # Data structure
    "CimArray",
    # Non-idealities (tier 1 — weight-level)
    "CimNoise",
    "ConductanceVariation",
    "ReadDisturb",
    # Non-idealities (tier 2 — VMM-level)
    "IrDropNoise",
    # Layer modules
    "CimLayer",
    "CimConv",
    "CimDense",
    # Entry point
    "to_cim",
]

from spikefi.hardware.cim_array import CimArray
from spikefi.hardware.nonidealities import (
    CimNoise,
    ConductanceVariation,
    IrDropNoise,
    ReadDisturb,
)
from spikefi.hardware.cim_layers import CimLayer, CimConv, CimDense
from spikefi.hardware.conversion import CimConfig, to_cim
