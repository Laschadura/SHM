# ─────────── src/config_schema.py ───────────
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Optional

@dataclass
class LossWeights:
    # canonical, sweep-friendly fields  ────────────────────────────
    mag_mse          : float = 1.0
    phase_dot        : float = 1.0
    phase_if         : float = 0.5
    phase_aw_abs     : float = 0.0
    time_consistency : float = 0.5
    wave_l1          : float = 2.0

    mag_curric_min   : float = 0.3
    mag_curric_max   : float = 1.5

    mask_px          : float = 0.0
    dice_w           : float = 0.0
    damage_initial   : float = 2.0
    damage_final     : float = 0.3
    focal_gamma_init : float = 1.0
    focal_gamma_late : float = 1.5

    # ─── convenience read-only aliases (no impact on Hydra) ───────
    @property
    def MAG_MSE_W(self):        return self.mag_mse
    @property
    def PHASE_DOT_W(self):      return self.phase_dot
    @property
    def PHASE_IF_W(self):       return self.phase_if
    @property
    def PHASE_AW_ABS_W(self):   return self.phase_aw_abs
    @property
    def TIME_CONSIST_W(self):   return self.time_consistency
    @property
    def WAVE_L1_W(self):        return self.wave_l1
    @property
    def MASK_PX_W(self):        return self.mask_px
# -----------------------------------------------------------------
@dataclass
class MainConfig:
    diffusion     : dict
    ae            : dict
    loss_weights  : LossWeights
    debug_mode    : bool = False
    debug_loss_weights: Optional[LossWeights] = None

cs = ConfigStore.instance()
cs.store(name="main_config",          node=MainConfig)
cs.store(group="loss_weights", name="base",  node=LossWeights)
cs.store(group="loss_weights", name="debug", node=LossWeights)
