"""High-level planner for the 200 m track bonus.

Supports two planner types selected via the JSON config field ``planner_type``:

  * ``"starter_pd"`` – original proportional-derivative baseline (unchanged).
  * ``"mlp"``        – learned MLP planner with CMA-ES-trained weights.

The evaluator entry point is always::

    planner = StarterTrackPlanner.load(path_to_config_json)
    cmd = planner.command(track_obs, t)          # -> np.ndarray shape (3,)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Union

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track

# ── Physical command limits ────────────────────────────────────────────────────
_VX_MIN: float = 0.15   # m/s – never walk slower than this
_VX_MAX: float = 0.50   # m/s – top forward speed
_VY_LIM: float = 0.10   # m/s – lateral correction limit
_YAW_LIM: float = 0.30  # rad/s – yaw rate limit


# ══════════════════════════════════════════════════════════════════════════════
# StarterPlannerConfig  (unchanged from original baseline)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: payload[k] for k in valid if k in payload})

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MLPTrackPlanner  – learned high-level controller
# ══════════════════════════════════════════════════════════════════════════════

class MLPTrackPlanner:
    """Two-hidden-layer MLP that maps the 5-D track observation to [vx, vy, yaw_rate].

    Architecture (default):  5 → 32 → 16 → 3  with tanh activations.

    The output layer uses tanh so all three outputs lie in (−1, 1), which are
    then affinely mapped to the physical command ranges:

        vx        ∈ [_VX_MIN, _VX_MAX]   (always positive → robot moves forward)
        vy        ∈ [−_VY_LIM, _VY_LIM]
        yaw_rate  ∈ [−_YAW_LIM, _YAW_LIM]

    Weights are stored as a .npz file.  The JSON config must contain:

        {
          "planner_type": "mlp",
          "mlp_weights_path": "planner_weights.npz",   // relative to config dir
          "mlp_hidden": [32, 16],                       // hidden layer widths
          "stand_seconds": 1.0
        }
    """

    _OBS_SIZE = 5
    _CMD_SIZE = 3

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        hidden_sizes: list[int],
        stand_seconds: float = 1.0,
    ) -> None:
        self.weights = {k: v.astype(np.float32) for k, v in weights.items()}
        self.hidden_sizes = list(hidden_sizes)
        self.stand_seconds = float(stand_seconds)
        self._n_layers = len(hidden_sizes) + 1

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "MLPTrackPlanner":
        """Load from a JSON config file (weights path is relative to config)."""
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        weights_path = Path(path).parent / cfg["mlp_weights_path"]
        weights = dict(np.load(str(weights_path)))
        return cls(
            weights=weights,
            hidden_sizes=cfg.get("mlp_hidden", [32, 16]),
            stand_seconds=float(cfg.get("stand_seconds", 1.0)),
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Pure NumPy forward pass.  x shape: (5,).  Returns shape (3,)."""
        for i in range(self._n_layers):
            x = x @ self.weights[f"W{i}"] + self.weights[f"b{i}"]
            if i < self._n_layers - 1:
                x = np.tanh(x)
        return np.tanh(x)  # final activation → (−1, 1)

    # ── Planner API ───────────────────────────────────────────────────────────

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        """Return [vx, vy, yaw_rate] command given track observation and time."""
        if t < self.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        raw = self._forward(obs.as_array())
        # Map tanh outputs to physical ranges
        vx = (raw[0] + 1.0) * 0.5 * (_VX_MAX - _VX_MIN) + _VX_MIN
        vy = raw[1] * _VY_LIM
        yaw_rate = raw[2] * _YAW_LIM
        return np.array([vx, vy, yaw_rate], dtype=np.float32)

    # ── Weight helpers used by the training script ────────────────────────────

    @staticmethod
    def make_weights(hidden_sizes: list[int], seed: int = 42) -> dict[str, np.ndarray]:
        """He-initialised weights; output bias set to give vx≈0.35 at start."""
        rng = np.random.default_rng(seed)
        sizes = [MLPTrackPlanner._OBS_SIZE] + hidden_sizes + [MLPTrackPlanner._CMD_SIZE]
        weights: dict[str, np.ndarray] = {}
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            weights[f"W{i}"] = (
                rng.standard_normal((fan_in, sizes[i + 1])) * np.sqrt(2.0 / fan_in)
            ).astype(np.float32)
            weights[f"b{i}"] = np.zeros(sizes[i + 1], dtype=np.float32)
        # Init output bias so vx ≈ 0.35 m/s (mid-range), vy=yaw=0
        # tanh(b) = (0.35 - VX_MIN) / (VX_MAX - VX_MIN) * 2 - 1
        vx_init_norm = (0.35 - _VX_MIN) / (_VX_MAX - _VX_MIN) * 2.0 - 1.0
        last = f"b{len(sizes)-2}"
        weights[last][0] = float(np.arctanh(np.clip(vx_init_norm, -0.99, 0.99)))
        return weights

    @staticmethod
    def pack(weights: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten weight dict to a 1-D float64 array for optimisers."""
        return np.concatenate(
            [weights[k].ravel().astype(np.float64) for k in sorted(weights)]
        )

    @staticmethod
    def unpack(theta: np.ndarray, hidden_sizes: list[int]) -> dict[str, np.ndarray]:
        """Restore weight dict from flat 1-D array."""
        sizes = [MLPTrackPlanner._OBS_SIZE] + hidden_sizes + [MLPTrackPlanner._CMD_SIZE]
        weights: dict[str, np.ndarray] = {}
        offset = 0
        for i in range(len(sizes) - 1):
            n_W = sizes[i] * sizes[i + 1]
            n_b = sizes[i + 1]
            weights[f"W{i}"] = theta[offset: offset + n_W].reshape(
                sizes[i], sizes[i + 1]
            ).astype(np.float32)
            offset += n_W
            weights[f"b{i}"] = theta[offset: offset + n_b].astype(np.float32)
            offset += n_b
        return weights

    @staticmethod
    def param_count(hidden_sizes: list[int]) -> int:
        sizes = [MLPTrackPlanner._OBS_SIZE] + hidden_sizes + [MLPTrackPlanner._CMD_SIZE]
        return sum(
            sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1)
        )


# ══════════════════════════════════════════════════════════════════════════════
# StarterTrackPlanner  – evaluator entry point (dispatches on planner_type)
# ══════════════════════════════════════════════════════════════════════════════

class StarterTrackPlanner:
    """Conservative PD baseline; also dispatches ``load()`` to MLPTrackPlanner.

    Students should improve this class, replace it with an MLP, or train a
    higher-level policy that produces the same [vx, vy, yaw_rate] command.
    """

    def __init__(self, config: StarterPlannerConfig) -> None:
        if config.planner_type != "starter_pd":
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()

    @classmethod
    def load(cls, path: Path) -> Union["StarterTrackPlanner", MLPTrackPlanner]:
        """Load planner from JSON config.  Dispatches to MLPTrackPlanner for 'mlp'."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("planner_type") == "mlp":
            return MLPTrackPlanner.load(path)
        return cls(StarterPlannerConfig.load(path))

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = (
            1.0
            - float(self.config.heading_slowdown)
            * min(abs(heading_error), math.pi)
            / math.pi
        )
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(
            float(self.track.turn_radius_m), 1e-6
        )
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
