"""CMA-ES training script for the learned MLP high-level track planner.

Usage (from the course repo root in Colab):

    python train_highlevel_mlp.py \\
        --checkpoint-dir artifacts/low_level_train/best_checkpoint \\
        --config configs/colab_runtime_config.json \\
        --output-dir artifacts/highlevel_mlp \\
        --iterations 40 \\
        --population 10 \\
        --eval-seconds 15

The script:
  1. Initialises MLP weights (5→32→16→3, tanh).
  2. Runs CMA-ES: each candidate saves a temp config+weights, calls
     run_track_bonus.py, and reads composite_score from results.json.
  3. Saves best weights to <output-dir>/planner_weights.npz and the
     matching config to <output-dir>/planner_config.json.
  4. Writes a search history to <output-dir>/search_history.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── make sure the course repo is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from track_bonus.planner import MLPTrackPlanner

# ══════════════════════════════════════════════════════════════════════════════
# Minimal CMA-ES  (no external dependency)
# ══════════════════════════════════════════════════════════════════════════════

class _CMAes:
    """Simplified (μ/μ_w, λ)-CMA-ES without covariance matrix update.

    Uses a diagonal adaptation so it scales to ~150-300 parameters without
    the O(n²) covariance cost.  Good enough for black-box planner tuning.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float = 0.3,
        popsize: int = 10,
        seed: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n = len(x0)
        self.mean = x0.copy().astype(np.float64)
        self.sigma = float(sigma0)
        self.lam = popsize
        self.mu = max(popsize // 2, 2)

        # Recombination weights
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = raw_w / raw_w.sum()
        self.mueff = 1.0 / float(np.sum(self.w ** 2))

        # Step-size adaptation coefficients (CSA)
        self.cs = (self.mueff + 2.0) / (self.n + self.mueff + 5.0)
        self.ds = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.n + 1.0)) - 1.0) + self.cs
        self.chiN = float(np.sqrt(self.n) * (1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n ** 2)))
        self.ps = np.zeros(self.n)

        # Diagonal variance  (replaces full covariance)
        self.var = np.ones(self.n)

    # ── public API ─────────────────────────────────────────────────────────────

    def ask(self) -> np.ndarray:
        """Return (lambda, n) array of candidate parameter vectors."""
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * np.sqrt(self.var) * z

    def tell(self, xs: np.ndarray, scores: np.ndarray) -> None:
        """Update distribution given candidates xs and their fitnesses (higher=better)."""
        order = np.argsort(-scores)          # descending
        elite = xs[order[: self.mu]]         # top-μ candidates

        old_mean = self.mean.copy()
        self.mean = (self.w[:, None] * elite).sum(axis=0)

        # Step in normalised space
        step = (self.mean - old_mean) / (self.sigma * np.sqrt(self.var) + 1e-12)
        # CSA path and sigma update
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * step
        self.sigma *= float(np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1.0)))
        self.sigma = float(np.clip(self.sigma, 1e-8, 2.0))

        # Diagonal variance adaptation (cumulative rank-one update)
        ys = (elite - old_mean) / (self.sigma * np.sqrt(self.var) + 1e-12)
        self.var = 0.9 * self.var + 0.1 * float(np.sum(self.w)) * (self.w[:, None] * ys ** 2).sum(axis=0)
        self.var = np.clip(self.var, 1e-10, None)

    @property
    def best(self) -> np.ndarray:
        return self.mean.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(
    theta: np.ndarray,
    hidden_sizes: list[int],
    eval_dir: Path,
    checkpoint_dir: Path,
    config_path: Path,
    planner_config_path: Path,
    eval_seconds: float,
    seed: int,
    entry_name: str = "mlp_cand",
) -> float:
    """Save weights, run run_track_bonus.py, return composite_score."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Write weights
    weights = MLPTrackPlanner.unpack(theta, hidden_sizes)
    weights_path = eval_dir / "planner_weights.npz"
    np.savez(str(weights_path), **weights)

    # Write config pointing at these weights
    cfg = json.loads(planner_config_path.read_text())
    cfg["mlp_weights_path"] = "planner_weights.npz"
    tmp_cfg = eval_dir / "planner_config.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2))

    # Run evaluation (short, no video)
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--planner-config", str(tmp_cfg),
        "--config", str(config_path),
        "--output-dir", str(eval_dir),
        "--entry-name", entry_name,
        "--duration-seconds", str(eval_seconds),
        "--no-render",
        "--seed", str(seed),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=eval_seconds * 10 + 60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"    [warn] eval failed: {exc}")
        return 0.0

    results_file = eval_dir / "results.json"
    if not results_file.exists():
        return 0.0
    try:
        data = json.loads(results_file.read_text())
        return float(data["scores"]["composite_score"])
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"    [warn] could not parse results: {exc}")
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMA-ES training for MLP track planner")
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--config", default="configs/colab_runtime_config.json", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--mlp-hidden", default="32,16", help="Hidden layer sizes, e.g. '32,16'")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--sigma0", type=float, default=0.3)
    p.add_argument("--eval-seconds", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    hidden_sizes = [int(h) for h in args.mlp_hidden.split(",")]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_params = MLPTrackPlanner.param_count(hidden_sizes)
    print(f"MLP architecture: 5→{'→'.join(str(h) for h in hidden_sizes)}→3  ({n_params} parameters)")
    print(f"CMA-ES: {args.iterations} generations × {args.population} candidates")
    print(f"Eval: {args.eval_seconds}s per candidate")
    print(f"Output dir: {output_dir}")

    # ── Initial weights ────────────────────────────────────────────────────────
    init_weights = MLPTrackPlanner.make_weights(hidden_sizes, seed=args.seed)
    theta0 = MLPTrackPlanner.pack(init_weights)

    # ── Create base planner config (mlp type, weights relative path) ───────────
    base_planner_cfg = {
        "planner_type": "mlp",
        "mlp_weights_path": "planner_weights.npz",
        "mlp_hidden": hidden_sizes,
        "stand_seconds": 1.0,
    }
    base_cfg_path = output_dir / "_base_planner_config.json"
    base_cfg_path.write_text(json.dumps(base_planner_cfg, indent=2))

    # ── CMA-ES ─────────────────────────────────────────────────────────────────
    es = _CMAes(theta0, sigma0=args.sigma0, popsize=args.population, seed=args.seed)

    history: list[dict] = []
    best_score = -1.0
    best_theta = theta0.copy()

    tmp_root = output_dir / "_candidates"

    for gen in range(args.iterations):
        t_gen_start = time.time()
        print(f"\n── Generation {gen + 1}/{args.iterations}  (σ={es.sigma:.4f}) ──")

        candidates = es.ask()
        scores = np.zeros(args.population)

        for idx, theta in enumerate(candidates):
            cand_dir = tmp_root / f"g{gen:03d}_c{idx:02d}"
            score = _evaluate(
                theta=theta,
                hidden_sizes=hidden_sizes,
                eval_dir=cand_dir,
                checkpoint_dir=args.checkpoint_dir,
                config_path=args.config,
                planner_config_path=base_cfg_path,
                eval_seconds=args.eval_seconds,
                seed=args.seed + gen * 100 + idx,
                entry_name=f"g{gen}c{idx}",
            )
            scores[idx] = score
            indicator = "★" if score >= best_score else " "
            print(f"  {indicator} [{idx+1:2d}/{args.population}] score={score:.4f}")

            if score > best_score:
                best_score = score
                best_theta = theta.copy()
                # Save current best immediately
                best_weights = MLPTrackPlanner.unpack(best_theta, hidden_sizes)
                np.savez(str(output_dir / "planner_weights.npz"), **best_weights)

        es.tell(candidates, scores)

        gen_time = time.time() - t_gen_start
        history.append({
            "generation": gen,
            "best_score": float(best_score),
            "gen_max_score": float(scores.max()),
            "gen_mean_score": float(scores.mean()),
            "sigma": float(es.sigma),
            "gen_time_s": gen_time,
        })
        print(f"  Best so far: {best_score:.4f}  |  Gen max: {scores.max():.4f}  ({gen_time:.0f}s)")

        # Save history after each generation
        (output_dir / "search_history.json").write_text(json.dumps(history, indent=2))

    # ── Final output ───────────────────────────────────────────────────────────
    # Save best weights
    best_weights = MLPTrackPlanner.unpack(best_theta, hidden_sizes)
    np.savez(str(output_dir / "planner_weights.npz"), **best_weights)

    # Save final planner config
    final_cfg = {
        "planner_type": "mlp",
        "mlp_weights_path": "planner_weights.npz",
        "mlp_hidden": hidden_sizes,
        "stand_seconds": 1.0,
        "training_info": {
            "iterations": args.iterations,
            "population": args.population,
            "eval_seconds": args.eval_seconds,
            "best_composite_score": float(best_score),
        },
    }
    (output_dir / "planner_config.json").write_text(json.dumps(final_cfg, indent=2))

    # Clean up candidate directories
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Training complete.  Best composite score: {best_score:.4f}")
    print(f"Weights saved to:   {output_dir / 'planner_weights.npz'}")
    print(f"Config saved to:    {output_dir / 'planner_config.json'}")
    print(f"{'='*60}")
    print("Next: run full evaluation with:")
    print(f"  python run_track_bonus.py \\")
    print(f"    --checkpoint-dir <your_checkpoint> \\")
    print(f"    --planner-config {output_dir / 'planner_config.json'} \\")
    print(f"    --config <config.json> \\")
    print(f"    --output-dir artifacts/track_eval \\")
    print(f"    --entry-name <your_team> \\")
    print(f"    --render-every 10 --render-fps 5")


if __name__ == "__main__":
    main()
