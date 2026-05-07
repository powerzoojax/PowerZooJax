"""TrainConfig — unified training configuration for PowerZooJax RL.

Maps common RL hyperparameters to Rejax and CMDP backends.

Usage::

    from powerzoojax.rl.config import TrainConfig, load_config, save_config

    config = TrainConfig(algo="ppo", total_timesteps=200_000, num_envs=32)
    config2 = config.replace(seed=0, learning_rate=1e-3)

    save_config(config, "experiment.yaml")
    config3 = load_config("experiment.yaml")
"""

import dataclasses
from typing import Sequence, Tuple

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Unified training configuration.

    Fields map to Rejax or self-implemented CMDP backends:

    - Rejax PPO:   ``num_steps`` ← ``n_steps``, ``num_epochs`` ← ``n_epochs``
    - Rejax SAC/TD3/DQN: only common fields are used
    - CMDP:        all fields used, ``cost_thresholds`` defines the vector budget
    """

    # ---- Common ----
    algo: str = "ppo"
    """Algorithm: "ppo" | "sac" | "td3" | "dqn" | "ppo_lagrangian" | "saute_ppo" | "ippo" | "ippo_typed" | "ippo_typed_lagrangian" | "mappo".

    ``"ippo_typed"`` — IPPO with type-specific parameter sharing: agents are
    partitioned by type prefix (e.g. ``"battery_*"``, ``"renewable_*"``,
    ``"flexload_*"``); each type gets an independent ``SharedActorCritic``.
    Used by the ``ders-medium`` and ``ders-medium-safe`` presets for heterogeneous DERs.

    ``"ippo_typed_lagrangian"`` — typed IPPO-Lagrangian for heterogeneous
    cooperative MARL: the per-type actors remain decentralized, while explicit
    CMDP constraint costs are enforced through local cost critics and a shared
    team-level dual variable.
    """
    total_timesteps: int = 100_000
    num_envs: int = 64
    seed: int = 42

    # ---- Optimizer ----
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    eval_episodes: int = 128
    """Episodes per in-training evaluation callback.

    This controls lightweight monitoring during training only. It does not
    override the benchmark's post-training evaluation episode budget.
    """

    # ---- PPO / CMDP ----
    n_steps: int = 128          # rollout length per update
    n_epochs: int = 4
    n_minibatches: int = 4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gae_lambda: float = 0.95

    # ---- CMDP (PPO-Lagrangian) ----
    cost_threshold: float = 0.0
    """Scalar fallback used when ``cost_thresholds`` is empty and n_constraints == 1."""
    cost_thresholds: Tuple[float, ...] = ()
    """Vector CMDP budget d_i in the same physical units as the raw env cost.
    Empty tuple falls back to ``cost_threshold`` broadcasting.
    The trainer divides both rollout costs and these thresholds by ``cost_scale``
    before any dual update, so values must be given in raw (unscaled) units."""
    lambda_lr: float | Tuple[float, ...] = 5e-3
    cost_gamma: float = 1.0
    cost_scale: float | Tuple[float, ...] = 1.0
    """Per-constraint cost scaling applied **inside the trainer only**.
    Divides raw env costs (and thresholds) before GAE and the dual update to keep
    per-step values numerically O(1).  Does not change the CMDP problem definition."""
    log_lambda_max: float = 5.0
    n_checkpoints: int = 1
    n_eval_envs: int = 0
    """Number of environments to use for greedy eval inside each CMDP update step.

    0  = skip eval (default, backward-compatible).
    >0 = run a greedy rollout of ``n_steps`` steps at every update, logs
         ``eval_returns`` (mean episodic return) and ``eval_cost_<name>``
         (mean per-step cost) — comparable to rejax PPO's ``eval_returns``.
         Adds ~(n_eval_envs / num_envs) overhead per update.
    """
    """Number of params snapshots to capture during training (incl. final).

    1 = current behaviour: only the final params are returned.
    K > 1 = intermediate params are returned in ``TrainResult.checkpoints`` for
    checkpoint selection / plotting metrics vs environment steps.
    Supported by the CMDP trainer and the Rejax single-agent trainers.
    """

    # ---- SautéRL (algo="saute_ppo") ----
    saute_horizon: int | None = None
    """Episode horizon for budget normalisation. ``None`` → reads ``params.max_steps``."""
    saute_unsafe_reward: float = 0.0
    """Reward substituted when any constraint budget is exhausted (z_i ≤ 0)."""
    saute_use_reward_shaping: bool = True
    """If False, only augment obs without shaping reward (ablation)."""

    # ---- Network (Rejax PPO: passed as agent_kwargs["hidden_layer_sizes"]) ----
    hidden_dims: Tuple[int, ...] = (64, 64)
    continuous_action_dist: str = "gaussian"
    """Continuous-action actor family for Rejax PPO-style trainers.

    ``"gaussian"`` keeps Rejax's default unbounded Gaussian actor.
    ``"beta"`` uses a bounded Beta policy whose support matches the Box action
    space directly.
    """

    # ---- Rejax PPO observation/reward scaling (helps high-dimensional obs) ----
    normalize_observations: bool = False
    normalize_rewards: bool = False

    # ---- Rejax SAC (optional; PPO also uses n_epochs, which SAC ignores) ----
    sac_num_epochs: int | None = None
    """If set and ``algo == \"sac\"``, passed to Rejax ``SAC`` as ``num_epochs``
    (gradient steps per environment transition batch).  If ``None``, Rejax's
    default applies (commonly 1) — we do not mirror PPO's ``n_epochs`` for SAC
    to avoid changing legacy behaviour.
    """
    sac_target_entropy_ratio: float | None = None
    """Optional override for Rejax SAC's ``target_entropy_ratio``."""

    # ---- Metrics (Rejax backend) ----
    eval_num_episodes: int = 128
    """Number of parallel eval rollouts passed to ``rejax.evaluate`` (``num_seeds``).

    Lower values speed up each eval checkpoint at the cost of noisier mean return.
    """

    record_eval_wall_time: bool = False
    """If True, Rejax backend records host wall time (s) at each eval via ``io_callback``.

    Single fused ``algo.train()`` is unchanged; overhead is one small host callback per
    eval (typically tens per run). Use for plots vs wall time instead of splitting
    ``train()`` into many short runs (which re-triggers compile / breaks scan fusion).

    Note: Rejax runs one **initial** eval before any training, then evals every
    ``eval_freq``; the first interval is therefore not comparable to later ones.
    """

    wall_time_warmup: bool = True
    """When ``record_eval_wall_time`` is True: run a short training pass first so XLA
    compiles, then start the wall clock at ``t0`` for the main run (excludes most JIT
    cost from ``eval_wall_time_s``). Set False for tests or if you want absolute time
    from process start."""

    wall_time_warmup_timesteps: int | None = None
    """Warm-up horizon (env steps). ``None`` → ``eval_freq`` (one eval period, minimal
    scan length). Clamped to ``[1, total_timesteps)``. Ignored if ``wall_time_warmup``
    is False or warm-up would equal the full run."""

    # ---- Convenience ----

    def replace(self, **kwargs) -> "TrainConfig":
        """Return a new TrainConfig with fields replaced by kwargs."""
        return dataclasses.replace(self, **kwargs)

    def _asdict(self) -> dict:
        """Return a plain dict (for JSON/YAML serialization)."""
        d = dataclasses.asdict(self)
        # tuple → list for YAML compatibility
        d["hidden_dims"] = list(d["hidden_dims"])
        d["cost_thresholds"] = list(d["cost_thresholds"])
        if isinstance(d["lambda_lr"], tuple):
            d["lambda_lr"] = list(d["lambda_lr"])
        if isinstance(d["cost_scale"], tuple):
            d["cost_scale"] = list(d["cost_scale"])
        return d

    def resolved_cost_thresholds(self, n_constraints: int) -> Tuple[float, ...]:
        """Return vector cost thresholds, broadcasting legacy scalar config."""
        if n_constraints < 0:
            raise ValueError("n_constraints must be non-negative")
        if n_constraints == 0:
            return ()
        if self.cost_thresholds:
            values = tuple(float(x) for x in self.cost_thresholds)
        else:
            values = tuple(float(self.cost_threshold) for _ in range(n_constraints))
        if len(values) != n_constraints:
            raise ValueError(
                f"Expected {n_constraints} cost thresholds, got {len(values)}."
            )
        return values

    def resolved_lambda_lr(self, n_constraints: int) -> Tuple[float, ...]:
        if n_constraints == 0:
            return ()
        value = self.lambda_lr
        if isinstance(value, (tuple, list)):
            values = tuple(float(x) for x in value)
        else:
            values = tuple(float(value) for _ in range(n_constraints))
        if len(values) != n_constraints:
            raise ValueError(
                f"Expected {n_constraints} lambda_lr values, got {len(values)}."
            )
        return values

    def resolved_cost_scale(self, n_constraints: int) -> Tuple[float, ...]:
        if n_constraints == 0:
            return ()
        value = self.cost_scale
        if isinstance(value, (tuple, list)):
            values = tuple(float(x) for x in value)
        else:
            values = tuple(float(value) for _ in range(n_constraints))
        if len(values) != n_constraints:
            raise ValueError(
                f"Expected {n_constraints} cost_scale values, got {len(values)}."
            )
        return values

    def to_rejax_kwargs(self) -> dict:
        """Map to Rejax create() kwargs (algo-dependent subset)."""
        common = dict(
            total_timesteps=self.total_timesteps,
            num_envs=self.num_envs,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            max_grad_norm=self.max_grad_norm,
            eval_freq=self.eval_freq,
        )
        if self.algo in ("ppo", "saute_ppo"):
            common.update(
                num_steps=self.n_steps,
                num_epochs=self.n_epochs,
                num_minibatches=self.n_minibatches,
                clip_eps=self.clip_eps,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                gae_lambda=self.gae_lambda,
                agent_kwargs={"hidden_layer_sizes": tuple(self.hidden_dims)},
                normalize_observations=self.normalize_observations,
                normalize_rewards=self.normalize_rewards,
            )
        elif self.algo == "sac":
            sac_kw = {
                "hidden_layer_sizes": tuple(self.hidden_dims),
                "normalize_observations": self.normalize_observations,
                "normalize_rewards": self.normalize_rewards,
            }
            if self.sac_num_epochs is not None:
                sac_kw["num_epochs"] = int(self.sac_num_epochs)
            if self.sac_target_entropy_ratio is not None:
                sac_kw["target_entropy_ratio"] = float(self.sac_target_entropy_ratio)
            common.update(sac_kw)
        return common


def load_config(source, config_cls=TrainConfig) -> TrainConfig:
    """Load a TrainConfig from a YAML file path or a plain dict.

    Args:
        source: str path to a YAML file, or a dict of field values.
        config_cls: Config class to instantiate (default: TrainConfig).

    Returns:
        A ``TrainConfig`` instance.
    """
    if isinstance(source, dict):
        d = source
    else:
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML config loading. "
                "Install with: pip install powerzoojax[rl]"
            )
        with open(source, "r") as f:
            d = _yaml.safe_load(f)

    # Coerce list-like fields to tuples
    if "hidden_dims" in d and isinstance(d["hidden_dims"], list):
        d["hidden_dims"] = tuple(d["hidden_dims"])
    if "cost_thresholds" in d and isinstance(d["cost_thresholds"], list):
        d["cost_thresholds"] = tuple(d["cost_thresholds"])
    if "lambda_lr" in d and isinstance(d["lambda_lr"], list):
        d["lambda_lr"] = tuple(d["lambda_lr"])
    if "cost_scale" in d and isinstance(d["cost_scale"], list):
        d["cost_scale"] = tuple(d["cost_scale"])

    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return config_cls(**filtered)


def save_config(config: TrainConfig, path: str) -> None:
    """Save a TrainConfig to a YAML file.

    Args:
        config: TrainConfig instance to save.
        path:   Destination file path (.yaml).
    """
    if not _HAS_YAML:
        raise ImportError(
            "PyYAML is required for YAML config saving. "
            "Install with: pip install powerzoojax[rl]"
        )
    with open(path, "w") as f:
        _yaml.dump(config._asdict(), f, default_flow_style=False, sort_keys=False)
