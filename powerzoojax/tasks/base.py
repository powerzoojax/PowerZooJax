"""TaskSpec Protocol — common interface for all benchmark tasks.

Every task class in ``powerzoojax/tasks/`` satisfies this Protocol via
structural subtyping (no inheritance required).  Benchmarks interact with
tasks only through this interface, keeping benchmark code free of
task-specific logic.

Usage::

    from powerzoojax.tasks.base import TaskSpec
    from powerzoojax.tasks.dso import DSOTask

    task: TaskSpec = DSOTask(v_min=0.94, v_max=1.06)
    env = task.make_env(split="train")
    params = task.episode_params("train", episode_idx=0, n_episodes=1, max_steps=48,
                                  strategy="seeded", seed=42)
    rollout = task.rollout(env, params, key, policy_fn)
    baseline = task.baseline_rollout(env, params, key, "no_control")
    metrics = task.compute_metrics(rollout, baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class ConstraintSpec:
    """Frozen task-level CMDP constraint selection and fallback recipe."""

    selected_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    fallback_weights: tuple[float, ...]


@runtime_checkable
class TaskSpec(Protocol):
    """Structural interface for all five benchmark tasks.

    Attributes
    ----------
    task_name : str
        Short identifier used in ``RunRecord.task`` (``"dso"``, ``"tso"``,
        ``"ders"``, ``"gencos"``, ``"dc_microgrid"``).
    default_splits : tuple[str, ...]
        Ordered evaluation splits this task supports, e.g.
        ``("train", "iid", "summer_ood", "zone_holdout")``.
    """

    task_name: str
    default_splits: tuple[str, ...]

    def make_env(self, split: str = "train") -> Any:
        """Return the environment for the given split.

        For single-agent tasks (DSO, TSO, DC Microgrid) the same stateless
        environment object is returned regardless of *split*.  For MARL
        tasks (DERs, GenCos) a split-specific env with embedded params is
        returned and cached internally for reuse across episodes.
        """
        ...

    def episode_params(
        self,
        split: str,
        episode_idx: int,
        n_episodes: int,
        max_steps: int,
        *,
        strategy: Literal["uniform", "seeded"] = "uniform",
        seed: int = 0,
    ) -> Any:
        """Return environment params for one episode.

        ``strategy="uniform"`` (eval): episode windows drawn via linspace
        across the full split to ensure coverage of all seasons.
        ``strategy="seeded"`` (training): one window sampled from *seed* so
        each seed trains on a different season.

        The task caches expensive per-split work (Ybus construction, data
        loading) internally.  Only cheap operations (load profile slicing,
        ``params.replace``) repeat per episode call.

        For MARL tasks where params do not vary per episode, the same object
        is returned on every call.  For single-agent tasks with temporal
        diversity (DSO, TSO, DC Microgrid), the returned params differ per call.
        """
        ...

    def rollout(self, env: Any, params: Any, key: Any, policy_fn: Any) -> Any:
        """Run one episode with *policy_fn* and return raw rollout data.

        For single-agent tasks *policy_fn* is a ``(obs, state, key) -> action``
        callable.  For MARL tasks it is a task-specific callable (e.g. an
        ``obs_dict -> actions_dict`` function for DERs).
        """
        ...

    def baseline_rollout(
        self, env: Any, params: Any, key: Any, baseline_name: str
    ) -> Any:
        """Run one episode with a named non-learning baseline (rule-based, no-control, …).

        Valid names are task-specific; use ``baseline_names()`` to list them.
        The ``"no_control"`` baseline is always available and serves as the
        reference in ``compute_metrics``.
        """
        ...

    def compute_metrics(
        self, agent_rollout: Any, baseline_rollout: Any
    ) -> dict[str, float]:
        """Compute episode-level scalar metrics.

        Compares *agent_rollout* against *baseline_rollout* (usually the
        ``"no_control"`` baseline).  Returns a flat ``dict[str, float]``
        suitable for storage in ``RunRecord.metrics``.
        """
        ...

    def baseline_names(self) -> tuple[str, ...]:
        """Names of available non-learning baselines.

        ``"no_control"`` is always the first entry and is used as the
        reference in ``compute_metrics``.
        """
        ...

    def constraint_spec(self) -> ConstraintSpec:
        """Return the task's frozen CMDP constraint recipe."""
        ...
