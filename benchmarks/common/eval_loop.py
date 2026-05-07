"""Shared evaluation loop for all benchmark tasks.

Provides :func:`run_episodes` — the single entry-point used by every
``benchmarks/<task>/eval.py`` and ``benchmarks/<task>/baselines.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax


def run_episodes(
    task,
    split: str,
    agent_fn: Callable,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Run *n_episodes* and return per-episode metric dicts.

    Parameters
    ----------
    task:
        Any object satisfying the :class:`~powerzoojax.tasks.base.TaskSpec`
        Protocol: ``make_env``, ``episode_params``, ``rollout``,
        ``baseline_rollout``, ``compute_metrics``.
    split:
        Evaluation split name passed to ``task.episode_params``.
    agent_fn:
        Callable ``(params, key) -> rollout_data``.  Typically a JIT-compiled
        closure wrapping ``task.rollout(env, params, key, policy_fn)``.
        For tasks with Python-loop rollouts (e.g. DC Microgrid), leave unJIT'd.
    n_episodes:
        Number of episodes.
    max_steps:
        Episode length in environment steps.
    seed:
        Base seed.  Episode keys are derived as
        ``jax.random.PRNGKey(seed * 10_000 + ep_idx)``.

    Returns
    -------
    list[dict[str, float]]
        One metrics dict per episode.
    """
    env = task.make_env(split)

    metrics_list = []
    for ep in range(n_episodes):
        key = jax.random.PRNGKey(seed * 10_000 + ep)
        ref_key = jax.random.PRNGKey(seed * 10_000 + ep + 50_000)
        params = task.episode_params(
            split, ep, n_episodes, max_steps,
            strategy="uniform", seed=seed,
        )
        agent_data = agent_fn(params, key)
        ref_data = task.baseline_rollout(env, params, ref_key, "no_control")
        metrics_list.append(task.compute_metrics(agent_data, ref_data))

    return metrics_list
