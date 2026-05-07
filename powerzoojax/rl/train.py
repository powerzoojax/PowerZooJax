"""Unified training entry point for PowerZooJax RL.

Usage::

    from powerzoojax.rl import train, TrainConfig

    # Preset mode — one-liner
    result = train("battery-soc-tracking")
    result = train("case5-economic-dispatch", seed=0, total_timesteps=500_000)

    # Direct env mode
    result = train(my_env, config=TrainConfig(algo="sac"))

    # Custom reward
    result = train(my_env,
                   reward_fn=lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5),
                   config=TrainConfig(algo="ppo"))

    # Inspect results
    print(result.summary)
    result.save("result.json")
"""

import jax

from powerzoojax.rl.config import TrainConfig
from powerzoojax.rl.presets import get_preset
from powerzoojax.rl.reward import RewardWrapper
from powerzoojax.rl.multi_agent import MultiAgentEnvironment
from powerzoojax.rl.trainer import make_train, TrainResult


def train(
    preset_or_env,
    config: TrainConfig = None,
    reward_fn=None,
    seed: int = 42,
    **config_overrides,
) -> TrainResult:
    """Train an RL agent on a PowerZooJax environment.

    Args:
        preset_or_env:  Either a preset name string (e.g. ``"battery-soc-tracking"``)
                        or a wrapped env (``LogWrapper`` / ``SafeRLWrapper``).
        config:         ``TrainConfig`` instance.  When using a preset this defaults
                        to the preset's built-in config.  When passing an env directly
                        it defaults to ``TrainConfig()`` (PPO).
        reward_fn:      Optional reward function with signature
                        ``(obs, action, next_obs, reward, info) -> scalar``
                        or ``(obs, action, next_obs, reward, costs, info) -> scalar``.
                        Wraps the env in a ``RewardWrapper`` when provided.
        seed:           Random seed (overrides ``config.seed``).
        **config_overrides: Additional ``TrainConfig`` field overrides
                        (e.g. ``total_timesteps=500_000, learning_rate=1e-3``).

    Returns:
        ``TrainResult`` with ``params``, ``metrics``, ``config``, ``env_name``.

    Examples::

        # Preset + seed override
        result = train("battery-soc-tracking", seed=0)

        # Preset + config override
        result = train("case5-economic-dispatch",
                       config=TrainConfig(algo="sac", total_timesteps=500_000))

        # Direct env
        result = train(env, config=TrainConfig(algo="ppo_lagrangian",
                                               cost_thresholds=(0.0,)))

        # Custom reward
        result = train("battery-soc-tracking",
                       reward_fn=lambda o, a, no, r, i: -jnp.abs(no[0] - 0.5))
    """
    env_name = ""

    if isinstance(preset_or_env, str):
        preset = get_preset(preset_or_env)
        env = preset.env_factory()
        env_name = preset_or_env
        reward_fn = reward_fn if reward_fn is not None else preset.reward_fn
        config = config if config is not None else preset.config
    else:
        env = preset_or_env
        env_name = getattr(env, "name", "")
        config = config if config is not None else TrainConfig()

    # Apply seed + any extra overrides
    config = config.replace(seed=seed, **config_overrides)

    # Inject custom reward wrapper (single-agent only; MARL envs handle
    # reward shaping internally via voltage_penalty or similar params)
    if reward_fn is not None and not isinstance(env, MultiAgentEnvironment):
        env = RewardWrapper(env, reward_fn)

    train_fn = make_train(env, config)
    key = jax.random.PRNGKey(config.seed)
    result = train_fn(key)
    result.env_name = env_name
    return result
