# Spaces

PowerZooJax uses lightweight JAX-friendly space descriptors for observations and actions.

## Space types

::: powerzoojax.envs.spaces.Space

::: powerzoojax.envs.spaces.Box

::: powerzoojax.envs.spaces.Discrete

::: powerzoojax.envs.spaces.MultiDiscrete

::: powerzoojax.envs.spaces.MultiBinary

## Helper constructors

::: powerzoojax.envs.spaces.make_box

::: powerzoojax.envs.spaces.make_discrete

::: powerzoojax.envs.spaces.make_multi_discrete

::: powerzoojax.envs.spaces.make_multi_binary

## Environment contract

All environments implement the same high-level interface:

::: powerzoojax.envs.base.Environment
    options:
      show_source: false
      members:
        - reset
        - step
        - step_auto_reset
        - observation_space
        - action_space

::: powerzoojax.envs.base.EnvState

::: powerzoojax.envs.base.EnvParams
