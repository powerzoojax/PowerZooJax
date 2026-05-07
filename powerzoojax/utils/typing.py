"""
Type Definitions and Hints

Provides consistent type annotations across PowerZooJAX.

Uses:
- chex.Array for JAX arrays
- TypeVar for generic typing
- Protocol for structural typing

This improves code readability and enables better IDE support.
"""

from typing import TypeVar, Union, Tuple, Dict, Any, Callable, Protocol
import chex
import jax.numpy as jnp

# Basic JAX types
PRNGKey = chex.PRNGKey
Array = chex.Array
Scalar = Union[float, chex.Array]

# Shape types
Shape = Tuple[int, ...]
DType = jnp.dtype

# Generic state/params types
StateT = TypeVar('StateT')
ParamsT = TypeVar('ParamsT')
ObsT = TypeVar('ObsT')
ActionT = TypeVar('ActionT')

# Environment return types
StepReturn = Tuple[Array, StateT, Scalar, bool, Dict[str, Any]]
ResetReturn = Tuple[Array, StateT]


class EnvironmentProtocol(Protocol[StateT, ParamsT]):
    """Protocol defining the environment interface.
    
    Environments implementing this protocol are guaranteed to have
    the standard reset/step methods with correct signatures.
    """
    
    def reset(self, key: PRNGKey, params: ParamsT) -> Tuple[Array, StateT]:
        ...
    
    def step(
        self, 
        key: PRNGKey, 
        state: StateT, 
        action: Array, 
        params: ParamsT
    ) -> StepReturn:
        ...


class PolicyProtocol(Protocol):
    """Protocol for policy functions.
    
    Policies take observations and return actions.
    """
    
    def __call__(self, obs: Array, key: PRNGKey) -> Array:
        ...


# Reward function type
RewardFn = Callable[[StateT, Array, ParamsT], Scalar]

# Observation function type  
ObsFn = Callable[[StateT, ParamsT], Array]
