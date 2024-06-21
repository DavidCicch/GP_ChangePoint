"""Poisson Process"""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions import uniform 

from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT


class Poisson_Process_hyper(distribution.Distribution):
  """POisson Process via the exponential distirbution with lambda paramter"""

  equiv_tfp_cls = tfd.Normal

  def __init__(self, process_length: Numeric, horizon: Numeric):
    """Initializes a Poisson distribution.

    Args:
      loc: Mean of the distribution (scale parameter).
      process_length: How long to run the process
      horizon: maximum value our samples can have
    """
    super().__init__()
    self.process_length = process_length
    self.horizon = horizon
    # self._scale = conversion.as_float_array(scale)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
     """Shape of batch of distribution samples."""
    #  print(self.process_length)
     return tuple([self.process_length])

  # @property
  # def loc(self) -> Array:
  #   """Mean of the distribution."""
  #   return jnp.broadcast_to(self._loc, self.batch_shape)

#   @property
#   def scale(self) -> Array:
#     """Scale of the distribution."""
#     return jnp.broadcast_to(self._scale, self.batch_shape)

  def helper(self, carry, xs):
    ys = xs-carry
    carry = xs
    return carry, ys

  def get_orig_value(self, value: EventT) -> Array:
    carry = 0
    carry, ys = jax.lax.scan(self.helper, carry, value)
    return ys
  
  def r(self, value: EventT) -> Numeric:                                                                                                   
    return value*jnp.log(value) - value + (jnp.log(value*(1+4*value*(1+2*value))))/6 + jnp.log(jnp.pi)/2
  
  def zero_func(self, value = 1) -> Numeric: 
    return 0.
  
  def neginf_func(self, value = 1) -> Numeric: 
    return -jnp.inf
  
  def helper(self, n):
    return jax.lax.cond(n==0, self.zero_func, self.r, n)
  
  def log_prob(self, value: EventT, alpha) -> Array:
    """See `Distribution.log_prob`."""
    n = jnp.sum(~jnp.isnan(value))
    mean = self.horizon/alpha
    # log_nfac = jax.vmap(self.helper)(n)
    log_nfac = jax.lax.cond(n==0, self.zero_func, self.r, n)
    # orig_vals = self.get_orig_value(value)
    bound_check = jnp.where(
        jnp.logical_or(value < 0, value > self.horizon),
        -jnp.inf,
        0) 
    return n*jnp.log(mean) - mean - log_nfac + jnp.sum(bound_check)
  
  def horizon_check(self, x):
    return jnp.where(x>self.horizon, jnp.nan, x)

  def _sample_n(self, key: PRNGKey, n: int, alpha) -> Array:
    """See `Distribution._sample_n`."""
    # out_shape = (n,) + self.batch_shape
    # dtype = jnp.result_type(loc)
    samples = jnp.cumsum(alpha*jax.random.exponential(key, [self.process_length, n]), axis = 0)
    new_samples = jax.vmap(jax.vmap(lambda a: self.horizon_check(a)))(samples)
    return new_samples.T