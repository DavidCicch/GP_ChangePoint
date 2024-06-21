# ==============================================================================
"""Cauchy distribution."""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT

_half_log2pi = 0.5 * math.log(2 * math.pi)


class Cauchy_dist(distribution.Distribution):
  """Cauchy distribution with location `loc` and `scale` parameters."""

#   equiv_tfp_cls = tfd.Normal

  def __init__(self, loc: Numeric, scale: Numeric, max_CP = 1):
    """Initializes a Cauchy distribution.

    Args:
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._scale = conversion.as_float_array(scale)
    self._max_cp = max_CP

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return (self._max_cp, )

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)

#   def _sample_from_std_normal(self, key: PRNGKey, n: int) -> Array:
#     out_shape = (n,) + self.batch_shape
#     dtype = jnp.result_type(self._loc, self._scale)
#     return jax.random.normal(key, shape=out_shape, dtype=dtype)
  
  def nothing(self, x):
    return x
  
  def nan_func(self, x):
    return jnp.nan

  def helper(self, carry, x):
    y = jax.lax.cond(carry['count'] > carry['k'], self.nan_func, self.nothing, x)
    carry['count'] = carry['count'] + 1
    return carry, y
  
  def horizon_check(self, x, vals):
    k = jnp.sum(~jnp.isnan(vals))
    # print(k.shape)
    carry = dict(count = 0,
                 k = k)
    carry, y = jax.lax.scan(self.helper, carry, x)
    return y

  def _sample_n(self, key: PRNGKey, n: int, k = None) -> Array:
    """See `Distribution._sample_n`."""
    uni_noise = jax.random.uniform(key, shape = (n, ))
    # rnd = self._sample_from_std_normal(key, n)
    scale = jnp.expand_dims(self._scale, range(uni_noise.ndim - self._scale.ndim))
    loc = jnp.expand_dims(self._loc, range(uni_noise.ndim - self._loc.ndim))
    cauchy_noise = loc + scale * jnp.tan(jnp.pi * (uni_noise - 1/2))

    # orig_vals = jnp.exp(jnp.array([scale * rnd + loc]).reshape(n, self._max_cp))
    # print(orig_vals.shape)
    if k is None:
      return jnp.squeeze(cauchy_noise)
    new_vals = jax.vmap(lambda a, b: self.horizon_check(a, b), in_axes=(0, 0))(cauchy_noise, k)
    return jnp.squeeze(new_vals)

#   def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
#     """See `Distribution._sample_n_and_log_prob`."""
#     rnd = self._sample_from_std_normal(key, n)
#     samples = self._scale * rnd + self._loc
#     log_prob = -0.5 * jnp.square(rnd) - _half_log2pi - jnp.log(self._scale)
#     return samples, log_prob
  
  def log_prob_calc(self, value):
    prob = jnp.pi * self._scale * (1 + ((value - self._loc)/self.scale)**2)
    return 1/(prob)
  
#   def neg_val(self, value):
#     return -jnp.inf

  def log_prob(self, value: EventT) -> Array:
    """See `Distribution.log_prob`."""
    if value.shape == ():
      prob = self.log_prob_calc(value)
    else:
      prob = jax.vmap(self.log_prob_calc(value))
    
    return prob

  def cdf(self, value: EventT) -> Array:
    """See `Distribution.cdf`."""
    return jax.scipy.special.ndtr(self._standardize(value))

  def log_cdf(self, value: EventT) -> Array:
    """See `Distribution.log_cdf`."""
    return jax.scipy.special.log_ndtr(self._standardize(value))

  def survival_function(self, value: EventT) -> Array:
    """See `Distribution.survival_function`."""
    return jax.scipy.special.ndtr(-self._standardize(value))

  def log_survival_function(self, value: EventT) -> Array:
    """See `Distribution.log_survival_function`."""
    return jax.scipy.special.log_ndtr(-self._standardize(value))

  def _standardize(self, value: EventT) -> Array:
    return (value - self._loc) / self._scale

  def entropy(self) -> Array:
    """Calculates the Shannon entropy (in nats)."""
    log_normalization = _half_log2pi + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy

  def mean(self) -> Array:
    """Calculates the mean."""
    return self.loc

  def variance(self) -> Array:
    """Calculates the variance."""
    return jnp.square(self.scale)

  def stddev(self) -> Array:
    """Calculates the standard deviation."""
    return self.scale

  def mode(self) -> Array:
    """Calculates the mode."""
    return self.mean()

  def median(self) -> Array:
    """Calculates the median."""
    return self.mean()