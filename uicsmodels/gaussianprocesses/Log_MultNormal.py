# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Normal distribution."""

import math
from typing import Tuple, Union

import chex
from distrax._src.distributions import distribution
# from distrax._src.distributions import MultivariateNormalFullCovariance

from distrax._src.utils import conversion
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
EventT = distribution.EventT

_half_log2pi = 0.5 * math.log(2 * math.pi)


class Log_MultNormal(distribution.Distribution):
  """Normal distribution with location `loc` and `scale` parameters."""

  equiv_tfp_cls = tfd.Normal

  def __init__(self, loc: Numeric, cov, scale = 1):
    """Initializes a Normal distribution.

    Args:
      loc: Mean of the distribution.
      scale: Standard deviation of the distribution.
    """
    super().__init__()
    self._loc = conversion.as_float_array(loc)
    self._cov = conversion.as_float_array(cov)
    self._scale = conversion.as_float_array(scale)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return ()

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Shape of batch of distribution samples."""
    return jax.lax.broadcast_shapes(self._loc.shape, self._scale.shape)

  @property
  def loc(self) -> Array:
    """Mean of the distribution."""
    return jnp.broadcast_to(self._loc, self.batch_shape)

  @property
  def scale(self) -> Array:
    """Scale of the distribution."""
    return jnp.broadcast_to(self._scale, self.batch_shape)
  

  'TODO'
  def _sample_n(self, key: PRNGKey, k, n: int) -> Array:
    return 0
  
  def multivariate_logprob_k(self, value):
    k = jnp.sum(jnp.count_nonzero(value))
    U, D, V = jnp.linalg.svd(self._cov)
    # print(D)
    D_inv = jnp.where(D == 0, 0, D)
    # print(D_inv)
    inv = V.T@(jnp.diag(D_inv))@U.T
    const = -k/2 * jnp.log(2 * jnp.pi) - 1/2 *jnp.log(2*jnp.sum(jnp.diag(D)))
    # print(const)
    prob = 1/2*(value - self.loc)@(inv)@(value - self.loc)
    # print(prob)
    return const - prob

  def log_prob(self, value: EventT) -> Array:
    py = self.multivariate_logprob_k(value) - jnp.sum(value)
    return py
  
  def multivariate_prob_k(self, k, value):
    const = (2 * jnp.pi)**(-k/2) *(k*self._scale)**(-1/2)
    # print(const)
    prob = jnp.exp(-1/2*(jnp.log(value) - self._loc)@(jnp.eye(k)*self._scale)@(jnp.log(value) - self._loc))
    return const*prob

  def prob(self, k, value: EventT) -> Array:
    py = self.multivariate_prob_k(k, value)/jnp.prod(value)
    return py



# def _kl_divergence_normal_normal(
#     dist1: Union[Normal, tfd.Normal],
#     dist2: Union[Normal, tfd.Normal],
#     *unused_args, **unused_kwargs,
#     ) -> Array:
#   """Obtain the batched KL divergence KL(dist1 || dist2) between two Normals.

#   Args:
#     dist1: A Normal distribution.
#     dist2: A Normal distribution.

#   Returns:
#     Batchwise `KL(dist1 || dist2)`.
#   """
#   diff_log_scale = jnp.log(dist1.scale) - jnp.log(dist2.scale)
#   return (
#       0.5 * jnp.square(dist1.loc / dist2.scale - dist2.loc / dist2.scale) +
#       0.5 * jnp.expm1(2. * diff_log_scale) -
#       diff_log_scale)


# # Register the KL functions with TFP.
# tfd.RegisterKL(Normal, Normal)(_kl_divergence_normal_normal)
# tfd.RegisterKL(Normal, Normal.equiv_tfp_cls)(_kl_divergence_normal_normal)
# tfd.RegisterKL(Normal.equiv_tfp_cls, Normal)(_kl_divergence_normal_normal)