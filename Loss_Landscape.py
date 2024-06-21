import matplotlib.pyplot as plt
import sys
import os

# sys.path.append('/home/davcic/CP_Testing')

args = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args}'
print(f'Selected GPU {args}')

import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

from New_kernel_1 import Discontinuous_multiple, Discontinuous_multiple_fixed

key = jrnd.PRNGKey(12345)

lengthscale_ = 0.2
output_scale_ = 5.0
obs_noise_ = 0.3
n = 100
x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

x0 = jnp.array([30, 70])/n
base_kernel = jk.RBF()
kernel = Discontinuous_multiple(base_kernel)
K = kernel.cross_covariance(params=dict(lengthscale=lengthscale_,
                                        variance=output_scale_,
                                        CP = x0),
                            x=x, y=x) + 1e-6*jnp.eye(n)

L = jnp.linalg.cholesky(K)
z = jrnd.normal(key, shape=(n,))

f_true = jnp.dot(L, z) + jnp.ones_like(z)
key, obs_key = jrnd.split(key)
y = f_true + obs_noise_*jrnd.normal(obs_key, shape=(n,))

num_CP = 2

size = 20
CP_loc = jnp.linspace(0.05, 1, size)

z = jnp.zeros((size, size))
num_particles = 1_000
num_mcmc_steps = 100
loss_all = jnp.zeros((num_particles, size, size))

for i, cp1 in enumerate(CP_loc):
    for j, cp2 in enumerate(CP_loc):
        base_kernel = jk.RBF()
        kernel = Discontinuous_multiple_fixed(base_kernel, jnp.array([cp1, cp2]))
        priors = dict(kernel=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                               scale=1.),
                                                     tfb.Exp()),
                          variance=dx.Transformed(dx.Normal(loc=0.,
                                                            scale=1.),
                                                  tfb.Exp()),
                                                              ),
              likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                 scale=1.),
                                                       tfb.Exp())),
                cov_log_dens=dict(loss=dx.Uniform(low=jnp.array([-jnp.inf]),
                                        high=jnp.array([0])))
                                                       )
        
        gp_latent = FullLatentGPModel(x, y, cov_fn=kernel, priors=priors)  # Defaults to likelihood=Gaussian()

        key, gpl_key = jrnd.split(key)
        lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpl_key,
                                                                mode='gibbs-in-smc',
                                                                sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))
        
        loss_all = loss_all.at[:, i, j].set(lgp_particles.particles['cov_log_dens']['loss'].flatten())
        z = z.at[i, j].set(jnp.mean(lgp_particles.particles['cov_log_dens']['loss']))

jnp.save("loss_lgp_particles", loss_all)

x = jnp.linspace(0.05, 1, size)
y = jnp.linspace(0.05, 1, size)

_xx, _yy = jnp.meshgrid(x, y)
_x, _y = _xx.ravel(), _yy.ravel()


top = jnp.reshape(z, size*size)
bottom = jnp.zeros_like(top)
width = depth = (3-0.01)/size

ax1 = plt.axes(projection ='3d')
ax1.bar3d(_x, _y, bottom, width, depth, top, shade=True)


plt.savefig("Loss_Landscape.png")
        