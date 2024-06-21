import matplotlib.pyplot as plt
import sys
import os

args = 5
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

# from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

sys.path.append('/home/davcic/CP_Testing')

from New_kernel_1 import Discontinuous_multiple
from New_kernel_1 import Discontinuous_multiple_params
from New_kernel_1 import Discontinuous_multiple_params_hyper
from Poisson_Process_added import Poisson_Process_hyper
# from fullgp import FullMarginalGPModelhyp_mult
from Uniform_modified import Uniform_mod
from Normal_modified import LogNormal_mod
# from fullgp import FullLatentGPModelhyper_mult

sys.path.append('/home/davcic/CP_Testing/Classes')

from GP_CP import GP_CP_Marginal, GP_CP_Latent

key = jrnd.PRNGKey(12345)

lengthscale_ = jnp.array([0.5, 0.1, 0.04, 0.2])
output_scale_ = jnp.array([3, 2, 1, 5])
obs_noise_ = 0.2
n = 100
x = jnp.linspace(0, 1, n)[:, jnp.newaxis]
x0 = jnp.array([0.3, 0.1, 0.7])
# x0 = jnp.concatenate((jnp.array([0]), x0, jnp.array([x.shape[0]])))
#x0 = jnp.append(jnp.zeros(1), x0, jnp.array(x.shape[0]))
base_kernel = jk.RBF()
kernel = Discontinuous_multiple_params(base_kernel, x0)
K = kernel.cross_covariance(params=dict(lengthscale=lengthscale_,
                                        variance=output_scale_,
                                        num=x0),
                            x=x, y=x)+ 1e-6*jnp.eye(n)

L = jnp.linalg.cholesky(K)
z = jrnd.normal(key, shape=(n,))

f_true = jnp.dot(L, z) + jnp.ones_like(z)
key, obs_key = jrnd.split(key)
y = f_true + obs_noise_*jrnd.normal(obs_key, shape=(n,))

ground_truth = dict(f=f_true,
                    lengthscale=lengthscale_,
                    variance=output_scale_,
                    obs_noise=obs_noise_)

size = 20
T = 1
max_CP = size
base_kernel = jk.RBF()
kernel = Discontinuous_multiple_params_hyper(base_kernel)

priors = dict(kernel=dict(lengthscale=LogNormal_mod(0, 3, max_CP+1),
                          variance=LogNormal_mod(0.5, 1, max_CP+1), 
                          num=Poisson_Process_hyper(size, T)
                                                              ),
              likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                 scale=1.),
                                                       tfb.Exp())),
              hyper = dict(hyper_pp = dx.Transformed(dx.Normal(loc=1.,
                                                               scale=1.),
                                                     tfb.Exp())))


num_particles = 1_000
num_mcmc_steps = 1000

# GP_latent = GP_CP_Latent(x, y, cov_fn=kernel, priors=priors, num_particles=num_particles, num_mcmc_steps=num_mcmc_steps)

num_runs = 3

x_pred = jnp.linspace(-0, 1, num=100)
key, key_pred = jrnd.split(key)

num_mcmc_steps_vec = jnp.arange(100, 1100, 100)
# num_mcmc_steps_vec = None

if num_mcmc_steps_vec is not None:        
    f_pred_all = jnp.zeros((num_particles, len(x_pred), num_runs, len(num_mcmc_steps_vec)))
    y_pred_all = jnp.zeros((num_particles, len(x_pred), num_runs, len(num_mcmc_steps_vec)))
else:
    f_pred_all = jnp.zeros((num_particles, len(x_pred), num_runs))
    y_pred_all = jnp.zeros((num_particles, len(x_pred), num_runs))

if num_mcmc_steps_vec is not None:  
    for j, mcmc_steps in enumerate(num_mcmc_steps_vec):
        print(j)
        for i in range(num_runs):
            key, gpm_key = jrnd.split(key)

            GP_latent = GP_CP_Latent(x, y, cov_fn=kernel, priors=priors, num_particles=num_particles, num_mcmc_steps=int(mcmc_steps))
            GP_latent.model_GP(gpm_key)

            jnp.save(f"particles_all_{mcmc_steps}_{i}scale1_1000000_new_convergence_all_nogibbs.npy", GP_latent.particles.particles)
            f_pred_all = f_pred_all.at[:, :, i, j].set(GP_latent.gp_fit.predict_f(key_pred, x_pred))
            y_pred_all = y_pred_all.at[:, :, i, j].set(GP_latent.gp_fit.predict_y(key_pred, x_pred))
else:
    for i in range(num_runs):
        print(i)
        key, gpm_key = jrnd.split(key)
        
        GP_latent = GP_CP_Latent(x, y, cov_fn=kernel, priors=priors, num_particles=num_particles, num_mcmc_steps=num_mcmc_steps)
        GP_latent.model_GP(gpm_key)

        jnp.save(f"particles_all_{num_mcmc_steps}_{i}_scale1_1000000_new_convergence_all_nogibbs.npy", GP_latent.particles.particles)
        f_pred_all = f_pred_all.at[:, :, i].set(GP_latent.gp_fit.predict_f(key_pred, x_pred))
        y_pred_all = y_pred_all.at[:, :, i].set(GP_latent.gp_fit.predict_y(key_pred, x_pred))


        # jnp.save("test.npy", pd_all)
if num_mcmc_steps_vec is not None: 
    jnp.save(f"f_pred_{num_mcmc_steps_vec[0]}_{num_mcmc_steps_vec[1]}_scale1_1000000_new_convergence_all_nogibbs.npy", f_pred_all)
    jnp.save(f"y_pred_{num_mcmc_steps_vec[0]}_{num_mcmc_steps_vec[1]}_scale1_1000000_new_convergence_all_nogibbs.npy", y_pred_all)
else:
    jnp.save(f"f_pred_{num_mcmc_steps}_scale1_1000000_new_convergence_all.npy", f_pred_all)
    jnp.save(f"y_pred_{num_mcmc_steps}_scale1_1000000_new_convergence_all.npy", y_pred_all)