import sys
import matplotlib.pyplot as plt
import os

sys.path.append('/home/davcic/CP_Testing')
args = 6
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args}'
print(f'Selected GPU {args}')

from load_dataset import TimeSeries
import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk
# import pandas as pd

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

from New_kernel_1 import Discontinuous_multiple
from New_kernel_1 import Discontinuous_multiple_params
from New_kernel_1 import Discontinuous_multiple_params_hyper
from New_kernel_1 import Discontinuous_multiple_params_hyper_periodic
from Poisson_Process_added import Poisson_Process_hyper
from fullgp import FullMarginalGPModelhyper_mult
from Uniform_modified import Uniform_mod
from Normal_modified import LogNormal_mod

readname = 'lga_passengers'
savename = "lga_passengers"
path = f'./{savename}'

data = TimeSeries.from_json(f'datasets/{readname}/{readname}.json')
# fig = plt.figure(figsize = (10, 6))
# y_data = data.y[15:]
# print(len(data.y))
# x_data = data.t[15:]-15
corr_data = data.y[99:]/jnp.max(data.y[99:])

key = jrnd.PRNGKey(123456)

size = 10
T = 1
max_CP = size
base_kernel1 = jk.RBF()
base_kernel2 = jk.Periodic()

# kernel = Discontinuous_multiple_params_hyper(base_kernel)
# kernel = Discontinuous_multiple_params_hyper_periodic(base_kernel)
# kernel = Discontinuous_multiple_params_hyper(base_kernel) + Discontinuous_multiple_params_hyper_periodic()

kernel = Discontinuous_multiple_params_hyper(base_kernel1) + Discontinuous_multiple_params_hyper_periodic(base_kernel2)

priors = dict(kernel=  [dict(lengthscale=LogNormal_mod(0, 2, max_CP+1),
                             variance=LogNormal_mod(0, 2, max_CP+1),
                             num=Poisson_Process_hyper(size, T)),
                        dict(lengthscale=LogNormal_mod(0, 2, max_CP+1),
                             variance=LogNormal_mod(0, 2, max_CP+1), 
                             period=LogNormal_mod(0, 3, max_CP+1),
                             num=Poisson_Process_hyper(size, T))],
              likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                 scale=1.),
                                                       tfb.Exp())),
              hyper = [dict(hyper_1 = dx.Transformed(dx.Normal(loc=0.,
                                                               scale=1.),
                                                     tfb.Exp())), 
                       dict(hyper_2 = dx.Transformed(dx.Normal(loc=0.,
                                                               scale=1.),
                                                     tfb.Exp()))])

y = corr_data.flatten()
x = jnp.array(data.t[:-99]/(len(data.t[:-99])-1))[:, jnp.newaxis]

gp_marginal = FullMarginalGPModelhyper_mult(x, y, cov_fn=kernel, priors=priors)  # Implies likelihood=Gaussian()

num_particles = 1000
num_mcmc_steps = 100

key, gpm_key = jrnd.split(key)
mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
                                                                  mode='gibbs-in-smc',
                                                                  sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps), 
                                                                  poisson = True)

x_pred = jnp.linspace(-0, 1, num=len(y))
key, key_pred = jrnd.split(key)
f_pred = gp_marginal.predict_f(key_pred, x_pred)
y_pred = gp_marginal.predict_y(key_pred, x_pred)

jnp.save(f"particles_{num_mcmc_steps}_{readname}_combined.npy", mgp_particles.particles)
jnp.save(f"f_pred_{num_mcmc_steps}_{readname}_combined.npy", f_pred)
jnp.save(f"y_pred_{num_mcmc_steps}_{readname}_combined.npy", y_pred)