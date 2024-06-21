import matplotlib.pyplot as plt
import sys
import os

args = 4
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args}'
print(f'Selected GPU {args}')

# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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


sys.path.append('/home/davcic/CP_Testing')

from New_kernel_1 import Discontinuous_multiple
from New_kernel_1 import Discontinuous_multiple_params
from New_kernel_1 import Discontinuous_multiple_params_hyper
from Poisson_Process_added import Poisson_Process_hyper
from Uniform_modified import Uniform_mod
from Normal_modified import LogNormal_mod
import gc


sys.path.append('/home/davcic/CP_Testing/Classes')

from GP_CP import GP_CP_Marginal, GP_CP_Latent

key = jrnd.PRNGKey(12345)

name = 'Toy_dataset_IHMM'

print(f'Running ' + name)

num_runs = 3

num_mcmc_steps_vec = None

'''Marginal GP'''
num_mcmc_steps = 100
if num_mcmc_steps_vec is not None:  
    for j, mcmc_steps in enumerate(num_mcmc_steps_vec):
        print(j)
        for i in range(num_runs):
            GP_Marginal = jnp.load(name + '/GP_Marginal/GP_marginal_untrained_orig.npy', allow_pickle = True)[()]
            GP_Marginal.num_mcmc_steps = num_mcmc_steps
            key, gpm_key = jrnd.split(key)

            GP_Marginal.model_GP(gpm_key)

            jnp.save(name + f'/GP_marginal_trained_{mcmc_steps}_{i}', GP_Marginal)

            gc.collect()

else:
    for i in range(num_runs):
        print(i)
        GP_Marginal = jnp.load(name + '/GP_Marginal/GP_marginal_untrained_orig.npy', allow_pickle = True)[()]
        GP_Marginal.num_mcmc_steps = num_mcmc_steps

        key, gpm_key = jrnd.split(key)     
        
        GP_Marginal.model_GP(gpm_key)

        jnp.save(name + f'/GP_Marginal/GP_marginal_trained_orig_{i}', GP_Marginal)
        gc.collect()


'''Latent GP'''
num_mcmc_steps = 1000
if num_mcmc_steps_vec is not None:  
    for j, mcmc_steps in enumerate(num_mcmc_steps_vec):
        print(j)
        for i in range(num_runs):
            GP_Latent = jnp.load(name + '/GP_Latent/GP_latent_untrained_orig.npy', allow_pickle = True)[()]
            GP_Latent.num_mcmc_steps = num_mcmc_steps
            key, gpm_key = jrnd.split(key)

            GP_Latent.model_GP(gpm_key)

            jnp.save(name + f'/GP_Latent/GP_latent_trained_{mcmc_steps}_{i}', GP_Latent)

            gc.collect()

else:
    for i in range(num_runs):
        print(i)
        GP_Latent = jnp.load(name + '/GP_Latent/GP_latent_untrained_orig.npy', allow_pickle = True)[()]
        GP_Latent.num_mcmc_steps = num_mcmc_steps

        key, gpm_key = jrnd.split(key)     
        
        GP_Latent.model_GP(gpm_key)

        jnp.save(name + f'/GP_Latent/GP_latent_trained_orig_{i}', GP_Latent)

        gc.collect()
