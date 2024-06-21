import os
import argparse
import datetime


# if __name__ == '__main__':
#     print(f'\ncurrent system time: {datetime.datetime.now()}')

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-gpu', dest='GPU', type=int)
#     args =  parser.parse_args()
    
#     if args.GPU:
#         print(f'Selected GPU {args.GPU}')
#         os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.GPU}'
#     else:
#         print('Selected CPU, hiding all cuda devices')
#         os.environ['CUDA_VISIBLE_DEVICES'] = f''  # hide all GPUs
#         os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # perhaps not needed when no GPUs are available
#     print()

    # loead and run experiment
    #import experiment
    #experiment.main(args)

# pip install numpy==1.23.5
import matplotlib.pyplot as plt
import os
args = 2
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

#from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

import sys
print('Jax version:        ', jax.__version__)
print('Python version:     ', sys.version)
print('Jax default backend:', jax.default_backend())
print('Jax devices:        ', jax.devices())

from New_kernel_1 import Discontinuous_multiple
from New_kernel_1 import Discontinuous_multiple_params
from Poisson_Process_added import Poisson_Process_hyper
from fullgp import FullMarginalGPModelhyper_mult
from Uniform_modified import Uniform_mod

key = jrnd.PRNGKey(12345)

lengthscale_ = jnp.array([0.5, 0.1, 0.01, 0.2])
output_scale_ = jnp.array([3, 2, 1, 5])
obs_noise_ = 0.3
n = 100
x = jnp.linspace(0, 1, n)[:, jnp.newaxis]
x0 = jnp.array([30, 10, 70])/n
# x0 = jnp.concatenate((jnp.array([0]), x0, jnp.array([x.shape[0]])))
#x0 = jnp.append(jnp.zeros(1), x0, jnp.array(x.shape[0]))
base_kernel = jk.RBF()
kernel = Discontinuous_multiple_params(base_kernel, x0)
K = kernel.cross_covariance(params=dict(lengthscale=lengthscale_,
                                        variance=output_scale_,
                                        CP=x0),
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

plt.figure(figsize=(12, 4))
plt.plot(x, f_true, 'k', label=r'')
plt.plot(x, y, 'rx', label='obs')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0., 1.])
plt.legend()
plt.savefig("test_data2.png")
