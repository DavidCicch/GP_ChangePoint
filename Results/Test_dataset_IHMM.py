import sys
sys.path.append('/home/davcic/CP_Testing')
sys.path.append('/home/davcic/CP_Testing/HDPHMM')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = f''

import hdphmm
from hdphmm import generate_timeseries as gent
import numpy as np
import matplotlib.pyplot as plt
import generate_timeseries as gent2
import IHMM

sys.path.append('/home/davcic/CP_Testing/Classes')

from GP_CP import GP_CP_Marginal, GP_CP_Latent

import jax.numpy as jnp

name = 'Toy_dataset_RBF_3'

print('Running ' + name)

num_runs = 3

y = jnp.load(name + '/orig_data.npy')

new_arr = np.expand_dims(np.asarray(y), axis=(1, 2))

difference = False  # take first order difference of solute trajectories
observation_model='AR'  # assume an autoregressive model
order = 1  # autoregressive order
max_states = 50
traj_no = 0 # np.arange(10).tolist()# [2]# None # np.arange(24)#2
# first_frame = 7000  # frame after which simulation is equilibrated
dim = [1]
prior = 'MNIW-N'  # MNIW-N or MNIW
link = False  # link trajectories and add phantom state
parameterize_each = True
# hyperparams = {'mu0': np.array([0.5])}
spline_params = {'npts_spline': 10, 'save': True, 'savename': 'spline_hdphmm.pl'}

for i in range(num_runs):
    print(i)
    ihmm = IHMM.InfiniteHMM(new_arr, traj_no=traj_no, load_com=False, difference=difference, observation_model=observation_model, order=order, 
                            max_states=max_states, dim=dim, spline_params=spline_params,
                            prior=prior, link=link, parameterize_each=parameterize_each)


    niter = 1000
    ihmm.inference(niter)
    np.save(name + f'/IHMM/ihmm_trained_{i}', ihmm)