# Copyright 2023- The Uncertainty in Complex Systems contributors.
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

from uicsmodels.sampling.inference import inference_loop, smc_inference_loop, smc_inference_loop_trace
from uicsmodels.sampling.inference import update_metropolis

from abc import ABC, abstractmethod


import jax
import jax.numpy as jnp
from jax import Array
import jax.random as jrnd
from jax.random import PRNGKeyArray as PRNGKey
from typing import Any, Union, NamedTuple, Dict, Any, Iterable, Mapping, Callable
from jaxtyping import Float
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from blackjax import adaptive_tempered_smc, rmh
from blackjax.types import Array, PRNGKey, PyTree
import blackjax.smc.resampling as resampling

from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
from jax.flatten_util import ravel_pytree

__all__ = ['GibbsState', 'BayesianModel']

class GibbsState(NamedTuple):

    position: ArrayTree


#
class RMHState(NamedTuple):
    """State of the RMH chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: PyTree
    log_density: float    

class RMHInfo(NamedTuple):
    """Additional information on the RMH chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RMHState

class BayesianModel(ABC):
    
    def init_fn(self, key: Array, num_particles: int = 1):
        """Initial state for MCMC/SMC.

        This function initializes all highest level latent variables. Children
        of this class need to implement initialization of intermediate latent
        variables according to the structure of the hierarchical model.

        Args:
            key: PRNGKey
            num_particles: int
                Number of particles to initialize a state for
        Returns:
            GibbsState

        """

        priors_flat, priors_treedef = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
        samples = list()
        for prior in priors_flat:
            key, subkey = jrnd.split(key)
            samples.append(prior.sample(seed=subkey, sample_shape=(num_particles,)))

        initial_position = jax.tree_util.tree_unflatten(priors_treedef, samples)
        if num_particles == 1:
            initial_position = tree_map(lambda x: jnp.squeeze(x), initial_position)        
        return GibbsState(position=initial_position)

    #  
    def sample_from_prior(self, key, num_samples=1):
        return self.init_fn(key, num_particles=num_samples)

    #
    def smc_init_fn(self, position: ArrayTree, kwargs):
        """Simply wrap the position dictionary in a GibbsState object. 

        Args:
            position: dict
                Current assignment of the state values
            kwargs: not used in our Gibbs kernel
        Returns:
            A Gibbs state object.
        """
        if isinstance(position, GibbsState):
            return position
        return GibbsState(position)

    #
    def loglikelihood_fn(self):
        pass

    #
    def logprior_fn(self) -> Callable:
        """Returns the log-prior function for the model given a state.

        This default logprior assumes a non-hierarchical model. If a 
        hierarchical model is used, the mode should implement its own 
        logprior_fn.

        Args:
            None
        Returns:
            A function that computes the log-prior of the model given a state.

        """

        def logprior_fn_(state: GibbsState):
            position = getattr(state, 'position', state)
            logprob = 0
            priors_flat, _ = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
            values_flat, _ = tree_flatten(position)
            for value, dist in zip(values_flat, priors_flat):
                logprob += jnp.sum(dist.log_prob(value))
            return logprob

        #
        return logprior_fn_

    #
    def inference(self, key: PRNGKey, mode='gibbs-in-smc', sampling_parameters: Dict = None, poisson = False):
        """A wrapper for training the GP model.

        An interface to Blackjax' MCMC or SMC inference loops, tailored to the
        current Bayesian model.

        Args:
            key: jrnd.KeyArray
                The random number seed, will be split into initialisation and
                inference.
            mode: {'mcmc', 'smc'}
                The desired inference approach. Defaults to SMC, which is
                generally prefered.
            sampling_parameters: dict
                Optional settings with defaults for the inference procedure.

        Returns:
            Depending on 'mode':
                smc:
                    num_iter: int
                        Number of tempering iterations.
                    particles:
                        The final states the SMC particles (at T=1).
                    marginal_likelihood: float
                        The approximated marginal likelihood of the model.
                mcmc:
                    states:
                        The MCMC states (including burn-in states).

        """
        if sampling_parameters is None:
            sampling_parameters = dict()

        key, key_init, key_inference = jrnd.split(key, 3)

        if mode == 'gibbs-in-smc' and not (hasattr(self, 'gibbs_fn') and callable(self.gibbs_fn)):
            sigma = 0.01
            print(f'No Gibbs kernel available, defaulting to Random Walk Metropolis MCMC, sigma = {sigma:.2f}') 
            mode = 'mcmc-in-smc'
            priors_flat, _ = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
            m = 0            
            for prior in priors_flat:
                m += jnp.prod(jnp.asarray(prior.batch_shape)) if prior.batch_shape else 1
                # m += jnp.prod(jnp.asarray(prior.batch_shape))

            sampling_parameters['kernel'] = rmh
            sampling_parameters['kernel_parameters'] = dict(sigma=sigma*jnp.eye(m))


        if mode == 'gibbs-in-smc' or mode == 'mcmc-in-smc':
            if mode == 'gibbs-in-smc':
                mcmc_step_fn = self.gibbs_fn
                mcmc_init_fn = self.smc_init_fn
            elif mode == 'mcmc-in-smc':

                # Set up tempered MCMC kernel
                def mcmc_step_fn(key, state, temperature, **mcmc_parameters):
                    def apply_mcmc_kernel(key, logdensity, pos):
                        kernel = kernel_type(logdensity, **kernel_parameters)
                        state_ = kernel.init(pos)
                        state_, info = kernel.step(key, state_)
                        return state_.position, info
                    
                    #
                    key, key_poisson = jrnd.split(key, 2)
                    
                    p = jrnd.bernoulli(key=key_poisson)   

                    def logdensity_(key, new_pos, old_pos, temperature):
                            key, key_accept = jrnd.split(key, 2)
                            # jax.debug.print("new_pos prob: {}", temperature * loglikelihood_fn_(new_pos) + logprior_fn_(new_pos))
                            # jax.debug.print("old_pos prob: {}", temperature * loglikelihood_fn_(old_pos) + logprior_fn_(old_pos))
                            delta = (temperature * loglikelihood_fn_(new_pos) + logprior_fn_(new_pos)) - \
                                    (temperature * loglikelihood_fn_(old_pos) + logprior_fn_(old_pos))
                            delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
                            p_accept = jnp.clip(jnp.exp(delta), a_max=1.0)
                            # jax.debug.print("p_accept: {}", p_accept)
                            do_accept = jax.random.bernoulli(key_accept, p_accept)
                            # accept_state = (new_pos, RMHInfo(p_accept, True, new_pos))
                            # reject_state = (old_pos, RMHInfo(p_accept, False, new_pos))

                            return jax.lax.cond(
                                do_accept, lambda _: new_pos, lambda _: old_pos, operand=None
                            )
                    
                    def poisson_process_step(key, position, temperature):

                        new_position = position.copy()

                        ## num parameter
                        num_params = jnp.sort(new_position['kernel']['num'])

                        def nothing(params, index):
                            return params
                        # if p: # add value
                        def add_param(params, index):
                            new_val = jrnd.uniform(key)
                            new_num_params = params.at[-1].set(new_val)
                            
                            return new_num_params

                        def add_num(params):
                            index = jnp.nanargmax(params).astype(int)
                            new_num_params = jax.lax.cond(index != (params.shape[0]-1), add_param, nothing, params, index)
                            new_index = jnp.argmax(jnp.argsort(new_num_params))
                            new_num_params = jnp.sort(new_num_params)
                            return new_num_params, new_index
                        
                        def remove_param(params, index):
                            new_num_params = params.at[index].set(jnp.nan)
                            return new_num_params
                        
                        def remove_num(params):
                            index = jnp.rint(jrnd.uniform(key)*jnp.nanargmax(params)).astype(int)
                            new_num_params = jax.lax.cond(jnp.nanargmax(params) != -1, remove_param, nothing, params, index)
                            # if index != 0:
                            return new_num_params, index

                        new_num_params, index = jax.lax.cond(p, add_num, remove_num, num_params)
                            
                        # else: # remove value
                            
                        new_position['kernel']['num'] = new_num_params
                        ##
                        ## covariance parameters
                        cov_params = dict(lengthscale = position['kernel']['lengthscale'],
                                         variance = position['kernel']['variance'])
                        cov_priors = dict(lengthscale = self.param_priors['kernel']['lengthscale'],
                                          variance = self.param_priors['kernel']['variance'])
                        priors_flat, priors_treedef = tree_flatten(cov_priors, lambda l: isinstance(l, (Distribution, Bijector)))
                        values_flat, _ = tree_flatten(cov_params)
                        # children = priors_treedef.children()
                        names = [x for x in priors_treedef.node_data()[1]]
                        # jax.debug.print(names)
                        # names = [item for sublist in leaves for item in sublist]
                        # jax.debug.print("pre_old_pos_l: {}", position['kernel']["lengthscale"] )
                        


                        def new_nothing(value, index, dist):
                            return value

                        def swap_param(i, value):
                            indices = jnp.array([value.shape[0]-1-(i+1), value.shape[0]-1-(i)])
                            swap_vals = jnp.array([value[value.shape[0]-1-(i)], value[value.shape[0]-1-(i+1)]])
                            value = value.at[indices].set(swap_vals)
                            return value

                        def add_cov_param(value, index, dist):
                            # jax.debug.print("index: {}", index)
                            # jax.debug.print("pre-swapped: {}", value)
                            k = jnp.zeros((1, value.shape[0]))
                            k = k.at[:].set(jnp.nan)
                            k = k.at[0].set(0)
                            value = value.at[-1].set(dist._sample_n(key_poisson, k, 1)[0, 0])
                            # indices = jnp.arange(index+1, value.shape[0])
                            # swap_vals = jnp.array([value[-1]].append(value[(index+1):(value.shape[0]-1)]))
                            # new_value = value.at[indices].set(swap_vals)
                            new_value = jax.lax.fori_loop(0, value.shape[0]-1-(index+1), swap_param, value)
                            # jax.debug.print("swapped: {}", new_value)
                            return new_value

                        def add_cov(value, index, dist):
                            value = jax.lax.cond(index != len(num_params-1), add_cov_param, new_nothing, value, index, dist)
                            return value
                        
                        def swap_nan(i, value):
                            indices = jnp.array([i, i+1])
                            swap_vals = jax.lax.cond(jnp.isnan(value[i]) & ~jnp.isnan(value[i+1]), 
                                                     lambda _: jnp.array([value[i+1], value[i]]),
                                                     lambda _: jnp.array([value[i], value[i+1]]), 
                                                     operand=None)
                            value = value.at[indices].set(swap_vals)
                            return value

                        def remove_cov_param(value, index):
                            value = value.at[index+1].set(jnp.nan)
                            # indices = jnp.arange(index+1, value.shape[0])
                            # swap_vals = jnp.array((value[(index+2):(value.shape[0])]).append([value[index+1]]))
                            # new_value = value.at[indices].set(swap_vals)
                            new_value = jax.lax.fori_loop(0, value.shape[0]-2, swap_nan, value)
                            return new_value

                        def remove_cov(value, index, dist):
                            value = jax.lax.cond(index != -1, remove_cov_param, nothing, value, index)
                            # if index != 0:
                                
                            return value
                        
                        for value, dist, name in zip(values_flat, priors_flat, names):
                            value = jax.lax.cond(p, add_cov, remove_cov, value, index, dist)
                            new_position['kernel'][name] = value

                        # jax.debug.print("new_pos_l: {}", new_position['kernel']["lengthscale"] )
                        return new_position
                        
                    # OldGibbsState = GibbsState(position=state.position)
                    position = state.position.copy()
                    loglikelihood_fn_ = self.loglikelihood_fn()
                    logprior_fn_ = self.logprior_fn()
                    logdensity = lambda state: temperature * loglikelihood_fn_(state) + logprior_fn_(state)

                    # kernel = kernel_type(logdensity, **kernel_parameters)
                    # state_ = kernel.init(state.position)

                    ptree, unravel_fn = ravel_pytree(position)
                    sample = jnp.zeros(shape=ptree.shape, dtype=ptree.dtype)
                    move_proposal = unravel_fn(sample)
                    old_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
                    # old_log_dense = logdensity(state_.position)

                    if poisson:
                        new_pos = poisson_process_step(key_poisson, position, temperature)
                        poisson_pos = logdensity_(key, new_pos, old_position, temperature)
                        new_position, info_ = apply_mcmc_kernel(key, logdensity, poisson_pos)
                    else: 
                        new_position, info_ = apply_mcmc_kernel(key, logdensity, position)
                    return GibbsState(position=new_position), None  

                #
                kernel_type = sampling_parameters.get('kernel')
                kernel_parameters = sampling_parameters.get('kernel_parameters')
                mcmc_init_fn = self.smc_init_fn
            
            #Set up adaptive tempered SMC
            smc = adaptive_tempered_smc(
                logprior_fn=self.logprior_fn(),
                loglikelihood_fn=self.loglikelihood_fn(),
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                mcmc_parameters=sampling_parameters.get('mcmc_parameters', dict()),
                resampling_fn=resampling.systematic,
                target_ess=sampling_parameters.get('target_ess', 0.5),
                num_mcmc_steps=sampling_parameters.get('num_mcmc_steps', 100)
            )
            num_particles = sampling_parameters.get('num_particles', 1_000)
            include_trace = sampling_parameters.get('include_trace', False)
            initial_particles = self.init_fn(key_init,
                                             num_particles=num_particles)
            initial_smc_state = smc.init(initial_particles.position)   
            
            if include_trace:
                smc_output = smc_inference_loop_trace(key_inference,
                                                      smc.step,
                                                      initial_smc_state)
                num_iter, particles, marginal_likelihood, trace = smc_output
            else:
                smc_output = smc_inference_loop(key_inference,
                                                smc.step,
                                                initial_smc_state)
                num_iter, particles, marginal_likelihood = smc_output
                
            self.particles = particles
            self.marginal_likelihood = marginal_likelihood

            if include_trace:
                return particles, num_iter, marginal_likelihood, trace
            return particles, num_iter, marginal_likelihood
        elif mode == 'gibbs' or mode == 'mcmc':
            num_burn = sampling_parameters.get('num_burn', 10_000)
            num_samples = sampling_parameters.get('num_samples', 10_000)
            num_thin = sampling_parameters.get('num_thin', 1)

            if mode == 'gibbs':
                step_fn = self.gibbs_fn
                initial_state = self.init_fn(key_init)
            elif mode == 'mcmc':
                kernel_type = sampling_parameters.get('kernel')
                kernel_parameters = sampling_parameters.get('kernel_parameters')
                loglikelihood_fn = self.loglikelihood_fn()
                logprior_fn = self.logprior_fn()

                logdensity_fn = lambda state: loglikelihood_fn(state) + logprior_fn(state)
                kernel = kernel_type(logdensity_fn, **kernel_parameters)
                step_fn = kernel.step
                initial_state = sampling_parameters.get('initial_state', kernel.init(self.init_fn(key_init).position))

            states = inference_loop(key_inference,
                                    step_fn,
                                    initial_state,
                                    num_burn + num_samples)

            # remove burn-in
            self.states = tree_map(lambda x: x[num_burn::num_thin], states)
            return states
        else:
            raise NotImplementedError(f'{mode} is not implemented as inference method. Valid options are:\ngibbs-in-smc\ngibbs\nmcmc-in-smc\nmcmc')

    #
    def get_monte_carlo_samples(self, mode='smc'):
        if mode == 'smc' and hasattr(self, 'particles'):
            return self.particles.particles
        elif mode == 'mcmc' and hasattr(self, 'states'):
            return self.states.position
        raise ValueError('No inference has been performed')

    #
    def plot_priors(self, axes=None):
        pass

    #

#
