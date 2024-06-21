from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
import matplotlib.pyplot as plt
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
from jax.tree_util import tree_flatten, tree_unflatten
from gputil_new import sample_prior, sample_predictive

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from fullgp import FullLatentGPModelhyper_mult
from fullgp import FullMarginalGPModelhyper_mult
import copy 


import jax 
import distrax as dx
import jaxkern as jk
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd
from blackjax import elliptical_slice, rmh

def potential_scale_reduction(
    input_array, chain_axis: int = 0, sample_axis: int = 1
):
    """Gelman and Rubin (1992)'s potential scale reduction for computing multiple MCMC chain convergence.

    Parameters
    ----------
    input_array:
        An array representing multiple chains of MCMC samples. The array must
        contains a chain dimension and a sample dimension.
    chain_axis
        The axis indicating the multiple chains. Default to 0.
    sample_axis
        The axis indicating a single chain of MCMC samples. Default to 1.

    Returns
    -------
    NDArray of the resulting statistics (r-hat), with the chain and sample dimensions squeezed.

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \\hat{R} = \\frac{\\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\\hat{V}` is the posterior variance
    estimate for the pooled traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged :cite:p:`stan_rhat,gelman1992inference`.

    """
    assert (
        input_array.shape[chain_axis] > 1
    ), "potential_scale_reduction as implemented only works for two or more chains."

    num_samples = input_array.shape[sample_axis]
    # Compute stats for each chain
    per_chain_mean = jnp.nanmean(input_array, axis=sample_axis, keepdims=True)
    per_chain_var = jnp.nanvar(input_array, axis=sample_axis, ddof=1, keepdims=True)
    # Compute between-chain stats
    between_chain_variance = num_samples * per_chain_mean.var(
        axis=chain_axis, ddof=1, keepdims=True
    )
    # Compute within-chain stats
    within_chain_variance = per_chain_var.mean(axis=chain_axis, keepdims=True)
    # Estimate of marginal posterior variance
    rhat_value = jnp.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1)
        / (num_samples)
    )
    return rhat_value.squeeze()

    
class GP_CP_Marginal():
    def __init__(self, X, y: Optional[Array]=None,
                 cov_fn: Optional[Callable]=None,
                 mean_fn: Callable = None,
                 priors: Dict = None, 
                 num_particles: int = None,
                 num_mcmc_steps: int = None,
                 ground_truth: Dict = None):
            if jnp.ndim(X) == 1:
                X = X[:, jnp.newaxis]  
            if cov_fn is None:
                raise ValueError(
                    f'Provide a covariance function for the GP!')
            # Validate arguments
            if y is not None and X.shape[0] > len(y):
                raise ValueError(
                    f'X and y should have the same leading dimension, '
                    f'but X has shape {X.shape} and y has shape {y.shape}.',
                    f'Use the `FullLatentGPModelRepeatedObs` model for repeated inputs.')
            self.X, self.y = X, y        
            self.n = self.X.shape[0]        
            if mean_fn is None:
                mean_fn = Zero()
            self.mean_fn = mean_fn
            self.cov_fn = cov_fn
            self.param_priors = priors
            self.num_particles = num_particles
            self.num_mcmc_steps = num_mcmc_steps
            self.particles = None
            self.gp_fit = FullMarginalGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors)
            self.likelihood = None
            if isinstance(self.cov_fn, jk.base.CombinationKernel):
                self.kernel_name = [kernel.name for kernel in cov_fn.kernel_set]
            else:
                self.kernel_name = [cov_fn.name]
            self.ground_truth = ground_truth

    # def predict_f_particle(self, key: Array, x_pred: ArrayTree, particles, num_subsample=-1):
    #     """Predict the latent f on unseen pointsand

    #     This function takes the approximated posterior (either by MCMC or SMC)
    #     and predicts new latent function evaluations f^*.

    #     Args:
    #         key: PRNGKey
    #         x_pred: x^*; the queried locations.
    #         num_subsample: By default, we return one predictive sample for each
    #         posterior sample. While accurate, this can be memory-intensive; this
    #         parameter can be used to thin the MC output to every n-th sample.

    #     Returns:
    #         f_samples: An array of samples of f^* from p(f^* | x^*, x, y)


    #     todo:
    #     - predict using either SMC or MCMC output
    #     - predict from prior if desired
    #     """
    #     if jnp.ndim(x_pred) == 1:
    #         x_pred = x_pred[:, jnp.newaxis]

    #     samples = particles
    #     flat_particles, _ = tree_flatten(samples)
    #     num_particles = flat_particles[0].shape[0]
    #     key_samples = jrnd.split(key, num_particles)

    #     mean_params = samples.get('mean', {})
    #     cov_params = samples['kernel']
    #     mean_params_in_axes = jax.tree_map(lambda l: 0, mean_params)
    #     cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
    #     sample_fun = lambda key, mean_params_, cov_params_, obs_noise_: sample_predictive(key,
    #                                                                         mean_params=mean_params_,
    #                                                                         cov_params=cov_params_,
    #                                                                         mean_fn=self.mean_fn,
    #                                                                         cov_fn=self.cov_fn,
    #                                                                         x=self.X,
    #                                                                         z=x_pred,
    #                                                                         target=self.y,
    #                                                                         obs_noise=obs_noise_)
    #     keys = jrnd.split(key, num_particles)
    #     target_pred = jax.vmap(jax.jit(sample_fun),
    #                     in_axes=(0,
    #                             {k: 0 for k in mean_params},
    #                             cov_param_in_axes,
    #                             0))(keys,
    #                                     mean_params,
    #                                     cov_params,
    #                                     samples['likelihood']['obs_noise'])

    #     return target_pred
    

    # def predict_y_particle(self, key, x_pred: Array, particles):
    #     """Samples from the posterior predictive distribution

    #     Args:
    #         key: PRNGKey
    #         x_pred: Array
    #             The test locatons
    #     Returns:
    #         Returns samples from the posterior predictive distribution:

    #         y* \sim p(y* | X, y x*) = \int p(y* | f*)p(f* | f)p(f | X, y) df

    #     """
    #     if jnp.ndim(x_pred) == 1:
    #         x_pred = x_pred[:, jnp.newaxis]

    #     samples = particles
    #     if samples is None:
    #         raise AssertionError(
    #             f'The posterior predictive distribution can only be called after training.')

    #     def forward(key, params, f):
    #         return self.likelihood.likelihood(params, f).sample(seed=key)

    #     #
    #     key, key_f, key_y = jrnd.split(key, 3)
    #     f_pred = self.predict_f_particle(key_f, x_pred, particles)
    #     flat_particles, _ = tree_flatten(samples)
    #     num_particles = flat_particles[0].shape[0]
    #     keys_y = jrnd.split(key_y, num_particles)
    #     likelihood_params = samples['likelihood']
    #     y_pred = jax.vmap(jax.jit(forward),
    #                         in_axes=(0,
    #                                 0,
    #                                 0))(keys_y,
    #                                 likelihood_params,
    #                                 f_pred)
    #     return y_pred

    def model_GP(self, key):
        print('Running Marginal GP')
        kernel = self.cov_fn

        priors = self.param_priors
        gp_marginal = self.gp_fit  # Implies likelihood=Gaussian()
        key, gpm_key = jrnd.split(key)
        mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
                                                                        mode='gibbs-in-smc',
                                                                        sampling_parameters=dict(num_particles=self.num_particles, num_mcmc_steps=self.num_mcmc_steps),
                                                                        poisson = True)
        self.particles = mgp_particles
        self.gp_fit = gp_marginal
        self.likelihood = mgp_marginal_likelihood

    # def model_GP(self, key):
        
    #     self._model_GP(key)

        
    def plot_post(self, ground_truth=None):
        ''' Only plots up to a maximum of 5 posteriors per default'''
            
        if isinstance(self.particles.particles['kernel'], dict):
            isdict = True
            num_kernels = 1
            # num_particles = particles.particles['kernel'][trainables[0]].shape[0]
            # num_CPs = jnp.max(jnp.sum(~jnp.isnan(particles.particles['kernel']['num']), axis = 1))
        else:
            isdict = False
            num_kernels = len(self.particles.particles['kernel'])
            # num_particles = particles.particles['kernel'][0][trainables[0]].shape[0]
        for k in range(num_kernels):
            if isdict:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel']['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            else:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel'][k]['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'][k])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            if trainables == []:
                raise ValueError(
                    f'No posteriors to plot!')
            
            num_params = len(trainables)

            symbols = [fr'{name[0]}' for name in trainables]
            
            num_CP = jnp.minimum(num_CPs, 5).tolist()
            _, axes = plt.subplots(nrows=num_params, ncols=num_CP+1, constrained_layout=True,
                                figsize=(16, 6))
            
            if num_CP == 0:
                axes = axes[:, jnp.newaxis]
            elif num_params == 1:
                axes = axes[:, jnp.newaxis].T

            for j, var in enumerate(trainables):
                    pd = tr[var]
                    for i in range(num_CP+1):
                        # There are some outliers that skew the axes
                        # pd_u, pd_l = jnp.nanpercentile(pd[:, i], q=99.9), jnp.nanpercentile(pd[:, i], q=0.1)
                        # pd_filtered = jnp.extract(pd[:, i]>pd_l, pd[:, i])
                        # pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
                        axes[j, i].hist(pd[~jnp.isnan(pd[:, i]), i], bins=30, density=True, color='tab:blue')
                        # axes[j, i].hist(pd[:, i][~jnp.isnan(pd[:, i])], bins=30, density=True, color='tab:blue')
                        if ground_truth is not None:
                            if isdict:
                                if len(ground_truth['kernel'][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][var][i], ls=':', c='k')
                            else:
                                if len(ground_truth['kernel'][k][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][k][var][i], ls=':', c='k')
                        
                        axes[j, i].set_xlabel(r'${:s}$'.format(f'{symbols[j]}_{i}'))


                    axes[j, 0].set_ylabel(var, rotation=0, ha='right')
                
            plt.suptitle(f'Posterior estimate of Bayesian Marginal GP {self.kernel_name[k]} kernel ({self.num_particles} particles)')
            plt.show();

    def _plot_fit(self, key, predict=True, f_true = None, ground_truth = None, particles = None):
        if predict:
            x_pred = jnp.linspace(-0.25, 1.25, num=int(1.5 * len(self.y)))
        else:
            x_pred = jnp.linspace(0, 1, num=len(self.y))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), sharex=True,
                                    sharey=True, constrained_layout=True)
        
    
        key, key_pred = jrnd.split(key)
        if particles is None:
            if isinstance(self.particles, dict):
                parts = self.particles
            else:
                parts = self.particles.particles
            f_pred = self.gp_fit.predict_f(key_pred, x_pred)
        else:
            parts = particles
            f_pred = self.gp_fit.predict_f_particle(key_pred, x_pred, particles)

        # There are some outliers that skew the axis
        # pd_u, pd_l = jnp.percentile(pd, q=99.9), jnp.percentile(pd, q=0.1)
        # pd_filtered = jnp.extract(pd>pd_l, pd)
        # pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)

        num_particles = self.num_particles
        ax = axes[0]
        for i in jnp.arange(0, num_particles, step=10):
            ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')


        colors = plt.cm.jet(jnp.linspace(0.3,1, len(self.kernel_name)))

        ax2 = ax.twinx()
        if isinstance(parts['kernel'], dict):
            pd = parts['kernel']['num']
            new_pd = pd[jnp.logical_not(jnp.isnan(pd))]
            ax2.hist(new_pd, bins=30, density=True, color='tab:blue', alpha=0.5)
            if ground_truth is not None:
                if 'num' in ground_truth['kernel'].keys():
                    for CP in ground_truth['kernel']['num']:
                        ax2.axvline(x=CP, ls=':', c='black')
        else:
            for i, pd in enumerate(parts['kernel']):
                new_pd = pd['num'][jnp.logical_not(jnp.isnan(pd['num']))]
                ax2.hist(new_pd, bins=30, density=True, color=colors[i], label = self.kernel_name[i], alpha=0.5)
                if ground_truth is not None:
                    if 'num' in ground_truth['kernel'][i].keys():
                        for CP in ground_truth['kernel'][i]['num']:
                            ax2.axvline(x=CP, ls=':', c=colors[i])
            ax2.legend()

        ax = axes[1]
        f_mean = jnp.nanmean(f_pred, axis=0)
        if particles is None:
            y_pred = self.gp_fit.predict_y(key_pred, x_pred)
        else:
            y_pred = self.gp_fit.predict_y(key_pred, x_pred, particles)
        
        f_hdi_lower = jnp.nanpercentile(y_pred, q=2.5, axis=0)
        f_hdi_upper = jnp.nanpercentile(y_pred, q=97.5, axis=0)
        # f_hdi_lower = jnp.percentile(f_pred, q=2.5, axis=0)
        # f_hdi_upper = jnp.percentile(f_pred, q=97.5, axis=0)

        ax.plot(x_pred, f_mean, color='tab:blue', lw=2)
        ax.fill_between(x_pred, f_hdi_lower, f_hdi_upper,
                        alpha=0.2, color='tab:blue', lw=0)
        # ax.set_title('Posterior 95% HDI', fontsize=16)
        #print(new_pd.shape)
        ax2 = ax.twinx()
        if isinstance(parts['kernel'], dict):
            pd = parts['kernel']['num']
            new_pd = pd[jnp.logical_not(jnp.isnan(pd))]
            ax2.hist(new_pd, bins=30, density=True, color='tab:blue', alpha=0.5)
            if ground_truth is not None:
                if 'num' in ground_truth['kernel'].keys():
                    for CP in ground_truth['kernel']['num']:
                        ax2.axvline(x=CP, ls=':', c='black')
        else:
            for i, pd in enumerate(parts['kernel']):
                new_pd = pd['num'][jnp.logical_not(jnp.isnan(pd['num']))]
                ax2.hist(new_pd, bins=30, density=True, color=colors[i], label = self.kernel_name[i], alpha=0.5)
                if ground_truth is not None:
                    if 'num' in ground_truth['kernel'][i].keys():
                        for CP in ground_truth['kernel'][i]['num']:
                            ax2.axvline(x=CP, ls=':', c=colors[i])
            ax2.legend()
        ax2.set_ylabel('CP probability', fontsize=16)

        for ax in axes:
            if f_true is not None:
                ax.plot(self.X, f_true, 'k', label=r'$f$')
            ax.plot(self.X, self.y, 'rx', label='obs')
            if predict:
                ax.set_xlim([-0.25, 1.25])
            else:
                ax.set_xlim([0., 1.])
            ax.set_ylim([jnp.min(self.y)-.5, jnp.max(self.y)+.5])
            ax.set_xlabel(r'$x$', fontsize=12)
            ax.set_ylabel(r'$y$', fontsize=12)
        return axes

    def plot_fit(self, key, predict=False, f_true = None, ground_truth = None, particles = None):
        axes = self._plot_fit(key, predict, f_true, ground_truth, particles)
        axes[0].set_title('SMC particles', fontsize=16)
        axes[0].set_ylabel('Marginal GP', rotation=0, ha='right', fontsize=14)
        axes[1].set_title('Posterior 95% HDI', fontsize=16)
        plt.show()

    def _plot_num(self):
        fig = plt.figure(figsize=(12, 6))
        colors = plt.cm.jet(jnp.linspace(0.3,1, len(self.kernel_name)))
        counts = jnp.zeros((self.num_particles, len(self.kernel_name)))

        if isinstance(self.particles.particles['kernel'], dict):
            pd = self.particles.particles['kernel']['num']
            counts = counts.at[:, 0].set(jnp.count_nonzero(~jnp.isnan(pd), axis = 1))
            uni_vals = jnp.sort(jnp.concatenate([jnp.unique(counts)-0.5, jnp.unique(counts)+0.5]))
        else:
            for i, pd in enumerate(self.particles.particles['kernel']):
                num_val = pd['num']
                counts = counts.at[:, i].set(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
                uni_vals = jnp.sort(jnp.concatenate([jnp.unique(counts)-0.5, jnp.unique(counts)+0.5]))
            
        plt.hist(counts.T, bins=uni_vals, rwidth = 0.5, color=colors, label=self.kernel_name)

    def plot_num(self):
        self._plot_num()
        plt.xlabel("Amount of change points")
        plt.ylabel("Amount of particles")
        plt.title('Amount of Change Points in Marginal GP')
        plt.legend()
        plt.show()

    def number_metric(self, ground_truth):
        if isinstance(self.particles.particles['kernel'], dict):
            true_number = len(ground_truth['kernel']['num'])
            num_val = self.particles.particles['kernel']['num']
            counts = jnp.mean(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
            max_num = jnp.maximum(num_val.shape[1] - true_number, true_number)
            return (counts - true_number)/max_num
        
        metric = jnp.zeros(len(self.particles.particles['kernel']))
        for i, kernel in enumerate(self.particles.particles['kernel']):
            true_number = len(ground_truth['kernel'][i]['num'])
            num_val = kernel['num']
            counts = jnp.mean(jnp.count_nonzero(~jnp.isnan(num_val), axis = 1))
            max_num = jnp.maximum(num_val.shape[1] - true_number, true_number)
            metric = metric.at[i].set((counts - true_number)/max_num)
        return dict(zip(self.kernel_name, metric.tolist()))
    
    def _loc_calculation(self, locs, diffs, true_locations, max_num, max_dist):
        for i, loc in enumerate(locs):       
            if jnp.count_nonzero(~jnp.isnan(loc)) == 0:
                dist = jnp.zeros((len(true_locations), 1))
            else:
                dist = jnp.zeros((len(true_locations), jnp.maximum(1, jnp.count_nonzero(~jnp.isnan(loc)))))
                for j, true_loc in enumerate(true_locations):
                    true_locs = jnp.sort(self.X.squeeze()[jnp.argsort(jnp.abs(self.X.squeeze() - true_loc))[:2]])
                    dist1 = (loc[~jnp.isnan(loc)] - true_locs[0])
                    dist2 = (loc[~jnp.isnan(loc)] - true_locs[1])
                    dist_comp = dist1 * dist2
                    dist_min = jnp.minimum(jnp.abs(dist1), jnp.abs(dist2))
                    dist_min = dist_min.at[dist_comp < 0].set(0)
                    dist = dist.at[j, :].set(jnp.abs(dist_min.squeeze()))
            # print(loc[~jnp.isnan(loc)])
            num_diff = jnp.abs(len(true_locations) - jnp.count_nonzero(~jnp.isnan(loc)))
            # print(num_diff)
            sorted_min_dist = jnp.sort(jnp.min(dist, axis = 1))
            # sorted_min_dist = sorted_min_dist.at[-num_diff:].set(max_dist)
            # print(sorted_min_dist)
            if num_diff == 0:
                # print(sorted_min_dist)
                diffs = diffs.at[i].set(jnp.sum(sorted_min_dist)/max_num)
            else: 
                sorted_min_dist = sorted_min_dist.at[-num_diff:].set(max_dist)
                diffs = diffs.at[i].set(jnp.sum(sorted_min_dist)/max_num)
        return jnp.mean(diffs)


    def location_metric(self, ground_truth):
        if isinstance(self.particles.particles['kernel'], dict):
            true_locations = ground_truth['kernel']['num']
            locs = self.particles.particles['kernel']['num']
            diffs = jnp.zeros(len(locs))
            max_num = locs.shape[1]
            max_dist = 1
            return self._loc_calculation(locs, diffs, true_locations, max_num, max_dist)
            
        
        metric = jnp.zeros(len(self.particles.particles['kernel']))
        for i, kernel in enumerate(self.particles.particles['kernel']):
            true_locations = ground_truth['kernel'][i]['num']
            locs = kernel['num']
            diffs = jnp.zeros(len(locs))
            max_num = locs.shape[1]
            max_dist = 1
            metric = metric.at[i].set(self._loc_calculation(locs, diffs, true_locations, max_num, max_dist))
        return dict(zip(self.kernel_name, metric.tolist()))
    

    def likelihood_metric(self, key):
        size = len(self.y)
        x_pred = jnp.linspace(-0, 1, num=size)
        key, key_pred = jrnd.split(key)
        if self.gp_fit is not None:
            f_pred = self.gp_fit.predict_f(key_pred, x_pred)
        else: 
            raise ValueError('No GP trained yet!')
        
        f_mean = jnp.nanmean(f_pred, axis=0)
        cov = jnp.zeros(size) + jnp.mean(self.particles.particles['likelihood']['obs_noise'])
        MVN = dx.MultivariateNormalDiag(f_mean, cov)
        return MVN.log_prob(self.y)
    

    # def _jaccard_calculation(self, locs, true_locations, card_y):
    #     jacc_index = jnp.zeros(locs.shape[0])
    #     for i, loc in enumerate(locs): 
    #         card_GP = jnp.sum(~jnp.isnan(loc))
    #         intersection = 0
    #         for true_loc in true_locations:
    #             true_locs = jnp.sort(self.X.squeeze()[jnp.argsort(jnp.abs(self.X.squeeze() - true_loc))[:2]])
    #             dist1 = (loc[~jnp.isnan(loc)] - true_locs[0])
    #             dist2 = (loc[~jnp.isnan(loc)] - true_locs[1])
    #             dist_comp = dist1 * dist2
    #             if len(jnp.array([dist_comp < 0]).nonzero()[0]) > 0:
    #                 intersection += 1
            
    #         jacc_index = jacc_index.at[i].set(intersection/(card_GP + card_y + intersection))

    #     return jnp.mean(jacc_index)

    def jaccard_metric(self, ground_truth):
        if isinstance(self.particles.particles['kernel'], dict):
            truth_K = self.cov_fn(ground_truth['kernel'], self.X, self.X)
            truth_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(truth_K.flatten()), (len(self.X), len(self.X)))
            # sum_diff = jnp.zeros(self.num_particles)
            kernel = self.particles.particles['kernel']
            cov_param_in_axes = jax.tree_map(lambda l: 0, kernel)
            est_K = jax.vmap(lambda a: self.cov_fn.cross_covariance(params = a, x= self.X, y=self.X), in_axes=(cov_param_in_axes, ))(kernel)
            est_K_abs = jax.vmap(lambda a: jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(a.flatten()), (len(self.X), len(self.X))), in_axes=(0, ))(est_K)
            diff_K = 1 - jnp.abs(truth_K_abs - est_K_abs)
            sum_diff = jnp.sum(diff_K.flatten())/(len(self.X)**2 * self.num_particles)
            # for j in range(self.num_particles):
            #     vals = [x[j] for x in kernel.values()]
            #     params = dict(zip(kernel.keys(), vals))
            #     est_K = self.cov_fn.cross_covariance(params = params, x= self.X, y=self.X)
            #     est_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(est_K.flatten()), (len(self.X), len(self.X)))
            #     diff_K = 1 - jnp.abs(truth_K_abs - est_K_abs)
            #     sum_diff = sum_diff.at[j].set(jnp.sum(diff_K.flatten())/(len(self.X)**2))
            # est_K = self.cov_fn(self.particles.particles['kernel'], self.X, self.X)
            # truth_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(truth_K.flatten()), (len(self.X), len(self.X)))
            # est_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(est_K.flatten()), (len(self.X), len(self.X)))
            # diff_K = jnp.abs(truth_K_abs - est_K_abs)
            # sum_diff = jnp.sum(diff_K.flatten())/(len(self.X)**2)
            # locs = self.particles.particles['kernel']['num']
            # true_locations = ground_truth['num']
            # card_y = len(true_locations)
            # return self._jaccard_calculation(locs, true_locations, card_y)
            return jnp.mean(sum_diff)
        
        metric = jnp.zeros(len(self.particles.particles['kernel']))
        for i, kernel in enumerate(self.particles.particles['kernel']):
            sum_diff = jnp.zeros(self.num_particles)
            truth_K = self.cov_fn.kernel_set[i].cross_covariance(params = ground_truth['kernel'][i], x = self.X, y = self.X)
            for j in range(self.num_particles):
                vals = [x[j] for x in kernel.values()]
                params = dict(zip(kernel.keys(), vals))
                est_K = self.cov_fn.kernel_set[i].cross_covariance(params = params, x= self.X, y=self.X)
                truth_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(truth_K.flatten()), (len(self.X), len(self.X)))
                est_K_abs = jnp.reshape(jax.vmap(lambda x: jax.lax.cond(x > 0, lambda: 1, lambda: 0))(est_K.flatten()), (len(self.X), len(self.X)))
                diff_K = 1 - jnp.abs(truth_K_abs - est_K_abs)
                sum_diff = sum_diff.at[j].set(jnp.sum(diff_K.flatten())/(len(self.X)**2))
            metric = metric.at[i].set(jnp.mean(sum_diff))
            # locs = self.particles.particles['kernel']['num']
            # true_locations = ground_truth['num']
            # card_y = len(true_locations)
            # return self._jaccard_calculation(locs, true_locations, card_y)
        return dict(zip(self.kernel_name, metric.tolist()))
    
    # def convergence_check(self):
    #     if isinstance(self.particles.particles['kernel'], dict):
    #         ### Kernel convergence
    #         kernel_vals = self.particles.particles['kernel'].values()
    #         kernel_names = self.particles.particles['kernel'].keys()
    #         kernel_conv = jnp.zeros((len(self.particles.particles['kernel']), kernel_vals.shape[1]))
    #         fig = plt.figure(figsize = (16, 6))
            
    #         for i, (vals, name) in enumerate(zip(kernel_vals, kernel_names)):
    #             kernel_conv = kernel_conv.at[i].set(potential_scale_reduction(vals, 2, 0))
    #             plt.plot(kernel_conv[i, :], label = f'{name}')
    #         plt.legend()
    #         plt.show()

    #     ### Hyper convergence
    #     hyper_vals = self.particles.particles['kernel'].values()
    #     hyper_names = self.particles.particles['kernel'].keys()
    #     hyper_conv = potential_scale_reduction(hyper_vals, 1, 0)
        
    #     ### Obs noise convergence


class GP_CP_Latent(GP_CP_Marginal):
    def __init__(self, X, y: Optional[Array]=None,
                 cov_fn: Optional[Callable]=None,
                 mean_fn: Callable = None,
                 priors: Dict = None, 
                 num_particles: int = None,
                 num_mcmc_steps: int = None,
                 likelihood = None,
                 **kwargs):
        super().__init__(X, y, cov_fn, mean_fn, priors, num_particles, num_mcmc_steps, **kwargs)        
        if [likelihood] is not None:
            self.gp_fit = FullLatentGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors, likelihood=likelihood)
        else:
            self.gp_fit = FullLatentGPModelhyper_mult(self.X, self.y, cov_fn=self.cov_fn, priors=self.param_priors)

    def model_GP(self, key):
        print('Running Latent GP')
        kernel = self.cov_fn

        priors = self.param_priors
        gp_latent = self.gp_fit # Implies likelihood=Gaussian()
        key, gpm_key = jrnd.split(key)
        lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpm_key,
                                                                        mode='gibbs-in-smc',
                                                                        sampling_parameters=dict(num_particles=self.num_particles, num_mcmc_steps=self.num_mcmc_steps),
                                                                        poisson = True)
        self.particles = lgp_particles
        self.gp_fit = gp_latent
        self.likelihood = lgp_marginal_likelihood

    def plot_fit(self, key, predict=True, f_true = None, ground_truth = None, particles = None):
        axes = self._plot_fit(key, predict, f_true, ground_truth, particles)
        axes[0].set_title('SMC particles', fontsize=16)
        axes[0].set_ylabel('Latent GP', rotation=0, ha='right', fontsize=14)
        axes[1].set_title('Posterior 95% HDI', fontsize=16)
        plt.show()

    def plot_num(self):
        self._plot_num()
        plt.xlabel("Amount of change points")
        plt.ylabel("Amount of particles")
        plt.title('Amount of Change Points in Latent GP')
        plt.legend()
        plt.show()

    def plot_post(self, ground_truth=None):
        ''' Only plots up to a maximum of 5 posteriors per default'''
            
        if isinstance(self.particles.particles['kernel'], dict):
            isdict = True
            num_kernels = 1
            # num_particles = particles.particles['kernel'][trainables[0]].shape[0]
            # num_CPs = jnp.max(jnp.sum(~jnp.isnan(particles.particles['kernel']['num']), axis = 1))
        else:
            isdict = False
            num_kernels = len(self.particles.particles['kernel'])
            # num_particles = particles.particles['kernel'][0][trainables[0]].shape[0]
        for k in range(num_kernels):
            if isdict:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel']['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            else:
                num_CPs = jnp.max(jnp.sum(~jnp.isnan(self.particles.particles['kernel'][k]['num']), axis = 1))
                tr = copy.deepcopy(self.particles.particles['kernel'][k])
                del tr['num'] 
                trainables = [name for name in tr.keys()]
            if trainables == []:
                raise ValueError(
                    f'No posteriors to plot!')
            
            num_params = len(trainables)

            symbols = [fr'{name[0]}' for name in trainables]
            
            num_CP = jnp.minimum(num_CPs, 5).tolist()
            _, axes = plt.subplots(nrows=num_params, ncols=num_CP+1, constrained_layout=True,
                                figsize=(16, 6))
            
            if num_CP == 0:
                axes = axes[:, jnp.newaxis]
            elif num_params == 1:
                axes = axes[:, jnp.newaxis].T

            for j, var in enumerate(trainables):
                    pd = tr[var]
                    for i in range(num_CP+1):
                        # There are some outliers that skew the axes
                        # pd_u, pd_l = jnp.nanpercentile(pd[:, i], q=99.9), jnp.nanpercentile(pd[:, i], q=0.1)
                        # pd_filtered = jnp.extract(pd[:, i]>pd_l, pd[:, i])
                        # pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
                        axes[j, i].hist(pd[~jnp.isnan(pd[:, i]), i], bins=30, density=True, color='tab:blue')
                        # axes[j, i].hist(pd[:, i][~jnp.isnan(pd[:, i])], bins=30, density=True, color='tab:blue')
                        if ground_truth is not None:
                            if isdict:
                                if len(ground_truth['kernel'][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][var][i], ls=':', c='k')
                            else:
                                if len(ground_truth['kernel'][k][var]) > i:
                                    axes[j, i].axvline(x=ground_truth['kernel'][k][var][i], ls=':', c='k')
                        
                        axes[j, i].set_xlabel(r'${:s}$'.format(f'{symbols[j]}_{i}'))
                    # for i in range(num_CP+1):
                        

                    axes[j, 0].set_ylabel(var, rotation=0, ha='right')
                
            plt.suptitle(f'Posterior estimate of Bayesian Latent GP {self.kernel_name[k]} kernel ({self.num_particles} particles)')
            plt.show();

    

