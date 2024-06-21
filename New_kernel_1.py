import jax
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jrnd
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional
from jaxtyping import Array, Float
from jax.nn import softmax
from jaxkern.computations import (
    DenseKernelComputation,
)

from typing import Dict, List, Optional

class Brownian(jk.base.AbstractKernel):

    def __init__(self) -> None:
        pass

    #
    def __call__(self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        # see https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/brownian.py
        n_x = x.shape[0]
        n_y = y.shape[0]
        x_mat = jnp.tile(jnp.squeeze(x), (n_y, 1))
        y_mat = jnp.tile(jnp.squeeze(y), (n_x, 1)).T        
        return (params['variance'] * jnp.where(jnp.sign(x_mat)==jnp.sign(y_mat), jnp.fmin(jnp.abs(x_mat), jnp.abs(y_mat)), 0)).T

    #
    def init_params(self, key: Array) -> dict:
        super().init_params(key)

    #
#

class SpectralMixture(jk.base.AbstractKernel):

    def __init__(self) -> None:
        # Note: we don't want to inherit here.
        pass
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def __euclidean_distance_einsum(self, X, Y):
        """Efficiently calculates the euclidean distance
        between two vectors using Numpys einsum function.

        Parameters
        ----------
        X : array, (n_samples x d_dimensions)
        Y : array, (n_samples x d_dimensions)

        Returns
        -------
        D : array, (n_samples, n_samples)
        """
        XX = jnp.einsum('ij,ij->i', X, X)[:, jnp.newaxis]
        YY = jnp.einsum('ij,ij->i', Y, Y)
        XY = 2 * jnp.dot(X, Y.T)
        return  XX + YY - XY

    #

    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The spectral mixture kernel is defined as

        .. math::

            \mu_q ~ N(.,.), for q = 1..Q
            log \nu_q ~ N(., .), for q = 1..Q
            beta_q ~ N(., .), for q = 2..Q
            w = softmax_centered(beta)
            k(tau) = \sum_{q=1}^Q w_q \prod_{i=1}^D \exp[-2pi^2 tau_i^2 \nu_q^({i})] cos(2pi tau_i \mu_q^{(i)}),

            with tau = x - y.        

        Importantly, we enforce identifiability of the posterior of these 
        parameters in two ways. First, w is drawn from a centered softmax, which
        ensures w_q > 0 and \sum w_q = 1, but in addition the weights are 
        anchored around the first element which is always forced to zero (i.e. 
        we sample only beta_2, ..., beta_Q, and set beta_1 = 0). Second, we sort
        the vector of means so that the smallest frequency component is always
        the first.

        This does not yet work in higher dimensions, as the sorting needs to be 
        defined there.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an n x m matrix of cross covariances (n = len(x), m = len(y))
        """

        def compsum(res, el):
            w_, mu_, nu_ = el
            res = res + w_ * jnp.exp(-2*jnp.pi**2 * tau**2 * nu_) * jnp.cos(2*jnp.pi * tau * mu_)
            return res, el

        #
        
        tau = jnp.sqrt(self.__euclidean_distance_einsum(x, y))
        beta = params['beta']
        w = softmax(jnp.insert(beta, 0, 0))
        mu = params['mu']
        # To solve the identifiability issue in mixture models, we sort according to the means:
        mu = jnp.sort(mu)
        nu = params['nu']     

        K, _ = jax.lax.scan(compsum, jnp.zeros((x.shape[0], y.shape[0])), (w, mu, nu))        
        return K

    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

#


class Discontinuous(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0 = []) -> None:
        self.base_kernel = base_kernel
        self.x0 = x0
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        def check_side(x_, y_):
           return 1.0*jnp.logical_or(jnp.logical_and(jnp.less(x_, self.x0), 
                                                     jnp.less(y_, self.x0)), 
                                     jnp.logical_and(jnp.greater_equal(x_, self.x0), 
                                                     jnp.greater_equal(y_, self.x0)))

        
        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

class Discontinuous_multiple_fixed(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0 = []) -> None:
        self.base_kernel = base_kernel
        self.x0 = x0
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        # def check_side(x_, y_):
        #     return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        def check_side(x_, y_):
            return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
                
        
        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)


class Discontinuous_multiple(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        # super().__init__(
        #     DenseKernelComputation,
        #     active_dims,
        #     # spectral_density=dx.Normal(loc=0.0, scale=1.0),
        #     # name=name,
        # )
        self.base_kernel = base_kernel
        self.temp = temp
        self._stationary = True        
    #
    

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # x_ = self.slice_input(x) 
        # y_ = self.slice_input(y) 
        # return self.check_side(params, x_, y_)*self.base_kernel.cross_covariance(params, x, y).squeeze()
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        # def check_side(x_, y_):
        #     return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        def check_side(params, x_, y_):
            return 1.0*(jnp.sum(jnp.less(x_, params["num"])) == jnp.sum(jnp.less(y_, params["num"]))) 
                
        
        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(params, x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze()) + self.temp*jnp.eye(x.shape[0], y.shape[0])
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

class Discontinuous_multiple_unknown(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel) -> None:
        self.base_kernel = base_kernel
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        #def check_side(x_, y_):
        #    return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        def check_zero(x):
            return jnp.where(x==0, jnp.nan, x)

        def check_smaller_num(x):
          return 1.0*jnp.less(x, params['num'])

        num_list = jnp.sort(jnp.flip(jnp.arange(len(params["num"]))))
        mask2 = jax.vmap(lambda x_: check_smaller_num(x_))(num_list) 
        num = jnp.multiply(params["num"], mask2)

        new_num = jax.vmap(lambda x: check_zero(x))(num)
        
        def check_side(x_, y_):
          return 1.0*(jnp.sum(jnp.less(x_, new_num)) == jnp.sum(jnp.less(y_, new_num)))      


        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #


class Discontinuous_Dirichlet(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel) -> None:
        self.base_kernel = base_kernel
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        #def check_side(x_, y_):
        #    return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        theta, count = params['num'][:, 3], params['num'][0, 4]

        def check_zero(x):
            return jnp.where(x==0, jnp.nan, x)

        def check_smaller_num(x):
          return 1.0*jnp.less(x, count)
        

        num_list = jnp.sort(jnp.flip(jnp.arange(len(theta))))

        mask2 = jax.vmap(lambda x_: check_smaller_num(x_))(num_list) 
        num = jnp.multiply(theta, mask2)

        new_num = jax.vmap(lambda x: check_zero(x))(num)
        
        def check_side(x_, y_):
          return 1.0*(jnp.sum(jnp.less(x_, new_num)) == jnp.sum(jnp.less(y_, new_num)))      


        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)


class Discontinuous_Poisson(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel) -> None:
        self.base_kernel = base_kernel
        self._stationary = True
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        #def check_side(x_, y_):
        #    return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        # theta, count = params['num'], jnp.count_nonzero()

        # def check_zero(x):
        #     return jnp.where(x==0, jnp.nan, x)

        # def check_smaller_num(x):
        #   return 1.0*jnp.less(x, count)
        

        # num_list = jnp.sort(jnp.flip(jnp.arange(len(theta))))

        # mask2 = jax.vmap(lambda x_: check_smaller_num(x_))(num_list) 
        # CP = jnp.multiply(theta, mask2)   

        # new_CP = jax.vmap(lambda x: check_zero(x))(CP)
        # new_CP = CP.at[CP == 0].set(jnp.nan)
        
        def check_side(x_, y_):
          return 1.0*(jnp.sum(jnp.less(x_, params["num"])) == jnp.sum(jnp.less(y_, params["num"])))      


        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

class Discontinuous_multiple_params(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0 = []) -> None:
        self.base_kernel = base_kernel
        # self.x0 = x0
        self._stationary = True
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        # def check_side(x_, y_):
        #     return 1.0*(jnp.sum(jnp.less(x_, self.x0)) == jnp.sum(jnp.less(y_, self.x0))) 
        # def check_side_mult(x_, y_, xs, l, v):
        #     # print(((jnp.less(x_, params["CP"][xs])) & jnp.less(y_, params["CP"][xs])).shape)
        #     return 1.0*((jnp.sum(jnp.less(x_, params["CP"])) == xs) & (jnp.sum(jnp.less(y_, params["CP"])) == xs))  

        def returnxcp(xcp, params, x_, y_):
            params = dict(lengthscale = params['lengthscale'][xcp],
                          variance = params['variance'][xcp])
            
            cov = jnp.squeeze(self.base_kernel.cross_covariance(params, jnp.array([x_]), jnp.array([y_])))
            return cov
        
        def zero_func(xcp, params, x_, y_):
            return 0.

        def check_side_mult(x_, y_, params):
            # print(((jnp.less(x_, params["CP"][xs])) & jnp.less(y_, params["CP"][xs])).shape)
            xcp = jnp.sum(jnp.greater(x_, params["num"]))
            ycp = jnp.sum(jnp.greater(y_, params["num"]))
            
            val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
            
            return val
        

        
        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
    

        # dict2 = {'x': x, 
        #          'y': y}
        # carry = params | dict2

        # print(params)
        # indexer = jnp.arange(0, params['lengthscale'].shape[0], dtype = int)
        # carry, K = jax.lax.scan(mult_cov, carry, indexer)


        return K
        
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)


class Discontinuous_multiple_params_hyper(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'RBF'
    
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """

        '''New version'''

        # jax.debug.breakpoint()
        
        
        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
        # jax.debug.print('val {}', K)

        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])

        return new_K
        
    def check_side_mult(self, x_, y_, params: Dict):

        
        def returnxcp(xcp, params, x_, y_):
            params = dict(lengthscale = params['lengthscale'][xcp],
                          variance = params['variance'][xcp])
            
            cov = jnp.squeeze(self.base_kernel.cross_covariance(params, jnp.array([x_]), jnp.array([y_])))
            return cov
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        # jax.debug.print('params{}', params)
        xcp = jnp.sum(jnp.greater(x_, params['num']))
        ycp = jnp.sum(jnp.greater(y_, params['num']))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        return val
    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

class Discontinuous_multiple_params_hyper_periodic(jk.base.AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'Periodic'

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -0.5 \\sum_{i=1}^{d} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        '''New version'''
        
        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
        return new_K
    
    def check_side_mult(self, x_, y_, params):

        def returnxcp(xcp, params, x, y):
            new_params = dict(lengthscale = params['lengthscale'][xcp],
                          variance = params['variance'][xcp],
                          period = params['period'][xcp])
            sine_squared = (jnp.sin(jnp.pi * (x - y) / new_params["period"]) / new_params["lengthscale"]
                            ) ** 2
            cov = new_params["variance"] * jnp.exp(-0.5 * sine_squared)
            return cov.squeeze()
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params["num"]))
        ycp = jnp.sum(jnp.greater(y_, params["num"]))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        
        return val
    
    def init_params(self, key: jrnd.KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "period": jnp.array([1.0] * self.ndims),
        }
    

class Discontinuous_multiple_params_hyper_linear(jk.base.AbstractKernel):
    """The periodic kernel.

    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    def __init__(self, base_kernel, x0 = [], temp = 0) -> None:
        self.base_kernel = base_kernel
        self.temp = temp
        self.x0 = x0
        self._stationary = True
        self.name = 'Linear'

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        if x.shape != ():
            return self.cross_covariance(params, x, y)
        if x.shape == ():
            return self.check_side_mult(x, y, params)
    
    def cross_covariance(self, params: Dict, x, y):
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance parameter :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 x^{T}y

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """

        K = jax.vmap(lambda x_, params: jax.vmap(lambda y_: self.check_side_mult(x_, y_, params))(y), in_axes=(0, None))(x, params)
        new_K = K + self.temp*jnp.eye(x.shape[0], y.shape[0])
        return new_K
    
    def check_side_mult(self, x_, y_, params):

        def returnxcp(xcp, params, x, y):
            new_params = dict(
                          variance = params['variance'][xcp],
                          )
            cov = new_params["variance"] * x * y
            return cov.squeeze()
        
        def zero_func(xcp, params, x_, y_):
            return 0.
        
        xcp = jnp.sum(jnp.greater(x_, params["num"]))
        ycp = jnp.sum(jnp.greater(y_, params["num"]))
        
        val = jax.lax.cond(xcp == ycp, returnxcp, zero_func, xcp, params, x_, y_)
        
        return val

    def init_params(self, key: jrnd.KeyArray) -> Dict:
        return {"variance": jnp.array([1.0])}