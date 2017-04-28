import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import Binomial

class Logit(sm.families.links.Logit):

    """Logit tranform that won't overflow with large numbers."""

    def inverse(self, z):
        return 1 / (1. + np.exp(-z))

class BB(GenericLikelihoodModel):
    
    """ `phi` as a precision parameter equal to `a + b` from the Beta parameters. """
    
    def __init__(self, endog, exog, Z=None, link=sm.families.links.Logit(), link_phi=sm.families.links.Log(), **kwds):
        """
        Parameters
        ----------
        endog : array-like
            2d array of endogenous values (i.e. responses, outcomes,
            dependent variables, or 'Y' values).
        exog : array-like
            2d array of exogeneous values (i.e. covariates, predictors,
            independent variables, regressors, or 'X' values). A nobs x k
            array where `nobs` is the number of observations and `k` is
            the number of regressors. An intercept is not included by
            default and should be added by the user. See
            `statsmodels.tools.add_constant`.
        Z : array-like
            2d array of variables for the precision phi.
        link : link
            Any link in sm.families.links for `exog`
        link_phi : link
            Any link in sm.families.links for `Z`
        """
        if Z is None:
            extra_names = ['phi']
            Z = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in \
                        (Z.columns if hasattr(Z, 'columns') else range(1, Z.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names
        
        super(BB, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_phi = link_phi
        
        self.Z = Z
        assert len(self.Z) == len(self.endog)
    
    def nloglikeobs(self, params):
            """
            Negative log-likelihood.

            Parameters
            ----------

            params : np.ndarray
                Parameter estimates
            """
            return -self._ll_br(self.endog, self.exog, self.Z, params)
    
    def fit(self, start_params=None, maxiter=100000, maxfun=5000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model.

        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        """

        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()).fit(disp=False).params
            start_params = np.append(start_params, [1.0] * self.Z.shape[1])
    
        return super(BB, self).fit(start_params=start_params,
                                        maxiter=maxiter, maxfun=maxfun,
                                        method=method, disp=disp, **kwds)
    
    def _ll_br(self, endog, exog, Z, params):
        k = endog[:,0]
        N = endog[:,1]
        X = exog
        
        nz = self.Z.shape[1]
        
        Xparams = params[:-nz]
        Zparams = params[-nz:]

        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_phi.inverse(np.dot(Z, Zparams))
        # TODO: derive a and b and constrain to > 0?

        if np.any(phi <= np.finfo(float).eps): return np.array(-np.inf)
        
        ll = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + gammaln(phi) - gammaln(mu * phi) \
             - gammaln(phi * (1 - mu)) + gammaln(k + phi * mu) + gammaln(N - k + phi * (1 - mu)) - gammaln(N + phi)

        return ll