import numpy as np
import GPy
from GPy import kern
from GPy.core.model import Model
from GPy.core.mapping import Mapping
from GPy import likelihoods
import logging
import warnings

def my_predictive_gradient(Xnew, GPmodel, test_point, kern=None):
        """
        Returns gradients of predictive mean, variance, and covariance
        """
        if kern is None:
            kern = GPmodel.kern
        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1],GPmodel.output_dim))

        for i in range(GPmodel.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(GPmodel.posterior.woodbury_vector[:,i:i+1].T, Xnew, GPmodel._predictive_variable)

        # gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        alpha = -2.*np.dot(kern.K(Xnew, GPmodel._predictive_variable), GPmodel.posterior.woodbury_inv)
        dv_dX += kern.gradients_X(alpha, Xnew, GPmodel._predictive_variable)
        
        # Gradient w.r.t. covariance
        dC_dX = kern.gradients_X(np.eye(Xnew.shape[0]), Xnew, test_point.reshape(1,test_point.size))
        alpha2 = -np.dot(kern.K(test_point.reshape(1,test_point.size), GPmodel._predictive_variable), GPmodel.posterior.woodbury_inv)
        dC_dX += kern.gradients_X(alpha2, Xnew, GPmodel._predictive_variable)
        
        return mean_jac, dv_dX - 2.*dC_dX
        # return mean_jac, dv_dX