from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from gpflow.model import GPModel
from gpflow.param import Param, DataHolder
from gpflow.mean_functions import Zero
from gpflow import likelihoods
from gpflow._settings import settings
from gpflow.densities import multivariate_normal
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class SGPR_PEP(GPModel):
    """
    Sparse GP regression using Power-EP.

    A unifying framework for Gaussian process pseudo-point
    approximations using Power Expectatin Propagation
    Thang D. Bui, Josiah Yan, Richard E. Turner
    JMLR 2017
    
    """

    def __init__(self, X, Y, kern, Z, alpha, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects

        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.alpha = alpha

    def _build_common_terms(self):
        Mb = tf.shape(self.Z)[0]
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        sigma2 = self.likelihood.variance
        alpha = self.alpha

        # b is inducing points
        # f is training points
        Kfdiag = self.kern.Kdiag(self.X)
        Kbf = self.kern.K(self.Z, self.X)
        Kbb = self.kern.K(self.Z) + tf.eye(Mb, dtype=float_type) * jitter

        err = self.Y - self.mean_function(self.X)
        Lb = tf.cholesky(Kbb)
        Lbinv_Kbf = tf.matrix_triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, tf.transpose(Lbinv_Kbf_LDff))
        D = tf.eye(Mb, dtype=float_type) + d1
        LD = tf.cholesky(D)

        Sinv_y = self.Y / tf.reshape(Dff, [self.num_data, 1])
        c = tf.matmul(Lbinv_Kbf, Sinv_y)
        LDinv_c = tf.matrix_triangular_solve(LD, c, lower=True)

        return (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        sigma2 = self.likelihood.variance
        N = self.num_data
        alpha = self.alpha

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._build_common_terms()

        bound = 0
        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.log(Dff))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LD)))

        # trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.log(Dff / sigma2))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4

        # b is inducing points
        # f is training points
        # s is test points
        Kbs = self.kern.K(self.Z, Xnew)
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._build_common_terms()

        Lbinv_Kbs = tf.matrix_triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.matrix_triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_c)

        if full_cov:
            Kss = self.kern.K(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=float_type)
            var1 = Kss
            var2 = - tf.matmul(tf.transpose(Lbinv_Kbs), Lbinv_Kbs)
            var3 = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_Lbinv_Kbs)
            var = var1 + var2 + var3
        else:
            var1 = self.kern.Kdiag(Xnew) + jitter
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), 0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), 0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var
