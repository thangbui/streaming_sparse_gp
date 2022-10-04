import tensorflow as tf
import numpy as np

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow import covariances


class SGPR_PEP(GPModel, InternalDataTrainingLossMixin):
    """
    Sparse GP regression using Power-EP.

    A unifying framework for Gaussian process pseudo-point
    approximations using Power Expectatin Propagation
    Thang D. Bui, Josiah Yan, Richard E. Turner
    JMLR 2017
    
    """

    def __init__(self, data, kernel, Z, alpha, mean_function=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects

        This method only works with a Gaussian likelihood.
        """
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.alpha = alpha

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self.likelihood.variance
        alpha = self.alpha

        # b is inducing points
        # f is training points
        Kfdiag = self.kernel(self.X, full_cov=False)
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)

        err = self.Y - self.mean_function(self.X)
        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, Lbinv_Kbf_LDff, transpose_b=True)
        D = tf.eye(Mb, dtype=d1.dtype) + d1
        LD = tf.linalg.cholesky(D)

        Sinv_y = self.Y / tf.reshape(Dff, [self.num_data, 1])
        c = tf.matmul(Lbinv_Kbf, Sinv_y)
        LDinv_c = tf.linalg.triangular_solve(LD, c, lower=True)

        return (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff)

    def maximum_log_likelihood_objective(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        sigma2 = self.likelihood.variance
        N = self.num_data
        alpha = self.alpha

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.math.log(Dff))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.math.log(Dff / sigma2))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)

        # b is inducing points
        # f is training points
        # s is test points
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False) + jitter
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var
