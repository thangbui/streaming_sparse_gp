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

class OSGPR_VFE(GPModel):
    """
    Online Sparse Variational GP regression.
    
    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self, X, Y, kern, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects
        mu_old, Su_old are mean and covariance of old q(u)
        Z_old is the old inducing inputs
        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

        self.mu_old = DataHolder(mu_old, on_shape_change='pass')
        self.M_old = Z_old.shape[0]
        self.Su_old = DataHolder(Su_old, on_shape_change='pass')
        self.Kaa_old = DataHolder(Kaa_old, on_shape_change='pass')
        self.Z_old = DataHolder(Z_old, on_shape_change='pass')

    def _build_common_terms(self):
        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = self.kern.K(self.Z, self.X)
        Kbb = self.kern.K(self.Z) + tf.eye(Mb, dtype=float_type) * jitter
        Kba = self.kern.K(self.Z, self.Z_old)
        Kaa_cur = self.kern.K(self.Z_old) + tf.eye(Ma, dtype=float_type) * jitter
        Kaa = self.Kaa_old + tf.eye(Ma, dtype=float_type) * jitter

        err = self.Y - self.mean_function(self.X)

        Sainv_ma = tf.matrix_solve(Saa, ma)
        Sinv_y = self.Y / sigma2
        c1 = tf.matmul(Kbf, Sinv_y)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2

        Lb = tf.cholesky(Kbb)
        Lbinv_c = tf.matrix_triangular_solve(Lb, c, lower=True)
        Lbinv_Kba = tf.matrix_triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.matrix_triangular_solve(Lb, Kbf, lower=True) / sigma
        d1 = tf.matmul(Lbinv_Kbf, tf.transpose(Lbinv_Kbf))

        LSa = tf.cholesky(Saa)
        Kab_Lbinv = tf.transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.matrix_triangular_solve(
            LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(tf.transpose(LSainv_Kab_Lbinv), LSainv_Kab_Lbinv)

        La = tf.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.matrix_triangular_solve(
            La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(tf.transpose(Lainv_Kab_Lbinv), Lainv_Kab_Lbinv)

        D = tf.eye(Mb, dtype=float_type) + d1 + d2 - d3
        D = D + tf.eye(Mb, dtype=float_type) * jitter
        LD = tf.cholesky(D)

        LDinv_Lbinv_c = tf.matrix_triangular_solve(LD, Lbinv_c, lower=True)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_Lbinv_c, err, d1)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        jitter = settings.numerics.jitter_level
        # jitter = 1e-4
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = self.kern.Kdiag(self.X)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._build_common_terms()

        LSa = tf.cholesky(Saa)
        Lainv_ma = tf.matrix_triangular_solve(LSa, ma, lower=True)

        bound = 0
        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # log det term
        bound += -0.5 * N * tf.reduce_sum(tf.log(sigma2))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LD)))

        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.diag_part(Qff))

        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.log(tf.diag_part(La)))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LSa)))

        Kaadiff = Kaa_cur - tf.matmul(tf.transpose(Lbinv_Kba), Lbinv_Kba)
        Sainv_Kaadiff = tf.matrix_solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.matrix_solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.diag_part(Sainv_Kaadiff) - tf.diag_part(Kainv_Kaadiff))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # jitter = settings.numerics.jitter_level
        jitter = 1e-4

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = self.kern.K(self.Z, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
            Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._build_common_terms()

        Lbinv_Kbs = tf.matrix_triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.matrix_triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_Lbinv_c)

        if full_cov:
            Kss = self.kern.K(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=float_type)
            var1 = Kss
            var2 = - tf.matmul(tf.transpose(Lbinv_Kbs), Lbinv_Kbs)
            var3 = tf.matmul(tf.transpose(LDinv_Lbinv_Kbs), LDinv_Lbinv_Kbs)
            var = var1 + var2 + var3
        else:
            var1 = self.kern.Kdiag(Xnew)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), 0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), 0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var


class OSGPR_PEP(GPModel):
    """
    Online Sparse GP regression using Power-EP.

    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self, X, Y, kern, mu_old, Su_old, Kaa_old, Z_old, Z, alpha, 
            mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects
        mu_old, Su_old are mean and covariance of old q(u)
        Z_old is the old inducing inputs

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
        
        self.mu_old = DataHolder(mu_old, on_shape_change='pass')
        self.M_old = Z_old.shape[0]
        self.Su_old = DataHolder(Su_old, on_shape_change='pass')
        self.Kaa_old = DataHolder(Kaa_old, on_shape_change='pass')
        self.Z_old = DataHolder(Z_old, on_shape_change='pass')

    def _build_common_terms(self):
        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        alpha = self.alpha

        Saa = self.Su_old
        ma = self.mu_old
        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kfdiag = self.kern.Kdiag(self.X)
        Kbf = self.kern.K(self.Z, self.X)
        Kbb = self.kern.K(self.Z) + tf.eye(Mb, dtype=float_type) * jitter
        Kba = self.kern.K(self.Z, self.Z_old)
        Kab = tf.transpose(Kba)
        Kaa_cur = self.kern.K(self.Z_old) + tf.eye(Ma, dtype=float_type) * jitter
        Kaa = self.Kaa_old + tf.eye(Ma, dtype=float_type) * jitter

        err = self.Y - self.mean_function(self.X)
        Lb = tf.cholesky(Kbb)
        Lbinv_Kbf = tf.matrix_triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, tf.transpose(Lbinv_Kbf_LDff))

        Lbinv_Kba = tf.matrix_triangular_solve(Lb, Kba, lower=True)
        Kab_Lbinv = tf.transpose(Lbinv_Kba)
        Sainv_Kab_Lbinv = tf.matrix_solve(Saa, Kab_Lbinv)
        Kainv_Kab_Lbinv = tf.matrix_solve(Kaa, Kab_Lbinv)
        Da_Kab_Lbinv = Sainv_Kab_Lbinv - Kainv_Kab_Lbinv
        d2 = tf.matmul(Lbinv_Kba, Da_Kab_Lbinv)

        Kaadiff = Kaa_cur - tf.matmul(Kab_Lbinv, Lbinv_Kba)
        LM = tf.cholesky(Kaadiff)
        LMT = tf.transpose(LM)
        Sainv_LM = tf.matrix_solve(Saa, LM)
        Kainv_LM = tf.matrix_solve(Kaa, LM)
        SK_LM = Sainv_LM - Kainv_LM
        LMT_SK_LM = tf.matmul(LMT, SK_LM)
        Q = tf.eye(Ma, dtype=float_type) + alpha * LMT_SK_LM
        LQ = tf.cholesky(Q)

        LMT_Da_Kab_Lbinv = tf.matmul(LMT, Da_Kab_Lbinv)
        Qinv_t1 = tf.matrix_solve(Q, LMT_Da_Kab_Lbinv)
        t1_Qinv_t1 = tf.matmul(tf.transpose(LMT_Da_Kab_Lbinv), Qinv_t1)
        d3 = - alpha * t1_Qinv_t1

        D = tf.eye(Mb, dtype=float_type) + d1 + d2 + d3
        D = D + tf.eye(Mb, dtype=float_type) * jitter
        LD = tf.cholesky(D)

        Sainv_ma = tf.matrix_solve(Saa, ma)
        LMT_Sainv_ma = tf.matmul(LMT, Sainv_ma)
        Lbinv_Kba_Da = tf.transpose(Da_Kab_Lbinv)
        Lbinv_Kba_Da_LM = tf.matmul(Lbinv_Kba_Da, LM)
        Qinv_LMT_Sainv_ma = tf.matrix_solve(Q, LMT_Sainv_ma)
        Sinv_y = self.Y / tf.reshape(Dff, [self.num_data, 1])
        c1 = tf.matmul(Lbinv_Kbf, Sinv_y)
        c2 = tf.matmul(Lbinv_Kba, Sainv_ma)
        c3 = - alpha * tf.matmul(Lbinv_Kba_Da_LM, Qinv_LMT_Sainv_ma)
        c = c1 + c2 + c3

        LDinv_c = tf.matrix_triangular_solve(LD, c, lower=True)
        LSa = tf.cholesky(Saa)
        La = tf.cholesky(Kaa)

        return (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_c, err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        alpha = self.alpha
        Saa = self.Su_old
        ma = self.mu_old
        N = self.num_data

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_c,
            err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM) = self._build_common_terms()

        Lainv_ma = tf.matrix_triangular_solve(LSa, ma, lower=True)

        bound = 0
        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))
        ma_Sainv_LM = tf.matmul(tf.transpose(Sainv_ma), LM)
        Qinv_LM_Sainv_ma = tf.matrix_solve(Q, tf.transpose(ma_Sainv_LM))
        bound += 0.5 * alpha * \
            tf.reduce_sum(tf.matmul(ma_Sainv_LM, Qinv_LM_Sainv_ma))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.log(Dff))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LD)))

        # delta 1: trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.log(Dff / sigma2))

        # delta 2
        bound += - 1.0 / alpha * tf.reduce_sum(tf.log(tf.diag_part(LQ)))
        bound += tf.reduce_sum(tf.log(tf.diag_part(La)))
        bound += - tf.reduce_sum(tf.log(tf.diag_part(LSa)))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # jitter = settings.numerics.jitter_level
        jitter = 1e-4

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = self.kern.K(self.Z, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_c,
            err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM) = self._build_common_terms()

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
            var1 = self.kern.Kdiag(Xnew)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), 0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), 0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var
